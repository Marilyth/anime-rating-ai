from sklearn.model_selection import train_test_split
import bert_processing
from data_analysis import random_seed, load_data
import torch
import numpy as np
from custom_bert import CustomBert
from torch.optim.adamw import AdamW
from pytorch_transformers.optimization import WarmupLinearSchedule
import os
import onnxruntime 
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pathlib import Path

class BertTrainer():
    """A training pipeline for a SoftBert neural network.
    """
    def __init__(self):
        """Initializes the BertTrainer.

        Args:
            data_directory (str, optional): The location of the grouped fda data. Defaults to "./Downloads/HierarchicalData".
            parent_node (str, optional): The node for which to learn to classify the children for. Defaults to "A".
        """
        self._raw_data = None
        self._labels = None
        self._texts = None
        self._tokenized_data = None
        self._example_input = None
        self._processed_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = "models"
        self.model_file = f"torch.onnx"

    def load_corpus(self):
        """Loads and encodes the corpus for the current parent_node.

        Args:
            max_samples_per_class (int): Downsamples each class to have a maximum of this amount of datapoints.
        """
        # Undersample classes to be less imbalanced. Oversampling is difficult with text.
        self._raw_data = load_data().to_numpy()
        self._labels = [code for code in self._raw_data[:,2]]
        self._texts = self._raw_data[:,0]
        self._tokenize_texts()
    
    def _tokenize_texts(self):
        self._tokenized_data = bert_processing.encode(bert_processing.tokenize(self._texts))

    def _setup_split(self, test_size):
        ids = self._tokenized_data["ids"]
        # Append additional feature columns to ids.
        inputs: np.ndarray = np.c_[ids, self._raw_data[:, 1], self._raw_data[:, 3] * 0].astype(np.int32)

        masks = self._tokenized_data["attention_masks"]
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(inputs, self._labels, random_state=random_seed, test_size=test_size)
        train_masks, validation_masks, _, _ = train_test_split(masks, ids, random_state=random_seed, test_size=test_size)

        # Minimize GPU VRAM load with proper dtype. Increases possible batch size.
        # BERT base has a vocabulary of ~30k wordpieces, should fit into 16 bits, but pretrained BERT expects 32.
        train_inputs = torch.tensor(data=train_inputs, dtype=torch.int32)
        validation_inputs = torch.tensor(data=validation_inputs, dtype=torch.int32)

        # Labels per model don't exceed 27, fits into 8 bits.
        train_labels = torch.tensor(data=train_labels, dtype=torch.float16)
        validation_labels = torch.tensor(data=validation_labels, dtype=torch.float16)
        
        # Masks are 0 or 1, fits into 8 bits.
        train_masks = torch.tensor(data=train_masks, dtype=torch.uint8)
        validation_masks = torch.tensor(data=validation_masks, dtype=torch.uint8)

        # Higher does not fit into 4GB.
        batch_size = 32
        self.train_data = TensorDataset(train_inputs, train_masks, train_labels)
        self.train_sampler = RandomSampler(self.train_data)
        self.train_dataloader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=batch_size)
        self.validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        self.validation_sampler = SequentialSampler(self.validation_data)
        self.validation_dataloader = DataLoader(self.validation_data, sampler=self.validation_sampler, batch_size=batch_size // 2)

    def train_bert(self, test_size, epochs=4):
        """Trains SoftBert for the specified amount of epochs.

        Args:
            test_size (int): The size of the test data.
            epochs (int, optional): The amount of times to iterate over the train data for gradient descent. 
                Google suggests 2 - 4 epochs for optimal results. Defaults to 4.
        """
        self._setup_split(test_size)
        
        model: CustomBert = CustomBert.from_pretrained("bert-base-uncased", num_labels=len(set(self._labels)), num_custom_features=2)
        self.model = model

        # Allows 4GB GPU to reach the recommended 32 batch size.
        model.half()

        if self.device.type == "cuda":
            model.cuda()
        
        total_steps = len(self.train_dataloader) * epochs
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-4) # epsilon default 1e-8 is rounded to 0 for half dtype.
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=total_steps)  # PyTorch scheduler

        train_loss_set = []
        past_loss = 0

        for epoch in range(epochs):  
            # Set our model to training mode to compute gradients.
            model.train()  
            batches = len(self.train_dataloader)

            # Tracking variables for our training loss.
            total_loss = 0
            number_training_steps = 0

            # Train the data for one epoch
            for batch in tqdm(self.train_dataloader, total=batches, desc=f"Epoch {epoch + 1}/{epochs}"):
                # Add batch to GPU if available.
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack the inputs from our dataloader.
                b_inputs, b_mask, b_labels = batch

                # Clear out the gradients from the previous batch.
                optimizer.zero_grad()

                # Forward pass.
                loss, predictions = model(b_inputs, b_mask, labels=b_labels)

                train_loss_set.append(loss.item())    

                # Backward pass.
                loss.backward()

                # Update parameters and take a step using the computed gradient
                optimizer.step()
                scheduler.step()
                
                # Update tracking variables
                total_loss += loss.item()
                number_training_steps += 1

            print("Train loss: {}".format(total_loss/number_training_steps))
            
            self.model.eval()
            loss = self.validate_model()
            print(f"{loss} ({loss - past_loss} increase)")
            past_loss = loss

    def save_model_onnx(self):
        self.model.eval()
        self.model.float()
        example_loader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=8)
        example_input, example_mask, example_label = tuple(t.to(self.device) for t in next(iter(example_loader)))

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        torch.onnx.export(self.model, (example_input, example_mask), os.path.join(self.model_dir, self.model_file), opset_version=12, input_names=['input_ids', 'input_mask'],
                          output_names=['output'], dynamic_axes={
                                                                    'input_ids': {0: 'batch_size'},
                                                                    'input_mask': {0: 'batch_size'},
                                                                    'output': {0: 'batch_size'},
                                                                })

        del example_input, example_mask, example_label, example_loader

        # Done, but assert the models are equal.
        acc_before = self.validate_model()
        self.load_model_onnx()
        acc_after = self.validate_model()

        assert abs(acc_before - acc_after) <= 0.01

    def load_model_onnx(self):
        # Load onnx model in GPU if available, else CPU.
        session = onnxruntime.InferenceSession(os.path.join(self.model_dir, self.model_file), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model = session

    def validate_model(self):
        """Returns the accuracy of self.model against the current validation dataset.

        Returns:
            float: The accuracy.
        """
        # Function to calculate the accuracy of our predictions vs labels.
        values = []
        labels = []

        # Evaluate data for one epoch
        for batch in self.validation_dataloader:
            # Add batch to GPU if available.
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader.
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate predictions.
                if isinstance(self.model, onnxruntime.InferenceSession):
                    values.append(torch.tensor(self.model.run(output_names=None, input_feed={"input_ids": b_input_ids.detach().cpu().numpy(), "input_mask": b_input_mask.detach().cpu().numpy()})[0]))
                else:
                    value = self.model(*(b_input_ids, b_input_mask))
                    # Move predictions and labels to CPU.
                    values.append(value.detach().cpu())
            labels.append(b_labels.to('cpu'))
        
        mean_squared_loss = 0
        data_points = 0
        for i in range(len(values)):
            for j in range(len(values[i])):
                error = (values[i][j] - labels[i][j]).item()
                mean_squared_loss += error ** 2
                data_points += 1

        return mean_squared_loss / data_points
