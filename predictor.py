from lib2to3.pgen2 import token
import os
import onnxruntime 
import bert_processing
import numpy as np

class BertPredictor():
    """A training pipeline for a SoftBert neural network.
    """
    def __init__(self):
        self.model_dir = "models"
        self.model_file = "torch.onnx"
        self.load_model_onnx()

    def load_model_onnx(self):
        # Load onnx model in GPU if available, else CPU.
        session = onnxruntime.InferenceSession(os.path.join(self.model_dir, self.model_file), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model = session

    def get_score(self, title, genre):
        title_length = len(title)
        tokenized_data = bert_processing.encode(bert_processing.tokenize([title]))

        input = tokenized_data["ids"][0]
        input = np.c_[input, np.array([title_length, genre])]

        attention_mask = np.array(tokenized_data["attention_mask"][0])

        prediction = self.model.run(output_names=None, input_feed={"input_ids": input, "input_mask": attention_mask})[0]
        return prediction
