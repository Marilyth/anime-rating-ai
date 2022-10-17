import os 
import bert_processing
import numpy as np
import torch

class BertPredictor():
    """A training pipeline for a SoftBert neural network.
    """
    def __init__(self):
        self.model_dir = "models"
        self.model_file = "anime.pkl"
        self.load_model()

    def load_model(self):
        session = torch.load(os.path.join(self.model_dir, self.model_file))
        self.model = session
        self.model.cpu()

    def get_score(self, title, genre):
        title_length = len(title)
        tokenized_data = bert_processing.encode(bert_processing.tokenize([title]))

        input = torch.tensor(np.array([tokenized_data["ids"][0]]))
        custom_data = np.array([[title_length, genre]])
        input = torch.tensor(np.c_[input, custom_data])

        attention_mask = torch.tensor(np.array([tokenized_data["attention_masks"][0]]))

        prediction = self.model.forward(*(input, attention_mask))[0].item()
        return prediction
