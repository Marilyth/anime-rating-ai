# For GPU, install CUDA toolkit version used by tensorflow (v11.2 atm): https://developer.nvidia.com/cuda-toolkit-archive
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import numpy as np
from random import seed
from training import BertTrainer
from data_analysis import rating_distribution, random_seed
from predictor import BertPredictor

def show_distribution():
    rating_distribution()

def create_model():
    trainer = BertTrainer()
    trainer.load_corpus()
    trainer.train_bert(test_size=2300, epochs=4)
    trainer.save_model_onnx()
    print("Done")

def use_model():
    predictor = BertPredictor()
    score = predictor.get_score("Dragon Ball Super Season 20", 1)
    print(score)

if __name__ == "__main__":
    #show_distribution()
    seed(random_seed)
    np.random.seed(random_seed)
    use_model()
