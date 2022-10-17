# For GPU, install CUDA toolkit version used by tensorflow (v11.2 atm): https://developer.nvidia.com/cuda-toolkit-archive
# Optimally install CUDA before requirements.txt for GPU support.
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import numpy as np
from random import random, seed
from training import BertTrainer
from data_analysis import rating_distribution, random_seed
from predictor import BertPredictor
import sys
import torch

def show_distribution():
    rating_distribution()

def create_model():
    trainer = BertTrainer()
    trainer.load_corpus()
    trainer.train_bert(test_size=500, epochs=5)
    trainer.save_model()
    print("Done")

def use_model():
    predictor = BertPredictor()
    while True:
        title = input("Please enter an anime name: ")
        score = predictor.get_score(title, 0)
        print(f"{title} has a rating of {score:.2f}")

if __name__ == "__main__":
    #show_distribution()
    torch.manual_seed(random_seed)
    seed(random_seed)
    np.random.seed(random_seed)
    if "--train" in sys.argv:
        create_model()
    use_model()
