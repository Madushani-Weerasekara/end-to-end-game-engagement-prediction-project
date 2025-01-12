# all the training code over here
# How many different kinds of model i want to use
# Probobly will call here the confusion Matrix if solving a clasification problem, R squard value if a regression problem
 
import torch
import torch.nn as nn
import torch.optim as optim

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import consensus_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

        def forward(self):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3)
            
            return x
class EngageModel:
    def __init__(self, input_dim, output_dim, path="artifacts\\model.pth", learning_rate=1e-3):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model = NeuralNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0
        self.path =path
             



