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
             

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        logging.info("Training has started...")
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(x_train.size()[0])
            for i in range(0, x_train.size(), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x_train[indices].to(self.device), y_train[indices].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                # Save the model when loss is reduced
                train_accuracy = self.evaluate(x_train, y_train)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Training Accuracy: {train_accuracy:.4f}")
                if self.best_accuracy < train_accuracy:
                    self.best_accuracy = train_accuracy
                    self.save_model()
                logging.info("Training completed")

