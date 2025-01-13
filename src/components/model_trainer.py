# all the training code over here
# How many different kinds of model i want to use
# Probobly will call here the confusion Matrix if solving a clasification problem, R squard value if a regression problem
 
import torch
import torch.nn as nn
import torch.optim as optim

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import consensus_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
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

                self.optimizer.zero_grad() # Clear gradients from the previous step
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()  # Calculate gradients (how much each weight contributed to the error)
                self.optimizer.step() # Adjust weights/update weights to minimize the loss

                # Save the model when loss is reduced
                train_accuracy = self.evaluate(x_train, y_train)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Training Accuracy: {train_accuracy:.4f}")
                if self.best_accuracy < train_accuracy:
                    self.best_accuracy = train_accuracy
                    self.save_model()
                logging.info("Training completed")

            
    def evaluateModel(self, x_test, y_test, batch_size):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, x_test.size(0), batch_size):
                batch_x = x_test[i:i+batch_size].to(self.device)
                batch_y = y_test[i:i+batch_size].to(self.device)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total=+ batch_y.size(0)
                correct=+ (predicted == batch_y).sum().item()
        accuracy = correct/total
        return accuracy

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)
        print(f'Model saved at {self.path}')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.to(self.device)
        print("Model loaded")

    def compute_confusion_matrix(self, x_test, y_test):
        self.model.eval()
        with torch.no_grad():
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            outputs = self.model(x_test)
            _, predicted = torch.max(outputs, 1)

        y_actual = y_test.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        cm = confusion_matrix(y_actual, y_pred)

        # plot confusion metrics
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot(cmap=plt.cm.Blues)
        plt.show()
        return cm



