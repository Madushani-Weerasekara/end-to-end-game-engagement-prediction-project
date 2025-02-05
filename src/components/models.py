import torch
import torch.nn as nn
import torch.optim as optim
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EngageModel:
    def __init__(self, input_dim, output_dim, path="artifacts\\model.pth", learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0
        self.path = path

    def train(self, X_train, y_train, epochs=10, batch_size=32):
         
        logging.info("training started")
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(X_train.size()[0])

            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices].to(self.device), y_train[indices].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            #  save the  model when loss is reduce
            train_accuracy = self.evaluateModel(X_train, y_train)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Training Accuracy: {train_accuracy:.4f}')
            if self.best_accuracy < train_accuracy:
                self.best_accuracy = train_accuracy
                self.save_model()

        logging.info("training completed")

    def evaluateModel(self, X_test, y_test, batch_size=32):
       
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, X_test.size(0), batch_size):
                batch_X = X_test[i:i+batch_size].to(self.device)
                batch_y = y_test[i:i+batch_size].to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        return accuracy

    def predict(self, X):
         
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)
        print(f"Model saved at {self.path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path, weights_only=True))
        self.model.to(self.device)
        print("Model loaded")


    def compute_confusion_matrix(self, X_test, y_test):
         
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
        
        y_true = y_test.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        return cm