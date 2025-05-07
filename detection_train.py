import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


data = np.load('loss_data.npy')
labels = np.load('loss_labels.npy')


dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


class differential_detection(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(differential_detection, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)

input_size = 1576
hidden_size = 800
output_size = 8
learning_rate = 0.001
num_epochs = 200

model = differential_detection(input_size, hidden_size, output_size)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calculate_accuracy(outputs, labels):
    predictions = (outputs > 0.6).float()
    correct = (predictions == labels).float().sum()
    total = labels.numel()  
    return correct / total

best_accuracy = 0.0
best_model_path = "loss_train/epoch_200_lr_1e-4_derivate/best_model.pth"

for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0
    for batch_data, batch_labels in dataloader:
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs, batch_labels)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    
  
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"New best accuracy: {best_accuracy:.4f}. Model weights saved to {best_model_path}.")


with torch.no_grad():
    model.eval()
    test_output = model(data)
    accuracy = calculate_accuracy(outputs, labels)
