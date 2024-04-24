
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNNPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNNPyTorch, self).__init__()
        
        # Define the layers
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Testing the network with some dummy data
input_size = 3
hidden_size = 5
output_size = 1
model = SimpleNNPyTorch(input_size, hidden_size, output_size)
dummy_input = torch.randn(10, input_size)  # Batch of 10 samples, each with 3 features
output = model(dummy_input)
output.shape
print (output)