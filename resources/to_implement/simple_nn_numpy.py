
import numpy as np
import matplotlib.pyplot as plt

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Forward pass through the network
        self.hidden_layer = self.sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return self.output_layer

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            self.forward(x)

            # Compute the error
            error = y - self.output_layer

            # Backpropagation
            # Output layer to Hidden layer
            d_output = error * self.sigmoid_derivative(self.output_layer)
            error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(self.hidden_layer)

            # Update weights and biases
            self.weights_hidden_output += self.hidden_layer.T.dot(d_output) * learning_rate
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

            self.weights_input_hidden += x.T.dot(d_hidden_layer) * learning_rate
            self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


# Sample data: XOR function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Neural network parameters
input_size = 2
hidden_size = 5
output_size = 1
learning_rate = 0.5
epochs = 10000

# Initialize the neural network
nn = SimpleNN(input_size, hidden_size, output_size)

# Store error values for visualization
errors = []

# Training the network
for epoch in range(epochs):
    nn.train(X, Y, 1, learning_rate)
    output = nn.forward(X)
    error = np.mean(np.square(Y - output))
    errors.append(error)

# Visualizing the error over epochs
plt.figure(figsize=(12, 6))
plt.plot(errors)
plt.title("Error over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()

# Evaluate the trained network
final_output = nn.forward(X)
final_output