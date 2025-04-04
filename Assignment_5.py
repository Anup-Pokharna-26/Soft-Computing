import numpy as np

# Step 1: Define the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR output

# Step 2: Initialize neural network parameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.5

# Initialize weights and biases
W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
B1 = np.random.uniform(-1, 1, (1, hidden_size))
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
B2 = np.random.uniform(-1, 1, (1, output_size))

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Step 3: Train the neural network
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + B1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + B2
    final_output = sigmoid(final_input)
    
    # Compute error
    error = y - final_output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    B2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    B1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f'Epoch {epoch}, Loss: {loss}')

# Step 4: Test the trained model
print('\nFinal Predictions:')
test_output = sigmoid(np.dot(sigmoid(np.dot(X, W1) + B1), W2) + B2)
print(test_output)
