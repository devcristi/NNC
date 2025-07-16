# MLP Neural Network in C

This project implements a simple Multi-Layer Perceptron (MLP) neural network from scratch in C, featuring:

* Dense layers with randomized weight and bias initialization
* ReLU activation for hidden layers
* Sigmoid activation for the output layer (binary classification)
* Forward propagation and backward propagation (backpropagation)
* Binary Cross-Entropy loss function
* Example training on the XOR problem

---

## Project Structure

```bash
project/
├── main.c                 # Main program: builds, trains, and tests the network on XOR
├── include/               # Header files
│   ├── neural_network.h   # Layer and NeuralNetwork structures and API
│   ├── relu.h             # ReLU activation prototype
│   ├── loss.h             # Sigmoid and Binary Cross-Entropy prototypes
│   ├── forward.h          # Forward propagation API
│   └── backpropagation.h  # Backpropagation API
├── src/                   # Source files
│   ├── neural_network.c   # Layer and network creation and cleanup
│   ├── relu.c             # ReLU implementation
│   ├── loss.c             # Sigmoid and BCE loss implementation
│   ├── forward.c          # Forward pass implementation
│   └── backpropagation.c  # Backpropagation and weight/bias updates
└── README.md              # This file
```

---

## Requirements

* GCC or any C99-compatible compiler
* GNU Make (optional)

---

## Build and Run

Open a terminal in the project root and execute:

```bash
# Compile all modules into main.exe
gcc main.c src/neural_network.c src/relu.c src/loss.c src/forward.c src/backpropagation.c -o main.exe

# Run the executable
./main.exe
```

---

## Example Output

```
Epoch 0, Loss: 0.416766
Epoch 1000, Loss: 0.058876
...
Testing network on XOR inputs:
Input: 0.0 0.0 -> Output: 0.0073
Input: 0.0 1.0 -> Output: 0.9959
Input: 1.0 0.0 -> Output: 0.9924
Input: 1.0 1.0 -> Output: 0.0073
```

---

## Next Steps

* Test on other datasets (Iris, flattened MNIST)
* Extend to CNN (convolutional layers, pooling)
* Save and load model parameters
* Experiment with hyperparameters (learning rate, epochs, architecture)

---

Built from scratch in C. Happy coding!
