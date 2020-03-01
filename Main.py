import numpy as np
import math

#Training input and outputs (if index [1] = 0, output is 1)
x = np.array(([1,1,1], [1,0,1], [0,1,1], [0,0,0]), dtype = float)
y = np.array(([0], [1], [0], [1]), dtype = float)

#Testing input and output to ensure it isn't fit to only the training data
test_x = np.array(([1,1,0], [1,0,0], [0,1,0], [0,0,1]), dtype = float)
test_y = np.array(([0], [1], [0], [1]), dtype = float)

#Defining the sigmoid and sigmoid-derivative functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

#Create the class NeuralNetwork
class NeuralNetwork:
    def __init__(self, x, y):
        #Take in the input values
        self.input = x
        #Random weights for the 4 hidden nodes
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        #Random weights going out of hidden layer
        self.weights2 = np.random.rand(4, 1)
        #Take in the output values
        self.y = y
        #Create an array to replace with the predicted values
        self.output = np.zeros(self.y.shape)
    
    #Define the feed-forward function
    def feedforward(self, input):
        #Calculate the values for each layer from input values
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        return self.output

    #Define the backpropagation function
    def backpropagation(self):
        #Calculate values for change in weights for each layer
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        #Adjust each layers' weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    #Define train function
    def train(self, x, y):
        #Calculate values for the input values
        self.output = self.feedforward(self.input)
        #Use those values and the real output values to backpropagate
        self.backpropagation()

#Create NeuralNetwork "nn"
nn = NeuralNetwork(x, y)
epochs = 100
#Train nn using training data "epochs" times
for i in range(epochs):
    #Show the epoch, input, predicted, actual and loss for specified number of iterations
    if i % 10 == 0:
        predicted = np.around(nn.feedforward(test_x))
        print("For iteration " + str(i) + "\n")
        print("Input: " + str(test_x) + "\n")
        print("Predicted: " + str(predicted) + "\n")
        print("Actual: " + str(test_y) + "\n")
        print("Loss: " + str(np.mean(np.square(test_y - nn.feedforward(test_x)))) + "\n")
    #Train nn
    nn.train(x, y)

