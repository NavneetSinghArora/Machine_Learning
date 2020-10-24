# Importing Libraries to carry out the Operations
import numpy
import matplotlib.pyplot as plt
# Importing features from the Libraries to carry out the Operations
from numpy import loadtxt


class LinearRegression:
    def __init__(self):
        # Training Examples File To be Loaded from a CSV/TXT file
        # Put Data Set Path
        data_path = r"/Users/***/Data Sets/Housing_Problem_Basic_data_Set.txt"
        data_file = open(data_path, 'r')
        data = loadtxt(data_file, delimiter=',')
        self.x = data[:, [0]]
        self.y = data[:, [1]]

        # Defining the Learning Rate and Total Iterations
        self.alpha = 0.01
        self.iterations = 1500

        # Defining the length of Training Set
        # Length of Input features should be equal to the Prediction/Output Data, else Raise Error
        if len(self.x) != len(self.y):
            raise TypeError("x and y should have same number of rows.")
        else:
            self.m = len(self.x)

        # Defining Theta
        self.theta = numpy.zeros([2, 1])
        # Keeping original data intact for Plotting purpose
        self.xxx = self.x
        self.yyy = self.y

        # Creating a vector array of ones and adding this array as first column
        # of Input Features. This is done because in equation Theta(0)X(0), X(0) = 1
        ones = numpy.ones(self.m)[:, numpy.newaxis]
        self.x = numpy.hstack((ones, self.x))

    def reinitialize(self):
        self.theta = numpy.zeros([2, 1])

    # This Cost Function is calculated using Batch Gradient Algorithm
    def cost_function(self):
        self.reinitialize()
        # Now, calculating the hypothesis as Least Squared Method
        hypothesis = numpy.dot(self.x, self.theta)
        squared_error = numpy.square(hypothesis - self.y)

        # Calculating the Cost Function
        cost = (1/(2 * self.m)) * numpy.sum(squared_error)
        # print(cost)
        print(self.theta)
        plt.figure()
        plt.scatter(self.xxx, self.yyy)
        plt.plot(self.xxx, numpy.dot(self.x, self.theta), 'b', label='Batch')

    # This Cost Function is calculated using Batch Gradient Algorithm and Learning rate
    def cost_function_with_learning_rate(self):
        self.reinitialize()
        for i in range(0, self.iterations):
            hypothesis = numpy.dot(self.x, self.theta)
            error = hypothesis - self.y
            squared_error = numpy.square(hypothesis - self.y)

            # Calculating Theta on every iteration based on Learning Rate
            self.theta = self.theta - (self.alpha/self.m) * numpy.dot(numpy.transpose(self.x), error)
            cost = (1/(2 * self.m)) * numpy.sum(squared_error)

        # print(cost)
        # print(self.theta)
        plt.plot(self.xxx, numpy.dot(self.x, self.theta), 'r', label='Batch - Learning')

    # This Cost Function is calculated using Stochastic Gradient Descent Algorithm and Learning rate
    def stochastic_cost_function(self):
        self.reinitialize()

        for xx, yy in zip(self.x, self.y):
            hypothesis = numpy.dot(xx, self.theta)
            error = hypothesis - yy
            squared_error = numpy.square(hypothesis - yy)

            # Calculating Theta on for Single Data Set
            self.theta = self.theta - ((self.alpha * error * xx).reshape(-1, 1))
            cost = (1 / (2 * self.m)) * numpy.sum(squared_error)

        plt.plot(self.xxx, numpy.dot(self.x, self.theta), 'g', label='Stochastic')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    linearRegression = LinearRegression()
    linearRegression.cost_function()
    linearRegression.cost_function_with_learning_rate()
    linearRegression.stochastic_cost_function()
