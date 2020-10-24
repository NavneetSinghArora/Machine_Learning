# Importing Libraries to carry out the Operations
import numpy
import matplotlib.pyplot as plt
# Importing features from the Libraries to carry out the Operations
from numpy import loadtxt


class LinearRegression:
    def __init__(self):
        # Training Examples File To be Loaded from a CSV/TXT file
        # Put Data Set Path
        data_path = r"/Data Sets/Housing_Problem_Basic_data_Set.txt"
        data_file = open(data_path, 'r')
        data = loadtxt(data_file, delimiter=',')
        self.x = data[:, [0]]
        self.y = data[:, [1]]

        # Defining the length of Training Set
        # Length of Input features should be equal to the Prediction/Output Data, else Raise Error
        if len(self.x) != len(self.y):
            raise TypeError("x and y should have same number of rows.")
        else:
            self.m = len(self.x)

        # Defining Theta
        self.theta = numpy.zeros([2, 1])

        # Creating a vector array of ones and adding this array as first column
        # of Input Features. This is done because in equation Theta(0)X(0), X(0) = 1
        ones = numpy.ones(self.m)[:, numpy.newaxis]
        self.x = numpy.hstack((ones, self.x))

    # This Cost Function is calculated using Batch Gradient Algorithm
    def cost_function(self):
        # Now, calculating the hypothesis as Least Squared Method
        hypothesis = numpy.dot(self.x, self.theta)
        squared_error = numpy.square(hypothesis - self.y)

        # Calculating the Cost Function
        cost = (1/(2 * self.m)) * numpy.sum(squared_error)


if __name__ == '__main__':
    linearRegression = LinearRegression()
    linearRegression.cost_function()
