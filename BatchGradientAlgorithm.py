# Importing Libraries to carry out the Operations
import numpy
# Importing features from the Libraries to carry out the Operations
from numpy import loadtxt


class LinearRegression:
    def __init__(self):
        # Training Examples File To be Loaded from a CSV/TXT file
        data_path = r"/Users/navneet/Archive/Documents/Study Material/Self Study/Data " \
                    r"Sets/Housing_Problem_Basic_data_Set.txt"
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

    # This Cost Function is calculated using Batch Gradient Algorithm
    def cost_function(self):
        # Creating a vector array of ones and adding this array as first column
        # of Input Features. This is done because in equation Theta(0)X(0), X(0) = 1
        ones = numpy.ones(self.m)[:, numpy.newaxis]
        self.x = numpy.hstack((ones, self.x))

        # Now, calculating the hypothesis as Least Squared Method
        hypothesis = numpy.dot(self.x, self.theta)
        squared_error = numpy.square(hypothesis - self.y)

        # Calculating the Cost Function
        cost = (1/(2 * self.m)) * numpy.sum(squared_error)
        print(cost)

    # This Cost Function is calculated using Batch Gradient Algorithm
    def cost_function_with_learning_rate(self):
        # Creating a vector array of ones and adding this array as first column
        # of Input Features. This is done because in equation Theta(0)X(0), X(0) = 1
        ones = numpy.ones(self.m)[:, numpy.newaxis]
        self.x = numpy.hstack((ones, self.x))

        for i in range(0, self.iterations):
            hypothesis = numpy.dot(self.x, self.theta)
            error = hypothesis - self.y
            squared_error = numpy.square(hypothesis - self.y)

            # Calculating Theta on every iteration based on Learning Rate
            self.theta = self.theta - (self.alpha/self.m) * numpy.dot(numpy.transpose(self.x), error)
            cost = (1/(2 * self.m)) * numpy.sum(squared_error)
            print(cost)


if __name__ == '__main__':
    linearRegression = LinearRegression()
    linearRegression.cost_function_with_learning_rate()
