# Importing Libraries to carry out the Operations
import numpy
import matplotlib.pyplot as plt
# Importing features from the Libraries to carry out the Operations
from numpy import loadtxt


class LogisticRegression:
    def __init__(self):
        # Training Examples File To be Loaded from a CSV/TXT file
        # Put Data Set Path
        data_path = r"/Users/**/College_Admission_Basic_Data_Set.txt"
        data_file = open(data_path, 'r')
        data = loadtxt(data_file, delimiter=',')
        self.x = data[:, [0, 1]]
        self.y = data[:, [2]]


        # Defining the length of Training Set
        # Length of Input features should be equal to the Prediction/Output Data, else Raise Error
        print()
        if len(self.x) != len(self.y):
            raise TypeError("x and y should have same number of rows.")
        else:
            self.m = len(self.x)

        # Defining Theta
        self.theta = numpy.zeros([len(self.x[0]) + 1, 1])

        # Keeping original data intact for Plotting purpose
        self.xxx = self.x
        self.yyy = self.y

        # Creating a vector array of ones and adding this array as first column
        # of Input Features. This is done because in equation Theta(0)X(0), X(0) = 1
        ones = numpy.ones(self.m)[:, numpy.newaxis]
        self.x = numpy.hstack((ones, self.x))

    def cost_function(self):
        hypothesis = numpy.dot(self.x, self.theta)
        sigmoid = 1 / (1 + numpy.exp(-hypothesis))
        cost = (1 / self.m) * sum(numpy.dot(numpy.transpose(-self.y), numpy.log(sigmoid)) - numpy.dot(numpy.transpose(1 - self.y), numpy.log(1 - sigmoid)))
        grad = (1 / self.m) * numpy.dot(numpy.transpose(self.x), (sigmoid - self.y))

        print(cost)
        print(grad)

    def plot_data(self):
        plt.figure()
        positive = [idx for idx, a in enumerate(self.yyy) if a == 1]
        negative = [idx for idx, a in enumerate(self.yyy) if a == 0]
        plt.plot(self.xxx[positive, 0], self.xxx[positive, 1], 'k+', label='Admitted')
        plt.plot(self.xxx[negative, 0], self.xxx[negative, 1], 'ko', label='Not Admitted')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    logisticRegression = LogisticRegression()
    logisticRegression.cost_function()
