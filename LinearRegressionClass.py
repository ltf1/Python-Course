import numpy as np
import pandas as pd

class LinearRegressionClass(object):

    def __init__(self):
        self.coefficients = []
        self.r_square = []

    def fit(self, X, y):
        if len(X.shape) == 1: X = self._reshape_x(X)

        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)


    def _reshape_x(self, X): #reshape features (X) to two dimensional in case there is only one feature
        return X.reshape(-1, 1)


    def StandardErrors(self, cross_tab):
        yT_y = cross_tab[-1:, -1:] # sum of squares total or SST
        n = cross_tab[:1, :1]
        y_bar_square = np.square(cross_tab[:1, -1:])
        SST = yT_y - (y_bar_square / n)

        XT = np.matrix.transpose(X) #Taking the transpose of X matrix
        XT_X = np.matmul(XT, X) # Multiply X transpose multipled X matrices
        XT_X_inv = np.linalg.inv(XT_X) # Find the inverse of this matrix
        XT_y = np.matmul(XT, y) # Multiply X transpose multipled y matrix
        SSR = np.sum(np.multiply(self.coefficients, XT_y)) - (y_bar_square / n)

        self.r_square = SSR / SST #r_square
