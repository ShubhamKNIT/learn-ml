import numpy as np
import pandas as pd

class MultipleLinearRegression:
    def __init__(self):
        """
            Ordinary least Square
        """
        self.intercept_ = None  #(ß0)
        self.coef_ = None       #(ß1, ß2, ... ßn)
    
    def fit(self, X_train, y_train):
        # np.insert(np.array(), row|col index, value, axis)
        X_train = np.insert(X_train, 0, 1, axis=1)
        """
        # Example usage of np.insert()
        a = np.array([[1, 2],
                    [3, 4]])

        a = np.insert(a, 0, 1, axis = 1)

        # Output
        print(a)
        [[1 1 2]
        [1 3 4]]
        """
        
        # calculate the parameters of matrix (ßx) using normal equation
        # coeffecients of normal equation are the most optimized parameters for miniimum MSE loss
        betas = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        # betas = np.pinv(X_train) @ y_train

        # betas = np.linalg.pinv(X_train) @ y_train
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
        return self

    def predict(self, X_test):
        return (self.coef_.T @ X_test) + self.intercept_


data = pd.read_csv('algorithms/LinearRegression/data.csv')

X = data.studytime
y = data.score

X = X.values.reshape(-1, 1)
y = y.values

mlr = MultipleLinearRegression()
mlr.fit(X, y)
print(mlr.predict([2.04]))
print(mlr.coef_, mlr.intercept_)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
print(lr.coef_, lr.intercept_)