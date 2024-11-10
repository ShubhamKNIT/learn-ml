import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SGDRegressor:
    """
    For each epoch(iteration) random instance
    gradient loss is calculated and parameters 
    are updated
    """

    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        m, n = X_train.shape
        self.coef_ = np.random.random()
        self.intercept_ = np.random.random()

        for _ in range(self.epochs):
            for _ in range(m):

                # random index
                idx = np.random.randint(0, m)
                X_i, y_i = (X_train[idx], y_train[idx])
                y_pred = np.dot(self.coef_, X_i) + self.intercept_

                # calculate derivates
                dcoef = - 2 * np.dot((y_i - y_pred), X_i)
                dintercept = - 2 * (y_i - y_pred)

                # adjust the parameters
                self.coef_ = self.coef_ - self.learning_rate * dcoef
                self.intercept_ = self.intercept_ - self.learning_rate * dintercept

                # loss_history
                errors = y_i - y_pred
                self.loss_history.append(errors ** 2)

        return self
    
    def predict(self, X_test):
        return np.dot(self.coef_, X_test) + self.intercept_
    
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss Convergence")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.yscale('log')
        plt.show()
    
data = pd.read_csv('data.csv')
X = data.studytime
y = data.score

X = X.values.reshape(-1, 1)
y = y.values

sgdr = SGDRegressor()
sgdr.fit(X, y)

print(sgdr.predict([2.04]))
print(sgdr.coef_, sgdr.intercept_)
print(sgdr.loss_history)
sgdr.plot_loss()