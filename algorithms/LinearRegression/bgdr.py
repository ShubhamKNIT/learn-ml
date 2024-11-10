import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BGDRegressor:
    """
        Batch Gradient Descent Regressor
        (Vanilla Gradient Descent)

        In one iteration the parameters for model is 
        adjusted for given input training data set
    """
    def __init__(self, learning_rate=0.001, epochs=200):
        self.intercept_ = None
        self.coef_ = None
        self.loss_history = []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.m, self.n = X_train.shape

        self.coef_ = np.random.random(self.n)
        self.intercept_ = np.random.random()

        for _ in range(self.epochs):
            # calculate prediction
            y_pred = self.predict(X_train)

            # calculate gradient of loss
            dintercept = - (2 * (X_train.T).dot(y_train - y_pred))/self.m
            dcoef = - (2 * np.sum(y_train - y_pred))/self.m

            # update parameters
            self.coef_ = self.coef_ - self.learning_rate * dcoef
            self.intercept_ = self.intercept_ - self.learning_rate * dintercept
            
            # loss_history
            errors = (y_train - y_pred)
            self.loss_history.append(np.mean(errors ** 2))

        return self

    
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
    
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss Convergence")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.yscale('log')  # Log scale for better visibility
        plt.show()

data = pd.read_csv('data.csv')
X = data.studytime
y = data.score

X = X.values.reshape(-1, 1)
y = y.values

bgdr = BGDRegressor()
bgdr.fit(X, y)

print(bgdr.predict([2.04]))
print(bgdr.coef_, bgdr.intercept_)
bgdr.plot_loss()
print(bgdr.loss_history)