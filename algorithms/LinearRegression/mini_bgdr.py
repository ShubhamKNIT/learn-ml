import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MiniBGDRegressor:
    """
    For each epoch(iteration), 
    random instances of size equal to batch_size
    called mini_batch is created and 
    gradient loss is computed over this mini_batch
    and parameters are updated
    """

    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32):
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        m, n = X_train.shape
        self.intercept_ = np.random.random()
        self.coef_ = np.random.random(n)
        
        for _ in range(self.epochs):
            # Shuffle the indices
            idx = np.random.permutation(m)
            X_train = X_train[idx]
            y_train = y_train[idx]

            for j in range(0, m, self.batch_size):
                # Create mini-batches
                X_i = X_train[j:j + self.batch_size]
                y_i = y_train[j:j + self.batch_size]

                # Ensure we don't have an empty batch
                if len(X_i) == 0:
                    continue

                # Calculate predictions
                y_pred = np.dot(X_i, self.coef_) + self.intercept_
                
                # Calculate gradients
                dcoef = -2 * np.dot((y_i - y_pred), X_i) / len(X_i)
                dintercept = -2 * np.mean(y_i - y_pred)

                # Update parameters
                self.coef_ -= self.learning_rate * dcoef
                self.intercept_ -= self.learning_rate * dintercept

                # loss_history
                errors = (y_i - y_pred)
                self.loss_history.append(np.mean(errors ** 2))

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
    
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss Convergence")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.yscale('log')  # Log scale for better visibility
        plt.show()

# Load the dataset
data = pd.read_csv('data.csv')
X = data.studytime.values.reshape(-1, 1)  # Ensure X is 2D
y = data.score.values

# Initialize and fit the model
mini_bgdr = MiniBGDRegressor(batch_size=int(X.shape[0] / 8))
mini_bgdr.fit(X, y)

# Make a prediction
print(mini_bgdr.predict([2.04]))
print(mini_bgdr.coef_, mini_bgdr.intercept_)
print(mini_bgdr.loss_history)
mini_bgdr.plot_loss()