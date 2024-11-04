import numpy as np
from sklearn.base import clone

def cv_score(model, X, y, cv=5):
    scores = []
    n_samples = len(y)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_size = n_samples // cv

    for i in range(cv):
        # Clone the model to ensure a fresh model for each iteration
        model = clone(model)

        # Split the dataset into train and test sets
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        # Create the train and test sets
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Fit the model and make predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate the accuracy of the model
        score = np.mean(y_pred == y_test)
        scores.append(score)
    
    return np.array(scores)