import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, random_state: int = 42, test_size: float = 0.2) -> tuple:
    """
    Split the dataset into train and test sets.
    Args:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The target values.
        random_state (int): The seed used by the random number generator.
        test_size (float): The proportion of the dataset to include in the test split.
    Returns:
        X_train (numpy.ndarray): The input data for training.
        y_train (numpy.ndarray): The target values for training.
        X_test (numpy.ndarray): The input data for testing.
        y_test (numpy.ndarray): The target values for testing.

    Example:
    ```
    # Sample dataset
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display the results
    print("X_train:\n", X_train)
    print("X_test:\n", X_test)
    print("y_train:\n", y_train)
    print("y_test:\n", y_test)

    Output:
    X_train:
    [[ 1  2]
    [ 7  8]
    [ 3  4]
    [ 9 10]]
    X_test:
    [0 1 1 0]
    y_train:
    [[5 6]]
    y_test:
    [0]
    ```
    """
    # sample size
    n_samples = X.shape[0]
    
    # shuffle the indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # split the indices into train and test
    split_index = int(n_samples * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # create train and test sets
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return (X_train, y_train, X_test, y_test)

# Sample dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the results
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)


# Split the dataset using Sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)