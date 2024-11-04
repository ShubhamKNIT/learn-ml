import numpy as np
import pandas as pd

class CustomMeanImputer:
    """
    Custom mean imputer for filling missing values in a dataset.

    Attributes:
    means (dict): The means of each column in the dataset.

    Methods:
    fit: Compute the means of each column in the dataset.
    transform: Fill missing values in the dataset with the means.
    fit_transform: Fit the imputer to the dataset and transform it.

    Example:
    ```
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [6, 7, 8, np.nan, 10],
        'C': [11, 12, 13, 14, 15]
    })

    imputer = CustomMeanImputer()
    data_imputed = imputer.fit_transform(data)

    print(data_imputed)

    Output:
            A     B   C
        0  1.00   6.0  11
        1  2.00   7.0  12
        2  3.00   8.0  13
        3  4.00   7.75 14
        4  5.00  10.0  15
    """
    def __init__(self):
        self.means = {}
    
    def fit(self, X):
        self.means = X.mean()
        return self
    
    def transform(self, X):
        return X.fillna(self.means)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, 7, 8, np.nan, 10],
    'C': [11, 12, 13, 14, 15]
})

imputer = CustomMeanImputer()
data_imputed = imputer.fit_transform(data)

print(data_imputed)

# Imputation using Sklearn
# from sklearn.impute import SimpleImputer

# data = pd.DataFrame({
#     'A': [1, 2, np.nan, 4, 5],
#     'B': [6, 7, 8, np.nan, 10],
#     'C': [11, 12, 13, 14, 15]
# })

# imputer = SimpleImputer(strategy='mean')
# data_imputed = imputer.fit_transform(data)

# print(data_imputed)

# Other imputers of Sklearn
# - SimpleImputer(strategy='mean')
# - SimpleImputer(strategy='median')
# - SimpleImputer(strategy='most_frequent')
# - SimpleImputer(strategy='constant', fill_value=0)
# - KNNImputer(n_neighbors=2)
# - IterativeImputer()
# - ColumnTransformer()
