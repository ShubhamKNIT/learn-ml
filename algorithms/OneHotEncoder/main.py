import pandas as pd

class OneHotEncoder:
    def __init__(self):
        """
        OneHotEncoder for encoding categorical variables in a dataset.

        Attributes:
        categories_ (list): The unique categories in the dataset.

        Methods:
        fit: Compute the unique categories in the dataset.
        transform: One-hot encode the categorical variable.
        fit_transform: Fit the encoder to the dataset and transform it.

        Example:
        ```

        data = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'blue', 'red']
        })

        encoder = OneHotEncoder()
        one_hot_encoded = encoder.fit_transform(data, 'color')
        print(one_hot_encoded)

        Output:
            color  blue  green  red
        0    red     0      0    1
        1   blue     1      0    0
        2  green     0      1    0
        3   blue     1      0    0
        4    red     0      0    1
        """
        self.categories_ = None
    
    def fit(self, data, column):
        self.categories_ = data[column].unique()
        return self

    def transform(self, data, column, drop=False):
        if self.categories_ is None:
            raise ValueError("The encoder is yet to be fitted. Call `fit()` first.")
        
        # Initialize the DataFrame for OneHotEncoding of categories_
        one_hot_df = pd.DataFrame(0, index=data.index, columns=self.categories_)

        # Populate the one_hot_df
        for idx, category in enumerate(data[column]):
            if category in self.categories_:
                one_hot_df.loc[idx, category] = 1
        
        if drop:
            return pd.concat([data.drop(columns=column), one_hot_df], axis = 1)
        else:
            return pd.concat([data, one_hot_df], axis = 1)

    def fit_transform(self, data, column, drop=False):
        self.categories_ = data[column].unique()
        return self.transform(data, column, drop)
    

data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red']
})

encoder = OneHotEncoder()

one_hot_encoded = encoder.fit_transform(data, 'color')
print(one_hot_encoded)