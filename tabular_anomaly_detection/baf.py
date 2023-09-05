import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class BAFDataset:
    def load_data(self, file_path):
        """
        Load the specified subset of the BAF dataset as a pandas DataFrame.

        Args:
        - subset (str): The subset to load. Choices are 'Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', and 'Variant V'.

        Returns:
        - df (pd.DataFrame): The BAF dataset as a pandas DataFrame.
        """
        ds_name = file_path.split("/")[-1]  # Split the path by "/" and get the last element
        subset = ds_name.split(".")[0]  # Split the filename by "." and get the first element
        
        if subset not in ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']:
            raise ValueError("Invalid subset type. Choices are 'Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', and 'Variant V'.")

        df = pd.read_csv(file_path).drop('device_fraud_count', axis=1)
        return df

    def train_test_split(self, df, month):
        """
        Split the BAF dataset into train and test sets based on the specified month.

        Args:
        - df (pd.DataFrame): The BAF dataset as a pandas DataFrame.
        - month (int): The month to use as the splitting point. Months 0-5 are in the train set, and 6-7 are in the test set, as proposed in the paper.

        Returns:
        - (X_train, y_train), (X_test, y_test) (tuple of pd.DataFrame): A tuple containing the train and test sets.
        """
        train_mask = df['month'] < month
        X_train, y_train = df[train_mask].drop(['month', 'fraud_bool'], axis=1), df[train_mask]['fraud_bool']
        X_test, y_test = df[~train_mask].drop(['month', 'fraud_bool'], axis=1), df[~train_mask]['fraud_bool']
        return (X_train, y_train), (X_test, y_test)
    
    def one_hot_encode_categorical(self, X_train, X_test):
        """
        One-hot encode the categorical features in the BAF dataset.

        This function takes the training and test sets as input, identifies the categorical features
        in the dataset, and one-hot encodes them. The one-hot encoding process converts categorical
        features into binary columns for each category/label. This is done separately for the training
        and test sets to avoid data leakage. The transformed data is then combined with the numerical
        features to form the final one-hot encoded train and test sets.

        Args:
        - X_train (pd.DataFrame): The training set features.
        - X_test (pd.DataFrame): The test set features.

        Returns:
        - X_train, X_test (tuple of pd.DataFrame): The one-hot encoded train and test sets.
        """
        # Identify the categorical features
        s = (X_train.dtypes == 'object')
        object_cols = list(s[s].index)

        # Initialize the one-hot encoder with 'ignore' for handling unknown categories in the test set
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        # Apply one-hot encoding to the categorical features in the train and test sets
        ohe_cols_train = pd.DataFrame(ohe.fit_transform(X_train[object_cols]))
        ohe_cols_test = pd.DataFrame(ohe.transform(X_test[object_cols]))

        # Set the index of the transformed data to match the original data
        ohe_cols_train.index = X_train.index
        ohe_cols_test.index = X_test.index

        # Remove the original categorical features from the train and test sets
        num_X_train = X_train.drop(object_cols, axis=1)
        num_X_test = X_test.drop(object_cols, axis=1)

        # Concatenate the numerical features with the one-hot encoded categorical features
        X_train = pd.concat([num_X_train, ohe_cols_train], axis=1)
        X_test = pd.concat([num_X_test, ohe_cols_test], axis=1)

        # Ensure the column names are strings to meet newer sklearn requirements
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        return X_train, X_test