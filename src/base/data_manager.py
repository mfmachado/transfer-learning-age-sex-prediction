import logging

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class TrainingDataManager:

    def binarization(self, df, column_to_bins, new_col, n_bins):
        df[new_col] = pd.cut(df[column_to_bins], bins=n_bins, labels=list(range(n_bins)))
        return df

    def stratified_spliter(self, df: pd.DataFrame, columns_paths: str, column_label: str,
                           number_training_samples: int, number_validation_samples: int,
                           number_test_samples: int):

        ss_test = StratifiedShuffleSplit(1, train_size=number_test_samples)
        ss_val = StratifiedShuffleSplit(1, train_size=number_validation_samples)
        ss_train = StratifiedShuffleSplit(1, train_size=number_training_samples)

        test_index, index_wout_test = ss_test.split(df[columns_paths], df[column_label]).__next__()
        df_test = df.iloc[test_index].copy()
        df_wout_test = df.iloc[index_wout_test].copy()

        val_index, index_wout_val = ss_val.split(df_wout_test[columns_paths], df_wout_test[column_label]).__next__()
        df_val = df_wout_test.iloc[val_index].copy()
        df_wout_val = df_wout_test.iloc[index_wout_val].copy()

        if df_wout_val.shape[0] == number_training_samples:
            df_train = df_wout_val
        elif df_wout_val.shape[0] > number_training_samples:
            train_index, _ = ss_train.split(df_wout_val[columns_paths], df_wout_val[column_label]).__next__()
            df_train = df_wout_val.iloc[train_index].copy()
        else:
            raise ValueError

        logging.info(f"Using {df_train.shape[0]} instances to train the model.")
        logging.info(f"Using {df_val.shape[0]} instances to validate the model.")
        logging.info(f"Using {df_test.shape[0]} instances to test the model.")

        return df_train, df_val, df_test
