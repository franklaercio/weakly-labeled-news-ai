# Remove duplicates based on the "text" column and "id"
# Source: https://github.com/eliasjacob/imd3011-datacentric_ai
# Original author: Elias Jacob
# License: MIT

import pandas as pd


def remove_duplicates_and_overlap(df_train: pd.DataFrame, df_dev: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicate rows and ensure no overlap of review IDs between training, development, and test sets.

    Args:
        df_train (pd.DataFrame): Training set DataFrame.
        df_dev (pd.DataFrame): Development set DataFrame.
        df_test (pd.DataFrame): Test set DataFrame.

    Returns:
        tuple: A tuple containing the cleaned training, development, and test DataFrames.
    """
    # Print initial shapes of the DataFrames
    print(df_train.shape, df_dev.shape, df_test.shape)

    # Remove duplicate rows based on 'id' and 'text' columns
    df_train.drop_duplicates(subset=["id", "text"], inplace=True)
    df_dev.drop_duplicates(subset=["id", "text"], inplace=True)
    df_test.drop_duplicates(subset=["id", "text"], inplace=True)

    # Print shapes of the DataFrames after removing duplicates
    print(df_train.shape, df_dev.shape, df_test.shape)

    # Remove rows from the training set where 'id' is present in either the development or test sets
    df_train = df_train.query("id not in @df_dev.id or id not in @df_test.id")

    # Remove rows from the development set where 'id' is present in either the training or test sets
    df_dev = df_dev.query("id not in @df_train.id or id not in @df_test.id")

    # Remove rows from the test set where 'id' is present in either the training or development sets
    df_test = df_test.query("id not in @df_train.id or id not in @df_dev.id")

    # Print final shapes of the DataFrames after ensuring no overlap of review IDs
    print(df_train.shape, df_dev.shape, df_test.shape)

    df_train.reset_index(drop=True, inplace=True)
    df_dev.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_dev, df_test


def print_random_sample(df, n=50, random_state=271828) -> None:
    """
    Print a random sample of rows from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to sample from. Must contain a 'text' column.
        n (int): Number of rows to print.
        random_state (int): Random seed for reproducibility.
    """
    for row in df.sample(n, random_state=random_state).itertuples():
        print(f"{row.Index} - {row.text}\n")
