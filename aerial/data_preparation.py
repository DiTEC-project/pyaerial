"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

This script implements data preparation functions for tabular for association rule mining with Aerial
"""

import concurrent
import logging
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from aerial.table import get_unique_values_per_column

logger = logging.getLogger("aerial")


def _one_hot_encoding_with_feature_tracking(transactions: pd.DataFrame, parallel_workers=1):
    """
    Create input vectors for training the Autoencoder in a one-hot encoded form. And returns indices of each feature
    values in a structured way for future tracking when extracting rules from a trained Autoencoder.

    :param transactions: pandas DataFrame of transactions
    :return: a python dictionary with 3 objects:
        - vector_list: transactions as a list of one-hot encoded vectors,
        - feature_value_indices: list of dicts indicating the position of each feature’s encoded values
    """
    # Ensure column names don't conflict with separator used in one-hot encoding
    transactions.columns = [col.replace('__', '--') for col in transactions.columns]
    columns = transactions.columns.tolist()

    # 1. Check if all columns are categorical
    non_categorical_cols = []
    for col in transactions.columns:
        if not is_effectively_categorical(transactions[col], col_name=col):
            non_categorical_cols.append(col)

    if non_categorical_cols:
        logger.error(
            f"Expected all columns to be categorical or already one-hot encoded. "
            f"The following columns do not meet that condition: {non_categorical_cols}."
            f"You can use discretization.equal_frequency_discretization(your_dataset, n_bins=5) as one way "
            f"to discretize your data"
        )
        return None, None

    # Collect unique values only for columns to encode
    unique_values = {col: sorted(transactions[col].dropna().unique()) for col in transactions.columns}

    # Build feature index mapping
    feature_value_indices = []
    vector_tracker = []
    start = 0

    # Track indices for encoded columns
    for feature, values in unique_values.items():
        end = start + len(values)
        feature_value_indices.append({'feature': feature, 'start': start, 'end': end})
        vector_tracker.extend([f"{feature}__{value}" for value in values])
        start = end

    # Total number of features
    value_count = len(vector_tracker)
    tracker_index_map = {key: idx for idx, key in enumerate(vector_tracker)}
    vector_list = np.zeros((len(transactions), value_count), dtype=int)

    # Function to process each row
    def process_transaction(transaction_idx, transaction_row):
        transaction_vector = np.zeros(value_count, dtype=int)
        for col_idx, col in enumerate(columns):
            val = transaction_row[col_idx]
            if pd.isna(val):
                continue
            key = f"{col}__{val}"
            transaction_vector[tracker_index_map[key]] = 1
        return transaction_idx, transaction_vector

    # Parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [
            executor.submit(process_transaction, idx, row)
            for idx, row in enumerate(transactions.itertuples(index=False, name=None))
        ]
        for future in concurrent.futures.as_completed(futures):
            idx, vec = future.result()
            vector_list[idx] = vec

    vector_list = pd.DataFrame(vector_list, columns=vector_tracker)
    return vector_list, feature_value_indices


def is_effectively_categorical(col: pd.Series, col_name: str, max_categories=10) -> bool:
    """
    Determine if a column should be treated as categorical.
    Logs info if a numeric column is treated as categorical due to few unique values.
    """

    if isinstance(col.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col):
        return True
    if pd.api.types.is_numeric_dtype(col):
        unique_vals = col.dropna().unique()
        # if a column has less than max_categories different numbers in it, we can treat that as categorical
        if len(unique_vals) <= max_categories:
            logger.debug(
                f"Column '{col_name}' is numeric with few unique values ({len(unique_vals)}). "
                "Treating it as categorical."
            )
            return True
    return False
