"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

This script implements data preparation functions for tabular for association rule mining with Aerial
"""

import concurrent
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from aerial.table import get_unique_values_per_column


def _one_hot_encoding_with_feature_tracking(transactions: pd.DataFrame, parallel_workers=1):
    """
    Create input vectors for training the Autoencoder in a one-hot encoded form. And returns indices of each feature
    values in a structured way for future tracking when extracting rules from a trained Autoencoder
    :param transactions: pandas DataFrame of transactions
    :return: a python dictionary with 3 objects
        vector_list: transactions as a list of one-hot encoded vectors,
        vector_tracker_list: a list ,
        "feature_value_indices": feature_value_indices,
    """
    # Aerial uses "__" to separate column names and their values when one-hot encoding, {COL_NAME}__{value}
    # therefore, replace all "__" in column names with "--" to avoid later confusion in naming
    transactions.columns = [col.replace('__', '--') for col in transactions.columns]
    columns = transactions.columns.tolist()

    # Get input vectors in the form of one-hot encoded vectors
    unique_values, value_count = get_unique_values_per_column(transactions)
    feature_value_indices = []
    vector_tracker = []
    start = 0

    # Track what each value in the input vector corresponds to
    # Track where do values for each feature start and end in the input feature
    for feature, values in unique_values.items():
        end = start + len(values)
        feature_value_indices.append({'feature': feature, 'start': start, 'end': end})
        vector_tracker.extend([f"{feature}__{value}" for value in values])
        start = end

    # Map tracker entries to indices for fast lookup
    tracker_index_map = {key: idx for idx, key in enumerate(vector_tracker)}

    # Preallocate vector list
    vector_list = np.zeros((len(transactions), value_count), dtype=int)

    # Function to process each transaction
    def process_transaction(transaction_idx, transaction):
        transaction_vector = np.zeros(value_count, dtype=int)
        for col_idx, value in enumerate(transaction):
            if not pd.isna(value):
                key = f"{columns[col_idx]}__{value}"
                transaction_vector[tracker_index_map[key]] = 1
        return transaction_idx, transaction_vector

    # Parallelize transaction processing
    # NOTE: Preparing the input data for each of the algorithms is not included in the execution time calculation
    # Therefore, we preprocess data in parallel where possible for each of the algorithm
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [
            executor.submit(process_transaction, transaction_idx, transaction)
            for transaction_idx, transaction in enumerate(transactions.itertuples(index=False))
        ]

        for future in concurrent.futures.as_completed(futures):
            transaction_idx, transaction_vector = future.result()
            vector_list[transaction_idx] = transaction_vector

    vector_list = pd.DataFrame(vector_list, columns=vector_tracker)
    return vector_list, feature_value_indices
