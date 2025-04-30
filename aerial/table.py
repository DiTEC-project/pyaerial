"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]
This script includes table operations relevant to association rule mining
"""

import pandas as pd


def get_unique_values_per_column(transactions):
    columns = list(transactions.columns)
    value_count = 0

    unique_values = {}
    for column in columns:
        unique_values[column] = []

    for transaction_index, transaction in transactions.iterrows():
        for index in range(len(list(transaction))):
            if not pd.isna(transaction.iloc[index]) and transaction.iloc[index] not in unique_values[columns[index]]:
                unique_values[columns[index]].append(transaction.iloc[index])
                value_count += 1

    return unique_values, value_count
