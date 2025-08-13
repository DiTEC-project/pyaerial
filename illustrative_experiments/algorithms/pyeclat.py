# Runs ECLAT with pyECLAT: https://github.com/jeffrichardchemistry/pyECLAT

import time

import numpy as np
import pandas as pd
from pyECLAT import ECLAT
from mlxtend.frequent_patterns import association_rules
from illustrative_experiments.util.preprocessing import *


def run_pyeclat(dataset, min_support=0.5, min_confidence=0.8, antecedents=2):
    transactions = dataset.apply(
        lambda row: [str(f"{col}__{row[col]}") for col in dataset.columns],
        axis=1
    )
    max_len = transactions.map(len).max()
    transaction_df = pd.DataFrame(transactions.tolist(), columns=list(range(max_len)))

    start = time.time()
    eclat_instance = ECLAT(data=transaction_df)
    indexes, support = eclat_instance.fit(min_support=min_support, min_combination=1, max_combination=antecedents + 1,
                                          verbose=False)
    exec_time = time.time() - start
    frequent_itemsets = pd.DataFrame({
        'itemsets': [frozenset(itemset.split(" & ")) for itemset in support.keys()],
        'support': list(support.values())
    })
    start = time.time()
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    exec_time += time.time() - start

    return {
        "rule_count": len(rules),
        "average_support": rules["support"].mean(),
        "average_confidence": rules["confidence"].mean(),
        "data_coverage": calculate_dataset_coverage_from_names(rules, one_hot_encode_dataframe(dataset)),
        "exec_time": exec_time
    }


def calculate_dataset_coverage_from_names(rules, transactions):
    # Map column names to integer indices
    col_index = {name: i for i, name in enumerate(transactions.columns)}
    trans_values = transactions.values
    covered = np.zeros(len(transactions), dtype=bool)

    for antecedents in rules.antecedents:
        idx = np.fromiter((col_index[name] for name in antecedents), dtype=int)
        covered |= trans_values[:, idx].all(axis=1)

    return covered.sum() / len(transactions)
