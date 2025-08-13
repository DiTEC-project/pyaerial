# Runs FP-Growth and HMine algorithms from MLxtend: https://rasbt.github.io/mlxtend/

import time

import numpy as np
import pandas as pd
from illustrative_experiments.util.preprocessing import *
from mlxtend.frequent_patterns import fpgrowth, association_rules, hmine
from mlxtend.preprocessing import TransactionEncoder


def run_mlxtend_fpgrowth(dataset, antecedents, min_support, min_confidence):
    start_time = time.time()
    frequent_itemsets = fpgrowth(dataset, min_support=min_support, max_len=(antecedents + 1))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    exec_time = time.time() - start_time

    return {
        "rule_count": len(rules),
        "average_support": rules["support"].mean(),
        "average_confidence": rules["confidence"].mean(),
        "data_coverage": calculate_dataset_coverage(rules, dataset),
        "exec_time": exec_time
    }


def run_mlxtend_hmine(dataset, antecedents, min_support, min_confidence):
    start_time = time.time()
    frequent_itemsets = hmine(dataset, min_support=min_support, max_len=(antecedents + 1))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    exec_time = time.time() - start_time

    return {
        "rule_count": len(rules),
        "average_support": rules["support"].mean(),
        "average_confidence": rules["confidence"].mean(),
        "data_coverage": calculate_dataset_coverage(rules, dataset),
        "exec_time": exec_time
    }


def mlxtend_reformat_rules(rules):
    reformatted_rules = []
    for rule_index, rule in rules.iterrows():
        new_rule = {'antecedents': list(rule['antecedents']), 'consequent': list(rule['consequents']),
                    'support': rule['support'], 'confidence': rule['confidence']}
        reformatted_rules.append(new_rule)
    return reformatted_rules


def calculate_dataset_coverage(rules, transactions):
    trans_values = transactions.values  # Boolean NumPy array
    covered = np.zeros(len(transactions), dtype=bool)

    for antecedents in rules.antecedents:
        idx = np.fromiter(antecedents, dtype=int)
        covered |= trans_values[:, idx].all(axis=1)

    return covered.sum() / len(transactions)

