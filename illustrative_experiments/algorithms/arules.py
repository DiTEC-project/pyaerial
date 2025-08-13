# Runs ECLAT and Apriori algorithms from arulespy library: https://github.com/mhahsler/arulespy

import time
import ast

import numpy as np
import pandas as pd
from arulespy.arules import Transactions, eclat, parameters, apriori
from mlxtend.frequent_patterns import association_rules


def run_arules_eclat(dataset, antecedents, min_support, min_confidence):
    dataset = pd.get_dummies(dataset, prefix_sep='__', dtype=bool)
    transactions = Transactions.from_df(dataset)

    start = time.time()
    freq_items = eclat(transactions, control=parameters({"verbose": False}), parameter=parameters({
        "minlen": 1, "maxlen": antecedents + 1, "supp": float(min_support)})).as_df()
    exec_time = time.time() - start

    formatted_freq_items = pd.DataFrame({
        'itemsets': freq_items['items'].apply(lambda x: frozenset(x.strip('{}').split(','))),
        'support': freq_items['support']
    })

    start = time.time()
    rules = association_rules(formatted_freq_items, metric='confidence', min_threshold=min_confidence)
    exec_time += time.time() - start

    # dataset COVERAGE for exhaustive methods become 1 at least after reducing support to 0.3.
    # So no need to check dataset coverage if the average support of Aerial+ is lower than 0.15
    return {
        "rule_count": len(rules),
        "average_support": rules["support"].mean(),
        "average_confidence": rules["confidence"].mean(),
        "data_coverage": calculate_dataset_coverage_from_names(rules, dataset),
        "exec_time": exec_time
    }


def run_arules_apriori(dataset, antecedents, min_support, min_confidence):
    dataset = pd.get_dummies(dataset, prefix_sep='__', dtype=bool)
    transactions = Transactions.from_df(dataset)

    start = time.time()
    rules = apriori(transactions, control=parameters({"verbose": False}), parameter=parameters({
        "minlen": 1, "maxlen": antecedents + 1, "supp": float(min_support), "conf": float(min_confidence)})).as_df()
    exec_time = time.time() - start

    # change the "LHS" to "antecedents", because the coverage calculation function uses the term "antecedents"
    rules.rename(columns={"LHS": "antecedents"}, inplace=True)
    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: frozenset(item for item in x.strip("{}").split(",") if item)
    )

    return {
        "rule_count": len(rules),
        "average_rule_coverage": rules["coverage"].mean(),
        "average_support": rules["support"].mean(),
        "average_confidence": rules["confidence"].mean(),
        "data_coverage": calculate_dataset_coverage_from_names(rules, dataset),
        "exec_time": exec_time
    }


def calculate_dataset_coverage_from_names(rules, transactions):
    # Map column names to integer indices
    col_index = {name: i for i, name in enumerate(transactions.columns)}
    trans_values = transactions.values
    covered = np.zeros(len(transactions), dtype=bool)

    for antecedents in rules.antecedents:
        if len(antecedents) > 0:
            idx = np.fromiter((col_index[name] for name in antecedents), dtype=int)
            covered |= trans_values[:, idx].all(axis=1)

    return covered.sum() / len(transactions)
