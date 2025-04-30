"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]
This script implements helper functions relevant to logical association rule quality metrics
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from joblib import Parallel, delayed


# Some well-known rule quality functions

def calculate_interestingness(confidence, support, rhs_support, input_length):
    """
    calculate interestingness rule quality criterion for a single rule
    :param confidence:
    :param support:
    :param rhs_support: consequent support
    :param input_length: number of transactions
    :return:
    """
    # formula taken from NiaPy 'rule.py'
    return confidence * (support / rhs_support) * (1 - (support / input_length))


def calculate_yulesq(full_count, not_ant_not_con, con_not_ant, ant_not_con):
    """
    calculate yules'q rule quality criterion for a single rule
    :param full_count: number of transactions that contain both antecedent and consequent side of a rule
    :param not_ant_not_con: number of transactions that does not contain neither antecedent nor consequent
    :param con_not_ant: number of transactions that contain consequent side but not antecedent
    :param ant_not_con: number of transactions that contain antecedent side but not consequent
    :return:
    """
    # formula taken from NiaPy 'rule.py'
    ad = full_count * not_ant_not_con
    bc = con_not_ant * ant_not_con
    yulesq = (ad - bc) / (ad + bc + 2.220446049250313e-16)
    return yulesq


def calculate_lift(support, confidence):
    return confidence / support


def calculate_conviction(support, confidence):
    return (1 - support) / (1 - confidence + 2.220446049250313e-16)


def calculate_zhangs_metric(support, support_ant, support_cons):
    """
    Taken from NiaARM's rule.py
    :param support_cons:
    :param support_ant:
    :param support:
    :return:
    """
    numerator = support - support_ant * support_cons
    denominator = (
            max(support * (1 - support_ant), support_ant * (support_cons - support))
            + 2.220446049250313e-16
    )
    return numerator / denominator


# Following are cumulative rule quality calculation functions for a given rule or itemset

def calculate_average_rule_quality(rules):
    stats = []
    for rule in rules:
        stats.append([
            rule["support"], rule["confidence"], rule["zhangs_metric"]
        ])

    stats = pd.DataFrame(stats).mean()
    stats = {
        "support": stats[0],
        "confidence": stats[1],
        "zhangs_metric": stats[2],
    }
    return stats


def calculate_basic_rule_stats(rules, transactions):
    """
    Calculate support and confidence in parallel.
    :param rules: List of rules to process.
    :param transactions: List of transactions.
    :return: Updated rules with support and confidence.
    """
    num_transactions = transactions.shape[0]
    columns = transactions.columns.tolist()

    def process_rule(rule):
        ant_count = 0
        cons_count = 0
        co_occurrence_count = 0

        for index in range(num_transactions):
            encoded_transaction = transactions.iloc[index]
            antecedent_match = all(
                encoded_transaction.iloc[columns.index(antecedent)] == 1
                for antecedent in rule['antecedents']
            )
            if antecedent_match:
                ant_count += 1
            if encoded_transaction.iloc[columns.index(rule['consequent'])] == 1:
                cons_count += 1
                if antecedent_match:
                    co_occurrence_count += 1

        support_body = ant_count / num_transactions
        rule['support'] = co_occurrence_count / num_transactions
        rule['confidence'] = rule['support'] / support_body if support_body != 0 else 0

        return rule

    # Use ThreadPoolExecutor to process rules in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        rules = list(executor.map(process_rule, rules))

    return rules if rules else None


def calculate_freq_item_support(freq_items, transactions):
    num_rows = len(transactions)
    support_values = {}

    for item in freq_items:
        conditions = {pair.split("__")[0]: pair.split("__")[1] for pair in item}
        print("conditions:", conditions)
        mask = pd.Series(True, index=transactions.index)
        for feature, value in conditions.items():
            print("transactions:", transactions[feature])
            mask &= (transactions[feature] == value)

        support_values[tuple(item)] = mask.sum() / num_rows

    average_support = sum(support_values.values()) / len(support_values) if support_values else 0
    return support_values, average_support


def calculate_rule_stats(rules, transactions, max_workers=1):
    """
    Calculate rule quality stats for the given set of rules based on the input transactions.
    """
    num_transactions = len(transactions)
    vector_tracker_list = transactions.columns.tolist()

    dataset_coverage = np.zeros(num_transactions, dtype=bool)

    def process_rule(rule):
        antecedents_indices = [vector_tracker_list.index(ant) for ant in rule['antecedents']]
        consequent_index = vector_tracker_list.index(rule['consequent'])

        # Find transactions where all antecedents are present
        antecedent_matches = np.all(transactions.iloc[:, antecedents_indices] == 1, axis=1)
        consequent_matches = transactions.iloc[:, consequent_index] == 1
        co_occurrence_matches = antecedent_matches & (transactions.iloc[:, consequent_index] == 1)

        antecedents_occurrence_count = np.sum(antecedent_matches)
        consequent_occurrence_count = np.sum(consequent_matches)
        co_occurrence_count = np.sum(co_occurrence_matches)

        support_body = antecedents_occurrence_count / num_transactions if num_transactions else 0
        support_head = consequent_occurrence_count / num_transactions if num_transactions else 0
        rule_support = co_occurrence_count / num_transactions if num_transactions else 0
        rule_confidence = rule_support / support_body if support_body != 0 else 0

        rule['support'] = rule_support
        rule['confidence'] = rule_confidence
        rule['zhangs_metric'] = calculate_zhangs_metric(rule_support, support_body, support_head)

        return antecedent_matches, rule

    # Parallel processing of rules
    results = Parallel(n_jobs=max_workers)(delayed(process_rule)(rule) for rule in rules)

    # Aggregate dataset coverage and collect updated rules
    updated_rules = []
    for antecedent_matches, rule in results:
        dataset_coverage |= antecedent_matches
        updated_rules.append(rule)

    if not updated_rules:
        return None, None

    stats = calculate_average_rule_quality(updated_rules)
    stats["coverage"] = np.sum(dataset_coverage) / num_transactions

    return {"rule_count": len(updated_rules), "average_support": stats['support'],
            "average_confidence": stats["confidence"], "average_coverage": stats["coverage"],
            "average_zhangs_metric": stats["zhangs_metric"]}, updated_rules
