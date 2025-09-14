"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]
This script implements helper functions relevant to logical association rule quality metrics
"""
from concurrent.futures import ThreadPoolExecutor

import logging
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

logger = logging.getLogger("aerial")


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
            rule["rule_coverage"], rule["support"], rule["confidence"], rule["zhangs_metric"]
        ])

    stats = pd.DataFrame(stats).mean()
    stats = {
        "rule_coverage": stats[0],
        "support": stats[1],
        "confidence": stats[2],
        "zhangs_metric": stats[3],
    }
    return stats


def calculate_basic_rule_stats(rules, transactions, num_workers=1):
    """
    Calculate support and confidence for rules in parallel using vectorized operations.
    :param rules: List of rules, each a dict with 'antecedents' and 'consequent'.
    :param transactions: DataFrame with binary transaction data.
    :param num_workers: Number of parallel threads to use.
    :return: Updated list of rules with support and confidence.
    """

    logger.debug(
        f"Calculating support and confidence metrics for {len(rules)} rules over {len(transactions)} transactions...")

    num_transactions = len(transactions)
    transaction_array = transactions.to_numpy()
    columns = transactions.columns.tolist()
    column_indices = {col: i for i, col in enumerate(columns)}

    def process_rule(rule):
        antecedent_indices = [column_indices[a] for a in rule['antecedents']]
        consequent_index = column_indices[rule['consequent']]

        # Vectorized masks
        antecedent_mask = np.all(transaction_array[:, antecedent_indices] == 1, axis=1)
        consequent_mask = transaction_array[:, consequent_index] == 1
        co_occurrence_mask = antecedent_mask & consequent_mask

        ant_count = np.sum(antecedent_mask)
        co_occurrence_count = np.sum(co_occurrence_mask)

        rule['support'] = co_occurrence_count / num_transactions
        rule['confidence'] = rule['support'] / (ant_count / num_transactions) if ant_count != 0 else 0

        return rule

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        rules = list(executor.map(process_rule, rules))

    return rules if rules else None


def calculate_freq_item_support(freq_items, transactions):
    num_rows = len(transactions)
    support_values = {}

    for item in freq_items:
        conditions = {pair.split("__")[0]: pair.split("__")[1] for pair in item}
        mask = pd.Series(True, index=transactions.index)
        for feature, value in conditions.items():
            mask &= (transactions[feature] == value)

        support_values[tuple(item)] = mask.sum() / num_rows

    average_support = sum(support_values.values()) / len(support_values) if support_values else 0
    return support_values, average_support


def calculate_rule_stats(rules, transactions, max_workers=1):
    """
    Calculate rule quality stats for the given set of rules based on the input transactions.
    """
    logger.debug(f"Calculating rule quality metrics for {len(rules)} rules over {len(transactions)} transactions ...")

    if max_workers == 1:
        logger.info("To speed up rule quality calculations, set max_workers > 1 in calculate_rule_stats() "
                    "to process rules in parallel.")
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

        rule['support'] = float(round(rule_support, 3))
        rule['confidence'] = float(round(rule_confidence, 3))
        rule['zhangs_metric'] = float(round(calculate_zhangs_metric(rule_support, support_body, support_head), 3))
        rule['rule_coverage'] = float(
            round(antecedents_occurrence_count / num_transactions if num_transactions else 0, 3))

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
    stats["data_coverage"] = np.sum(dataset_coverage) / num_transactions

    return {"rule_count": len(updated_rules), "average_rule_coverage": float(round(stats["rule_coverage"], 3)),
            "average_support": float(round(stats['support'], 3)),
            "average_confidence": float(round(stats["confidence"], 3)),
            "data_coverage": float(round(stats["data_coverage"], 3)),
            "average_zhangs_metric": float(round(stats["zhangs_metric"], 3))}, updated_rules
