"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]
This script implements helper functions relevant to logical association rule quality metrics
"""
import logging
import numpy as np

from joblib import Parallel, delayed

logger = logging.getLogger("aerial")

# Available quality metrics
AVAILABLE_METRICS = ['support', 'confidence', 'zhangs_metric', 'lift', 'conviction', 'yulesq', 'interestingness']
DEFAULT_RULE_METRICS = ['support', 'confidence', 'zhangs_metric']


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


def _calculate_rule_quality_from_indices(antecedent_indices, consequent_index, transaction_array,
                                         num_transactions, quality_metrics):
    """
    Fast rule quality calculation using integer indices directly (internal function).

    :param antecedent_indices: List of column indices for antecedents
    :param consequent_index: Column index for consequent
    :param transaction_array: Numpy array of binary transaction data
    :param num_transactions: Total number of transactions
    :param quality_metrics: List of quality metrics to calculate
    :return: Dictionary with requested quality metrics
    """
    # Vectorized masks for fast computation
    if len(antecedent_indices) > 0:
        antecedent_mask = np.all(transaction_array[:, antecedent_indices] == 1, axis=1)
    else:
        antecedent_mask = np.ones(num_transactions, dtype=bool)

    consequent_mask = transaction_array[:, consequent_index] == 1
    co_occurrence_mask = antecedent_mask & consequent_mask

    ant_count = np.sum(antecedent_mask)
    cons_count = np.sum(consequent_mask)
    co_occurrence_count = np.sum(co_occurrence_mask)

    # Calculate basic metrics
    support_body = ant_count / num_transactions if num_transactions else 0
    support_head = cons_count / num_transactions if num_transactions else 0
    rule_support = co_occurrence_count / num_transactions if num_transactions else 0
    rule_confidence = rule_support / support_body if support_body != 0 else 0

    result = {}

    # Calculate requested metrics
    if 'support' in quality_metrics:
        result['support'] = float(round(rule_support, 3))

    if 'confidence' in quality_metrics:
        result['confidence'] = float(round(rule_confidence, 3))

    if 'zhangs_metric' in quality_metrics:
        result['zhangs_metric'] = float(round(
            calculate_zhangs_metric(rule_support, support_body, support_head), 3))

    if 'lift' in quality_metrics:
        result['lift'] = float(round(calculate_lift(rule_support, rule_confidence), 3))

    if 'conviction' in quality_metrics:
        result['conviction'] = float(round(calculate_conviction(rule_support, rule_confidence), 3))

    if 'yulesq' in quality_metrics:
        not_ant_not_con = num_transactions - ant_count - cons_count + co_occurrence_count
        con_not_ant = cons_count - co_occurrence_count
        ant_not_con = ant_count - co_occurrence_count
        result['yulesq'] = float(round(
            calculate_yulesq(co_occurrence_count, not_ant_not_con, con_not_ant, ant_not_con), 3))

    if 'interestingness' in quality_metrics:
        result['interestingness'] = float(round(
            calculate_interestingness(rule_confidence, rule_support, support_head, num_transactions), 3))

    # Always include rule_coverage for internal calculations
    result['rule_coverage'] = float(round(support_body, 3))
    result['_antecedent_mask'] = antecedent_mask  # For dataset coverage calculation

    return result


def _calculate_itemset_support_from_indices(itemset_indices, transaction_array, num_transactions):
    """
    Fast itemset support calculation using integer indices directly (internal function).

    :param itemset_indices: List of column indices for the itemset
    :param transaction_array: Numpy array of binary transaction data
    :param num_transactions: Total number of transactions
    :return: Float support value
    """
    if len(itemset_indices) > 0:
        mask = np.all(transaction_array[:, itemset_indices] == 1, axis=1)
    else:
        mask = np.ones(num_transactions, dtype=bool)

    support = np.sum(mask) / num_transactions
    return float(round(support, 3))


def calculate_rule_metrics(rules_with_indices, transaction_array, quality_metrics, num_workers=1):
    """
    Calculate quality metrics for multiple rules with optional parallelization.

    :param rules_with_indices: List of dicts with 'antecedent_indices' and 'consequent_index'
    :param transaction_array: Numpy array of binary transaction data
    :param quality_metrics: List of quality metrics to calculate
    :param num_workers: Number of parallel workers (default=1 for sequential processing).
                        Parallelization is only used when rule count >= 1000 to avoid overhead.
    :return: List of tuples (metrics_dict, antecedent_mask) for each rule
    """
    num_transactions = len(transaction_array)
    num_rules = len(rules_with_indices)

    # Automatic threshold: only use parallelization for large rule counts
    if num_workers > 1 and num_rules < 1000:
        logger.info(f"Automatically using sequential processing (num_workers=1) for {num_rules} rules. "
                   f"Parallelization is likely only beneficial for 1000+ rules due to overhead costs.")
        num_workers = 1

    def process_single_rule(rule_idx):
        ant_indices = rule_idx['antecedent_indices']
        cons_index = rule_idx['consequent_index']
        return _calculate_rule_quality_from_indices(
            ant_indices, cons_index, transaction_array, num_transactions, quality_metrics
        )

    if num_workers == 1:
        # Sequential processing
        results = [process_single_rule(rule) for rule in rules_with_indices]
    else:
        # Parallel processing
        logger.info(f"Using parallel processing with {num_workers} workers for {num_rules} rules.")
        results = Parallel(n_jobs=num_workers)(
            delayed(process_single_rule)(rule) for rule in rules_with_indices
        )

    return results


def calculate_itemset_metrics(itemsets_with_indices, transaction_array, num_workers=1):
    """
    Calculate support for multiple itemsets with optional parallelization.

    :param itemsets_with_indices: List of itemsets (each is a list of column indices)
    :param transaction_array: Numpy array of binary transaction data
    :param num_workers: Number of parallel workers (default=1 for sequential processing).
                        Parallelization is only used when itemset count >= 1000 to avoid overhead.
    :return: List of float support values for each itemset
    """
    num_transactions = len(transaction_array)
    num_itemsets = len(itemsets_with_indices)

    # Automatic threshold: only use parallelization for large itemset counts
    if num_workers > 1 and num_itemsets < 1000:
        logger.info(f"Automatically using sequential processing (num_workers=1) for {num_itemsets} itemsets. "
                   f"Parallelization is likely only beneficial for 1000+ itemsets due to overhead costs.")
        num_workers = 1

    def process_single_itemset(itemset_indices):
        return _calculate_itemset_support_from_indices(
            itemset_indices, transaction_array, num_transactions
        )

    if num_workers == 1:
        # Sequential processing
        results = [process_single_itemset(itemset) for itemset in itemsets_with_indices]
    else:
        # Parallel processing
        logger.info(f"Using parallel processing with {num_workers} workers for {num_itemsets} itemsets.")
        results = Parallel(n_jobs=num_workers)(
            delayed(process_single_itemset)(itemset) for itemset in itemsets_with_indices
        )

    return results
