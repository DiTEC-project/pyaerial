"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

Includes the Aerial algorithm's source code for association rule (and frequent itemsets) extraction from a
trained Autoencoder (Neurosymbolic association rule mining from tabular data - https://proceedings.mlr.press/v284/karabulut25a.html)
"""
from collections import defaultdict

import torch

from itertools import combinations

from aerial.model import AutoEncoder
from aerial.rule_quality import (
    calculate_rule_metrics,
    calculate_itemset_metrics,
    DEFAULT_RULE_METRICS,
    AVAILABLE_METRICS
)
import numpy as np
import logging

logger = logging.getLogger("aerial")


def generate_rules(autoencoder: AutoEncoder, features_of_interest: list = None, ant_similarity=0.5, cons_similarity=0.8,
                   max_antecedents=2, target_classes=None, quality_metrics=None, num_workers=1):
    """
    Extract association rules from a trained Autoencoder using Aerial+ algorithm.
    Rule quality metrics are calculated automatically and included in the output.

    :param autoencoder: a trained Autoencoder for ARM
    :param features_of_interest: list: only look for rules that have these features of interest on the antecedent side
        accepted form ["feature1", "feature2", {"feature3": "value1}, ...], either a feature name as str, or specific value
        of a feature in object form
    :param ant_similarity: antecedent similarity threshold (default=0.5)
    :param cons_similarity: consequent similarity threshold (default=0.8)
    :param max_antecedents: max number of antecedents that the rules will contain (default=2)
    :param target_classes: list: if given a list of target classes, generate rules with the target classes on the
        right hand side only, the content of the list is as same as features_of_interest
    :param quality_metrics: list of quality metrics to calculate. Default is ['support', 'confidence', 'zhangs_metric'].
        Available metrics: 'support', 'confidence', 'zhangs_metric', 'lift', 'conviction', 'yulesq', 'interestingness'
    :param num_workers: number of parallel workers for quality metric calculation (default=1 for sequential processing)
    :return: dict with 'rules' (list of rules with quality metrics) and 'statistics' (aggregate stats)
    """
    if not autoencoder:
        logger.error("A trained Autoencoder has to be provided before generating rules.")
        return None

    # Validate quality metrics
    if quality_metrics is None:
        quality_metrics = DEFAULT_RULE_METRICS.copy()
    else:
        invalid_metrics = [m for m in quality_metrics if m not in AVAILABLE_METRICS]
        if invalid_metrics:
            logger.error(f"Invalid quality metrics: {invalid_metrics}. Available: {AVAILABLE_METRICS}")
            return None

    logger.info("Mining association rules...")
    logger.debug("Extracting association rules from the given trained Autoencoder ...")

    # Store rules with their integer indices for fast quality calculation
    rules_with_indices = []
    input_vector_size = autoencoder.encoder[0].in_features

    # process features of interest
    significant_features, insignificant_feature_values = extract_significant_features_and_ignored_indices(
        features_of_interest, autoencoder)

    feature_value_indices = autoencoder.feature_value_indices

    # Initialize input vectors with all equal probability per feature value
    unmarked_features = _initialize_input_vectors(input_vector_size, feature_value_indices)

    # Precompute target indices for softmax to speed things up
    softmax_ranges = [range(cat['start'], cat['end']) for cat in significant_features]

    # Precompute index-to-feature-range mapping for fast feature conflict detection
    # This maps each index to its feature range (start, end) for O(1) lookup
    index_to_feature_range = {}
    for cat in feature_value_indices:
        for idx in range(cat['start'], cat['end']):
            index_to_feature_range[idx] = (cat['start'], cat['end'])

    # If target_classes are specified, narrow the target range and features to constrain the consequent side of a rule
    significant_consequents, insignificant_consequent_values = extract_significant_features_and_ignored_indices(
        target_classes, autoencoder)
    significant_consequent_indices = [
        index
        for feature in significant_consequents
        for index in range(feature['start'], feature['end'])
        if index not in insignificant_consequent_values
    ]

    feature_value_indices = [range(cat['start'], cat['end']) for cat in feature_value_indices]

    for r in range(1, max_antecedents + 1):
        if r == 2:
            softmax_ranges = [
                feature_range for feature_range in softmax_ranges if
                not all(idx in insignificant_feature_values for idx in range(feature_range.start, feature_range.stop))
            ]

        feature_combinations = list(combinations(softmax_ranges, r))  # Generate combinations

        # Vectorized model evaluation batch
        batch_vectors = []
        batch_candidate_antecedent_list = []

        for category_list in feature_combinations:
            test_vectors, candidate_antecedent_list = _mark_features(unmarked_features, list(category_list),
                                                                     insignificant_feature_values)
            if len(test_vectors) > 0:
                batch_vectors.extend(test_vectors)
                batch_candidate_antecedent_list.extend(candidate_antecedent_list)

        if batch_vectors:
            batch_vectors = torch.tensor(np.array(batch_vectors), dtype=torch.float32)
            batch_vectors = batch_vectors.to(next(autoencoder.parameters()).device)
            # Perform a single model evaluation for the batch
            implications_batch = autoencoder(batch_vectors, feature_value_indices).detach().cpu().numpy()
            for test_vector, implication_probabilities, candidate_antecedents \
                    in zip(batch_vectors, implications_batch, batch_candidate_antecedent_list):
                if len(candidate_antecedents) == 0:
                    continue

                # Identify low-support antecedents
                if any(implication_probabilities[ant] <= ant_similarity for ant in candidate_antecedents):
                    if r == 1:
                        insignificant_feature_values = np.append(insignificant_feature_values, candidate_antecedents)
                    continue

                # Get the feature ranges used in antecedents to prevent same-feature rules
                antecedent_ranges = set(index_to_feature_range[ant_idx] for ant_idx in candidate_antecedents)

                # Identify high-support consequents (excluding same features as antecedents)
                consequent_list = [
                    prob_index for prob_index in significant_consequent_indices
                    if index_to_feature_range[prob_index] not in antecedent_ranges and
                       implication_probabilities[prob_index] >= cons_similarity
                ]

                if consequent_list:
                    # Store rules with integer indices for fast quality calculation
                    for consequent_idx in consequent_list:
                        rules_with_indices.append({
                            'antecedent_indices': candidate_antecedents.tolist() if isinstance(candidate_antecedents, np.ndarray) else list(candidate_antecedents),
                            'consequent_index': int(consequent_idx)
                        })

    logger.info(f"Found {len(rules_with_indices)} rules")

    if len(rules_with_indices) == 0:
        logger.info("No rules found")
        return {'rules': [], 'statistics': {}}

    # Calculate quality metrics using batch processing with optional parallelization
    logger.info(f"Calculating quality metrics: {', '.join(quality_metrics)}")
    transaction_array = autoencoder.input_vectors.to_numpy()
    num_transactions = len(transaction_array)

    # Batch calculate metrics for all rules (with optional parallelization)
    all_metrics = calculate_rule_metrics(
        rules_with_indices, transaction_array, quality_metrics, num_workers=num_workers
    )

    # Build final rules and calculate dataset coverage
    final_rules = []
    dataset_coverage = np.zeros(num_transactions, dtype=bool)

    for rule_idx, metrics in zip(rules_with_indices, all_metrics):
        ant_indices = rule_idx['antecedent_indices']
        cons_index = rule_idx['consequent_index']

        # Extract antecedent mask for dataset coverage
        antecedent_mask = metrics.pop('_antecedent_mask')
        dataset_coverage |= antecedent_mask

        # Convert indices to human-readable format
        antecedents = [
            {'feature': autoencoder.feature_values[idx].split('__', 1)[0],
             'value': autoencoder.feature_values[idx].split('__', 1)[1]}
            for idx in ant_indices
        ]
        consequent = {
            'feature': autoencoder.feature_values[cons_index].split('__', 1)[0],
            'value': autoencoder.feature_values[cons_index].split('__', 1)[1]
        }

        # Build final rule with quality metrics
        rule = {
            'antecedents': antecedents,
            'consequent': consequent
        }
        rule.update(metrics)
        final_rules.append(rule)

    # Calculate aggregate statistics
    logger.info("Calculating aggregate statistics")
    stats = _calculate_aggregate_stats(final_rules, dataset_coverage, num_transactions, quality_metrics)

    logger.info(f"Mining complete: {len(final_rules)} rules with avg support={stats.get('average_support', 0):.3f}, "
                f"avg confidence={stats.get('average_confidence', 0):.3f}")

    return {'rules': final_rules, 'statistics': stats}


def generate_frequent_itemsets(autoencoder: AutoEncoder, features_of_interest=None, similarity=0.5, max_length=2, num_workers=1):
    """
    Generate frequent itemsets using the Aerial+ algorithm.
    Support values are calculated automatically and included in the output.

    :param autoencoder: a trained Autoencoder
    :param features_of_interest: list: only look for itemsets that have these features of interest
        accepted form ["feature1", "feature2", {"feature3": "value1}, ...], either a feature name as str, or specific value
        of a feature in object form
    :param similarity: similarity threshold (default=0.5)
    :param max_length: max itemset length (default=2)
    :param num_workers: number of parallel workers for support calculation (default=1 for sequential processing)
    :return: dict with 'itemsets' (list of itemsets with support) and 'statistics' (aggregate stats)
        Example: {
            'itemsets': [
                {'itemset': [{'feature': 'age', 'value': '30-39'}], 'support': 0.524},
                {'itemset': [{'feature': 'age', 'value': '30-39'}, {'feature': 'tumor-size', 'value': '20-24'}], 'support': 0.312}
            ],
            'statistics': {'itemset_count': 2, 'average_support': 0.418}
        }
    """
    if not autoencoder:
        logger.error("A trained Autoencoder has to be provided before extracting frequent items.")
        return None

    logger.info("Mining frequent itemsets...")
    logger.debug("Extracting frequent items from the given trained Autoencoder ...")

    # Store itemsets with their integer indices for fast support calculation
    itemsets_with_indices = []
    input_vector_size = len(autoencoder.feature_values)

    # process features of interest
    significant_features, insignificant_feature_values = extract_significant_features_and_ignored_indices(
        features_of_interest, autoencoder)

    feature_value_indices = autoencoder.feature_value_indices

    # Initialize input vectors once
    unmarked_features = _initialize_input_vectors(input_vector_size, feature_value_indices)

    # Precompute target indices for softmax
    feature_value_indices = [range(cat['start'], cat['end']) for cat in feature_value_indices]
    softmax_ranges = [range(cat['start'], cat['end']) for cat in significant_features]

    # Iteratively process combinations of increasing size
    for r in range(1, max_length + 1):
        softmax_ranges = [
            feature_range for feature_range in softmax_ranges if
            not all(idx in insignificant_feature_values for idx in range(feature_range.start, feature_range.stop))
        ]

        feature_combinations = list(combinations(softmax_ranges, r))  # Generate combinations

        # Vectorized model evaluation batch
        batch_vectors = []
        batch_candidate_antecedent_list = []

        for category_list in feature_combinations:
            test_vectors, candidate_antecedent_list = _mark_features(unmarked_features, list(category_list),
                                                                     insignificant_feature_values)
            if len(test_vectors) > 0:
                batch_vectors.extend(test_vectors)
                batch_candidate_antecedent_list.extend(candidate_antecedent_list)
        if batch_vectors:
            batch_vectors = torch.tensor(np.array(batch_vectors), dtype=torch.float32)
            batch_vectors = batch_vectors.to(next(autoencoder.parameters()).device)
            # Perform a single model evaluation for the batch
            implications_batch = autoencoder(batch_vectors, feature_value_indices).detach().cpu().numpy()
            for test_vector, implication_probabilities, candidate_antecedents \
                    in zip(batch_vectors, implications_batch, batch_candidate_antecedent_list):
                if len(candidate_antecedents) == 0:
                    continue

                # Identify low-support antecedents
                if any(implication_probabilities[ant] <= similarity for ant in candidate_antecedents):
                    if r == 1:
                        insignificant_feature_values = np.append(insignificant_feature_values, candidate_antecedents)
                    continue

                # Store itemsets with integer indices for fast support calculation
                itemsets_with_indices.append(
                    candidate_antecedents.tolist() if isinstance(candidate_antecedents, np.ndarray) else list(candidate_antecedents)
                )

    logger.info(f"Found {len(itemsets_with_indices)} itemsets")

    if len(itemsets_with_indices) == 0:
        logger.info("No itemsets found")
        return {'itemsets': [], 'statistics': {}}

    # Calculate support using batch processing with optional parallelization
    logger.info("Calculating support values")
    transaction_array = autoencoder.input_vectors.to_numpy()
    num_transactions = len(transaction_array)

    # Batch calculate support for all itemsets (with optional parallelization)
    all_supports = calculate_itemset_metrics(
        itemsets_with_indices, transaction_array, num_workers=num_workers
    )

    # Build final itemsets with support values
    final_itemsets = []
    for itemset_indices, support in zip(itemsets_with_indices, all_supports):
        # Convert indices to human-readable format
        itemset = [
            {'feature': autoencoder.feature_values[idx].split('__', 1)[0],
             'value': autoencoder.feature_values[idx].split('__', 1)[1]}
            for idx in itemset_indices
        ]

        final_itemsets.append({'itemset': itemset, 'support': support})

    # Calculate statistics
    logger.info("Calculating aggregate statistics")
    avg_support = float(round(np.mean([item['support'] for item in final_itemsets]), 3))
    stats = {'itemset_count': len(final_itemsets), 'average_support': avg_support}

    logger.info(f"Mining complete: {len(final_itemsets)} itemsets with avg support={avg_support:.3f}")

    return {'itemsets': final_itemsets, 'statistics': stats}


def extract_significant_features_and_ignored_indices(features_of_interest, autoencoder):
    feature_value_indices = autoencoder.feature_value_indices
    feature_values = autoencoder.feature_values

    if not (features_of_interest and type(features_of_interest) == list and len(features_of_interest) > 0):
        return feature_value_indices, []

    value_constraints = defaultdict(set)
    interest_features = set()

    for f in features_of_interest:
        if isinstance(f, str):
            interest_features.add(f)
        elif isinstance(f, dict):
            for k, v in f.items():
                interest_features.add(k)
                value_constraints[k].add(v)

    # Significant features
    significant_features = [f for f in feature_value_indices if f['feature'] in interest_features]

    # Indices to ignore from constrained features
    values_to_ignore = [
        i for f in feature_value_indices if f['feature'] in value_constraints
        for i in range(f['start'], f['end'])
        if feature_values[i].split('__', 1)[-1] not in value_constraints[f['feature']]
    ]

    return significant_features, values_to_ignore


def _mark_features(unmarked_test_vector, features, insignificant_feature_values):
    """
    Create a list of test vectors by marking the given features in the unmarked test vector.
    This optimized version processes features in bulk using NumPy operations.
    """
    if unmarked_test_vector is None:
        return np.empty((0, 0), dtype=float), []

    unmarked = np.asarray(unmarked_test_vector)
    if unmarked.ndim != 1:
        raise ValueError("`unmarked_test_vector` must be a 1D array-like.")
    input_vector_size = unmarked.shape[0]

    if not features:  # None or empty
        return np.empty((0, input_vector_size), dtype=unmarked.dtype), []

    # Normalize insignificant indices
    if insignificant_feature_values is None:
        insignificant_feature_values = np.array([], dtype=int)
    else:
        insignificant_feature_values = np.asarray(insignificant_feature_values, dtype=int).ravel()

    input_vector_size = unmarked_test_vector.shape[0]

    # Compute valid feature ranges excluding insignificant_feature_values
    feature_ranges = [
        np.setdiff1d(np.array(feature_range), insignificant_feature_values)
        for feature_range in features
    ]

    # Create all combinations of feature indices
    combinations = np.array(np.meshgrid(*feature_ranges)).T.reshape(-1, len(features))

    # Initialize test_vectors and candidate_antecedents
    n_combinations = combinations.shape[0]
    test_vectors = np.tile(unmarked_test_vector, (n_combinations, 1))
    candidate_antecedents = [[] for _ in range(n_combinations)]

    # Vectorized marking of test_vectors
    for i, feature_range in enumerate(features):
        # Get the feature range
        valid_indices = combinations[:, i]

        # Ensure indices are within bounds
        valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < input_vector_size)]

        # Mark test_vectors based on valid indices for the current feature
        for j, idx in enumerate(valid_indices):
            test_vectors[j, feature_range.start:feature_range.stop] = 0  # Set feature range to 0
            test_vectors[j, idx] = 1  # Mark the valid index with 1
            candidate_antecedents[j].append(idx)  # Append the index to the j-th test vector's antecedents

    # Convert lists of candidate_antecedents to numpy arrays
    candidate_antecedents = [np.array(lst) for lst in candidate_antecedents]
    return test_vectors, candidate_antecedents


def _initialize_input_vectors(input_vector_size, categories):
    """
    Initialize the input vectors with equal probabilities for each feature range.
    """
    vector_with_unmarked_features = np.zeros(input_vector_size)
    for category in categories:
        vector_with_unmarked_features[category['start']:category['end']] = 1 / (
                category['end'] - category['start'])
    return vector_with_unmarked_features


def _calculate_aggregate_stats(rules, dataset_coverage, num_transactions, quality_metrics):
    """
    Calculate aggregate statistics for a set of rules.

    :param rules: List of rules with quality metrics
    :param dataset_coverage: Boolean array indicating which transactions are covered
    :param num_transactions: Total number of transactions
    :param quality_metrics: List of quality metrics that were calculated
    :return: Dictionary of aggregate statistics
    """
    if not rules:
        return {}

    stats = {'rule_count': len(rules)}

    # Calculate averages for each requested metric
    for metric in quality_metrics:
        values = [rule[metric] for rule in rules if metric in rule]
        if values:
            stats[f'average_{metric}'] = float(round(np.mean(values), 3))

    # Always include rule_coverage and data_coverage
    if 'rule_coverage' in rules[0]:
        coverage_values = [rule['rule_coverage'] for rule in rules]
        stats['average_coverage'] = float(round(np.mean(coverage_values), 3))

    stats['data_coverage'] = float(round(np.sum(dataset_coverage) / num_transactions, 3))

    return stats
