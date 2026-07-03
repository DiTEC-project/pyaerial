"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

Includes the Aerial algorithm's source code for association rule (and frequent itemsets) extraction from a
trained Autoencoder (Neurosymbolic association rule mining from tabular data - https://proceedings.mlr.press/v284/karabulut25a.html)
"""
from collections import defaultdict

import torch

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

# Sentinel distinguishing "not passed" (mirror min_rule_strength) from an explicit `None`
# (disable the confidence post-filter entirely).
_MIRROR_MIN_RULE_STRENGTH = object()


def generate_rules(autoencoder: AutoEncoder, features_of_interest: list = None, min_rule_frequency=0.5,
                   min_rule_strength=0.8,
                   max_antecedents=2, target_classes=None, quality_metrics=None, num_workers=1,
                   filter_min_confidence=_MIRROR_MIN_RULE_STRENGTH, filter_min_support: float = 0.0001):
    """
    Extract association rules from a trained Autoencoder using Aerial+ algorithm.
    Rule quality metrics are calculated automatically and included in the output.

    :param autoencoder: a trained Autoencoder for ARM
    :param features_of_interest: list: only look for rules that have these features of interest on the antecedent side
        accepted form ["feature1", "feature2", {"feature3": "value1}, ...], either a feature name as str, or specific value
        of a feature in object form
    :param min_rule_frequency: minimum frequency threshold for patterns (default=0.5). Higher values yield fewer,
        more common patterns. Originally named 'ant_frequency' in the Aerial paper. This gates rule generation using
        the Autoencoder's own (approximate) reconstructed probabilities.
    :param min_rule_strength: minimum strength threshold for rules (default=0.8). Higher values yield fewer,
        stronger rules. Originally named 'cons_frequency' in the Aerial paper. This gates rule generation using
        the Autoencoder's own (approximate) reconstructed probabilities.
    :param max_antecedents: max number of antecedents that the rules will contain (default=2).
        Pass None for no limit: antecedent combinations are grown until none passes the frequency gate.
    :param target_classes: list: if given a list of target classes, generate rules with the target classes on the
        right hand side only, the content of the list is as same as features_of_interest
    :param quality_metrics: list of quality metrics to calculate. Default is ['support', 'confidence', 'zhangs_metric'].
        Available metrics: 'support', 'confidence', 'zhangs_metric', 'lift', 'conviction', 'yulesq', 'interestingness'
    :param num_workers: number of parallel workers for quality metric calculation (default=1 for sequential processing)
    :param filter_min_confidence: post-filter, applied after generation, to only keep rules whose *exact* confidence
        (computed from the real data, not the Autoencoder's approximation) is >= this value. If not passed, defaults
        to whatever `min_rule_strength` was used for this call (confidence and strength are both conditional-
        probability-like quantities, so mirroring is meaningful). Pass None to disable this post-filter entirely.
    :param filter_min_support: post-filter, applied after generation, to only keep rules whose *exact* support
        (computed from the real data, not the Autoencoder's approximation) is >= this value (default=0.0001, i.e.
        only excludes degenerate zero/near-zero support rules). Deliberately not mirrored to `min_rule_frequency`:
        rule support is antecedent-and-consequent support, which is mathematically <= antecedent-only frequency, so
        the two are not comparable at the same threshold. Set to None to disable this post-filter entirely.
    :return: dict with 'rules' (list of rules with quality metrics) and 'statistics' (aggregate stats)
    """
    if not autoencoder:
        logger.error("A trained Autoencoder has to be provided before generating rules.")
        return None

    if quality_metrics is not None:
        invalid = [m for m in quality_metrics if m not in AVAILABLE_METRICS]
        if invalid:
            logger.error(f"Invalid quality metrics: {invalid}. Available: {AVAILABLE_METRICS}")
            return None

    if filter_min_confidence is _MIRROR_MIN_RULE_STRENGTH:
        filter_min_confidence = min_rule_strength

    logger.info("Mining association rules...")
    result = _generate_rules_core(autoencoder, features_of_interest, min_rule_frequency, min_rule_strength,
                                  max_antecedents, target_classes, quality_metrics, num_workers)

    if filter_min_confidence is not None or filter_min_support is not None:
        result = _apply_post_filters(result, autoencoder, filter_min_confidence, filter_min_support, quality_metrics)

    logger.info(
        f"Mining complete: {len(result['rules'])} rules with avg support={result['statistics'].get('average_support', 0):.3f}, "
        f"avg confidence={result['statistics'].get('average_confidence', 0):.3f}")
    return result


def _apply_post_filters(result, autoencoder, filter_min_confidence, filter_min_support, quality_metrics):
    """Apply post-filters and recalculate statistics if rules were filtered."""
    rules = result['rules']
    filtered = [r for r in rules
                if (filter_min_confidence is None or r.get('confidence', 1.0) >= filter_min_confidence)
                and (filter_min_support is None or r.get('support', 1.0) >= filter_min_support)]

    if len(filtered) == len(rules):
        return result

    logger.info(f"Post-filtering: {len(rules)} -> {len(filtered)} rules")

    if not filtered:
        logger.info("No rules remaining after post-filtering. Consider tuning parameters: "
                    "https://pyaerial.readthedocs.io/en/latest/parameter_guide.html#quick-parameter-reference")
        return {'rules': [], 'statistics': {}}

    # Recalculate coverage
    transaction_array = autoencoder.input_vectors.to_numpy(copy=True)
    coverage = np.zeros(len(transaction_array), dtype=bool)
    for rule in filtered:
        indices = [autoencoder.feature_values.index(f"{a['feature']}__{a['value']}") for a in rule['antecedents']]
        coverage |= np.all(transaction_array[:, indices] == 1, axis=1)

    metrics = quality_metrics if quality_metrics else DEFAULT_RULE_METRICS.copy()
    return {'rules': filtered,
            'statistics': _calculate_aggregate_stats(filtered, coverage, len(transaction_array), metrics)}


def _generate_rules_core(autoencoder, features_of_interest, min_rule_frequency, min_rule_strength,
                         max_antecedents, target_classes, quality_metrics, num_workers):
    """
    Core rule generation logic without auto-tuning or post-filtering.
    This is the internal implementation used by both generate_rules and auto-tuning.
    """
    # Validate quality metrics
    if quality_metrics is None:
        quality_metrics = DEFAULT_RULE_METRICS.copy()
    else:
        invalid_metrics = [m for m in quality_metrics if m not in AVAILABLE_METRICS]
        if invalid_metrics:
            logger.error(f"Invalid quality metrics: {invalid_metrics}. Available: {AVAILABLE_METRICS}")
            return {'rules': [], 'statistics': {}}

    # Store rules with their integer indices for fast quality calculation
    rules_with_indices = []
    input_vector_size = autoencoder.encoder[0].in_features

    # No limit: antecedents can hold at most one value per feature
    if max_antecedents is None:
        max_antecedents = len(autoencoder.feature_value_indices)

    # process features of interest
    significant_features, insignificant_feature_values = extract_significant_features_and_ignored_indices(
        features_of_interest, autoencoder)

    feature_value_indices = autoencoder.feature_value_indices

    # Initialize input vectors with all equal probability per feature value
    unmarked_features = _initialize_input_vectors(input_vector_size, feature_value_indices)

    # Precompute index-to-feature-range mapping for fast feature conflict detection
    index_to_feature_range = {}
    for cat in feature_value_indices:
        for idx in range(cat['start'], cat['end']):
            index_to_feature_range[idx] = (cat['start'], cat['end'])

    # If target_classes are specified, narrow the target range
    significant_consequents, insignificant_consequent_values = extract_significant_features_and_ignored_indices(
        target_classes, autoencoder)
    significant_consequent_indices = [
        index
        for feature in significant_consequents
        for index in range(feature['start'], feature['end'])
        if index not in insignificant_consequent_values
    ]

    feature_value_indices = [range(cat['start'], cat['end']) for cat in feature_value_indices]

    transaction_array = autoencoder.input_vectors.to_numpy(copy=True)
    num_transactions = len(transaction_array)

    ignored = set(int(idx) for idx in np.asarray(insignificant_feature_values, dtype=int).ravel())
    significant_feature_values = [
        idx
        for feature in significant_features
        for idx in range(feature['start'], feature['end'])
        if idx not in ignored
    ]

    def _forward(antecedent_lists):
        test_vectors = np.tile(unmarked_features, (len(antecedent_lists), 1))
        for row, antecedents in enumerate(antecedent_lists):
            for idx in antecedents:
                start, end = index_to_feature_range[idx]
                test_vectors[row, start:end] = 0
                test_vectors[row, idx] = 1
        batch = torch.tensor(test_vectors, dtype=torch.float32).to(next(autoencoder.parameters()).device)
        return autoencoder(batch, feature_value_indices).detach().cpu().numpy()

    def _emit_rules(antecedents, implication_probabilities):
        antecedent_ranges = set(index_to_feature_range[idx] for idx in antecedents)
        for consequent_idx in significant_consequent_indices:
            if index_to_feature_range[consequent_idx] in antecedent_ranges:
                continue
            if implication_probabilities[consequent_idx] >= min_rule_strength:
                rules_with_indices.append({
                    'antecedent_indices': sorted(int(idx) for idx in antecedents),
                    'consequent_index': int(consequent_idx)
                })

    # Frequency of a feature value is the Autoencoder's own implication probability of that
    # value when it is marked in the input, as in the iterative Aerial+ algorithm
    frequent_feature_values = []
    single_antecedent_probs = []
    feature_value_frequencies = np.zeros(input_vector_size)
    if significant_feature_values:
        for idx, implication_probabilities in zip(
                significant_feature_values, _forward([[idx] for idx in significant_feature_values])):
            feature_value_frequencies[idx] = implication_probabilities[idx]
            if implication_probabilities[idx] > min_rule_frequency:
                frequent_feature_values.append(idx)
                single_antecedent_probs.append(implication_probabilities)

    if frequent_feature_values:
        # Implication probabilities per marked feature value, in log space for the
        # frequency estimation of antecedent combinations
        log_frequencies = np.log(np.maximum(feature_value_frequencies, 1e-12))
        log_implication_probs = {}
        for idx, implication_probabilities in zip(frequent_feature_values, single_antecedent_probs):
            _emit_rules((idx,), implication_probabilities)
            log_implication_probs[idx] = np.log(np.maximum(implication_probabilities, 1e-12))

        # Fixed order (descending frequency): every antecedent combination is built exactly
        # once, by extending it only with feature values that come later in this order
        frequent_feature_values.sort(key=lambda idx: -feature_value_frequencies[idx])
        feature_value_rank = {idx: pos for pos, idx in enumerate(frequent_feature_values)}

        # Grow antecedent combinations from frequent ones only: a combination reaches the
        # Autoencoder only if its estimated frequency passes the gate, and only combinations
        # that passed are extended further. Estimated frequency of an antecedent combination:
        # geomean over ordered pairs of P(b|a)·P(a), both being the Autoencoder's implication
        # probabilities when a is marked in the input.
        antecedent_combinations = [((idx,), 0.0) for idx in frequent_feature_values]

        for r in range(2, max_antecedents + 1):
            n_pairs = r * (r - 1)
            extended_combinations = []
            for antecedents, log_sum in antecedent_combinations:
                antecedent_ranges = set(index_to_feature_range[idx] for idx in antecedents)
                for new_antecedent in frequent_feature_values[feature_value_rank[antecedents[-1]] + 1:]:
                    if index_to_feature_range[new_antecedent] in antecedent_ranges:
                        continue
                    extended_log_sum = log_sum + sum(
                        log_implication_probs[idx][new_antecedent] + log_frequencies[idx]
                        + log_implication_probs[new_antecedent][idx] + log_frequencies[new_antecedent]
                        for idx in antecedents
                    )
                    if np.exp(extended_log_sum / n_pairs) <= min_rule_frequency:
                        continue
                    extended_combinations.append((antecedents + (new_antecedent,), extended_log_sum))

            if not extended_combinations:
                break

            implications_batch = _forward([antecedents for antecedents, _ in extended_combinations])
            for (antecedents, _), implication_probabilities in zip(extended_combinations, implications_batch):
                _emit_rules(antecedents, implication_probabilities)
            antecedent_combinations = extended_combinations

    if len(rules_with_indices) == 0:
        logger.info("No rules found. Consider tuning parameters: "
                    "https://pyaerial.readthedocs.io/en/latest/parameter_guide.html#quick-parameter-reference")
        return {'rules': [], 'statistics': {}}

    all_metrics = calculate_rule_metrics(
        rules_with_indices, transaction_array, quality_metrics, num_workers=num_workers
    )

    # Build final rules and calculate dataset coverage
    final_rules = []
    dataset_coverage = np.zeros(num_transactions, dtype=bool)

    for rule_idx, metrics in zip(rules_with_indices, all_metrics):
        ant_indices = rule_idx['antecedent_indices']
        cons_index = rule_idx['consequent_index']

        antecedent_mask = metrics.pop('_antecedent_mask')
        dataset_coverage |= antecedent_mask

        antecedents = [
            {'feature': autoencoder.feature_values[idx].split('__', 1)[0],
             'value': autoencoder.feature_values[idx].split('__', 1)[1]}
            for idx in ant_indices
        ]
        consequent = {
            'feature': autoencoder.feature_values[cons_index].split('__', 1)[0],
            'value': autoencoder.feature_values[cons_index].split('__', 1)[1]
        }

        rule = {
            'antecedents': antecedents,
            'consequent': consequent
        }
        rule.update(metrics)
        final_rules.append(rule)

    stats = _calculate_aggregate_stats(final_rules, dataset_coverage, num_transactions, quality_metrics)

    return {'rules': final_rules, 'statistics': stats}


def generate_frequent_itemsets(autoencoder: AutoEncoder, features_of_interest=None, frequency=0.5, max_length=2,
                               num_workers=1):
    """
    Generate frequent itemsets using the Aerial+ algorithm.
    Support values are calculated automatically and included in the output.

    :param autoencoder: a trained Autoencoder
    :param features_of_interest: list: only look for itemsets that have these features of interest
        accepted form ["feature1", "feature2", {"feature3": "value1}, ...], either a feature name as str, or specific value
        of a feature in object form
    :param frequency: minimum frequency threshold for itemsets (default=0.5). Originally named 'frequency' in the Aerial paper.
    :param max_length: max itemset length (default=2).
        Pass None for no limit: itemsets are grown until none passes the frequency threshold.
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

    # No limit: an itemset can hold at most one value per feature
    if max_length is None:
        max_length = len(autoencoder.feature_value_indices)

    # process features of interest
    significant_features, insignificant_feature_values = extract_significant_features_and_ignored_indices(
        features_of_interest, autoencoder)

    feature_value_indices = autoencoder.feature_value_indices

    # Initialize input vectors once
    unmarked_features = _initialize_input_vectors(input_vector_size, feature_value_indices)

    # Precompute index-to-feature-range mapping for fast feature conflict detection
    index_to_feature_range = {}
    for cat in feature_value_indices:
        for idx in range(cat['start'], cat['end']):
            index_to_feature_range[idx] = (cat['start'], cat['end'])

    feature_value_indices = [range(cat['start'], cat['end']) for cat in feature_value_indices]

    ignored = set(int(idx) for idx in np.asarray(insignificant_feature_values, dtype=int).ravel())
    significant_feature_values = [
        idx
        for feature in significant_features
        for idx in range(feature['start'], feature['end'])
        if idx not in ignored
    ]

    def _forward(itemset_lists):
        test_vectors = np.tile(unmarked_features, (len(itemset_lists), 1))
        for row, itemset in enumerate(itemset_lists):
            for idx in itemset:
                start, end = index_to_feature_range[idx]
                test_vectors[row, start:end] = 0
                test_vectors[row, idx] = 1
        batch = torch.tensor(test_vectors, dtype=torch.float32).to(next(autoencoder.parameters()).device)
        return autoencoder(batch, feature_value_indices).detach().cpu().numpy()

    # Frequency of a feature value is the Autoencoder's own implication probability of that
    # value when it is marked in the input
    frequent_feature_values = []
    feature_value_frequencies = np.zeros(input_vector_size)
    if significant_feature_values:
        for idx, implication_probabilities in zip(
                significant_feature_values, _forward([[idx] for idx in significant_feature_values])):
            feature_value_frequencies[idx] = implication_probabilities[idx]
            if implication_probabilities[idx] > frequency:
                frequent_feature_values.append(idx)
                itemsets_with_indices.append([idx])

    if frequent_feature_values:
        # Fixed order (descending frequency): every itemset is built exactly once,
        # by extending it only with feature values that come later in this order
        frequent_feature_values.sort(key=lambda idx: -feature_value_frequencies[idx])
        feature_value_rank = {idx: pos for pos, idx in enumerate(frequent_feature_values)}

        # Grow itemsets from frequent ones only: an itemset is frequent if the implication
        # probability of every marked feature value in it stays above the threshold, and
        # only frequent itemsets are extended further
        itemset_combinations = [(idx,) for idx in frequent_feature_values]

        for r in range(2, max_length + 1):
            extended_combinations = []
            for itemset in itemset_combinations:
                itemset_ranges = set(index_to_feature_range[idx] for idx in itemset)
                for new_value in frequent_feature_values[feature_value_rank[itemset[-1]] + 1:]:
                    if index_to_feature_range[new_value] in itemset_ranges:
                        continue
                    extended_combinations.append(itemset + (new_value,))

            if not extended_combinations:
                break

            frequent_extended = []
            implications_batch = _forward([list(itemset) for itemset in extended_combinations])
            for itemset, implication_probabilities in zip(extended_combinations, implications_batch):
                if all(implication_probabilities[idx] > frequency for idx in itemset):
                    frequent_extended.append(itemset)
                    itemsets_with_indices.append(sorted(int(idx) for idx in itemset))
            itemset_combinations = frequent_extended

    logger.info(f"Found {len(itemsets_with_indices)} itemsets")

    if len(itemsets_with_indices) == 0:
        logger.info("No itemsets found. Consider tuning parameters: "
                    "https://pyaerial.readthedocs.io/en/latest/parameter_guide.html#quick-parameter-reference")
        return {'itemsets': [], 'statistics': {}}

    # Calculate support using batch processing with optional parallelization
    logger.info("Calculating support values")
    transaction_array = autoencoder.input_vectors.to_numpy(copy=True)
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
