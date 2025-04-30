"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

Includes the Aerial algorithm's source code for association rule (and frequent itemsets) extraction from a
trained Autoencoder (Neurosymbolic association rule mining from tabular data - https://arxiv.org/abs/2504.19354)
"""

import torch

from itertools import combinations

from aerial.model import AutoEncoder
import numpy as np
import logging

logger = logging.getLogger("aerial")


def generate_rules(autoencoder: AutoEncoder, ant_similarity=0.5, cons_similarity=0.8, max_antecedents=2,
                   target_class=None):
    """
    extract rules from a trained Autoencoder using Aerial+ algorithm
    @param target_class: if given a target class, generate rules with the target class on the right hand side only
    :param max_antecedents: max number of antecedents that the rules will contain
    :param cons_similarity: consequent simi
    :param ant_similarity:
    :param autoencoder:
    """
    if not autoencoder:
        logger.error("A trained Autoencoder has to be provided before generating rules.")
        return None

    logger.debug("Extracting association rules from the given trained Autoencoder ...")

    association_rules = []
    input_vector_size = autoencoder.encoder[0].in_features

    feature_value_indices = autoencoder.feature_value_indices
    target_range = range(input_vector_size)

    # If target_class is specified, narrow the target range and features
    # this is to do "constraint-based rule mining"
    if target_class:
        for feature in feature_value_indices:
            if feature["feature"] == target_class:
                target_range = range(feature["start"], feature["end"])
                break

    low_support_antecedents = np.array([])

    # Initialize input vectors
    unmarked_features = _initialize_input_vectors(input_vector_size, feature_value_indices)

    # Precompute target indices for softmax to speed things up
    feature_value_indices = [(cat['start'], cat['end']) for cat in feature_value_indices]
    softmax_ranges = feature_value_indices

    for r in range(1, max_antecedents + 1):
        if r == 2:
            softmax_ranges = [
                (start, end) for (start, end) in softmax_ranges
                if not all(idx in low_support_antecedents for idx in range(start, end))
            ]

        feature_combinations = list(combinations(softmax_ranges, r))  # Generate combinations

        # Vectorized model evaluation batch
        batch_vectors = []
        batch_candidate_antecedent_list = []

        for category_list in feature_combinations:
            test_vectors, candidate_antecedent_list = _mark_features(unmarked_features, list(category_list),
                                                                     low_support_antecedents)
            if len(test_vectors) > 0:
                batch_vectors.extend(test_vectors)
                batch_candidate_antecedent_list.extend(candidate_antecedent_list)

        if batch_vectors:
            batch_vectors = torch.tensor(np.array(batch_vectors), dtype=torch.float32)
            # Perform a single model evaluation for the batch
            implications_batch = autoencoder(batch_vectors, feature_value_indices).detach().numpy()
            for test_vector, implication_probabilities, candidate_antecedents \
                    in zip(batch_vectors, implications_batch, batch_candidate_antecedent_list):
                if len(candidate_antecedents) == 0:
                    continue

                # Identify low-support antecedents
                if any(implication_probabilities[ant] <= ant_similarity for ant in candidate_antecedents):
                    if r == 1:
                        low_support_antecedents = np.append(low_support_antecedents, candidate_antecedents)
                    continue

                # Identify high-support consequents
                consequent_list = [
                    prob_index for prob_index in target_range
                    if prob_index not in candidate_antecedents and
                       implication_probabilities[prob_index] >= cons_similarity
                ]

                if consequent_list:
                    new_rule = _get_rule(candidate_antecedents, consequent_list, autoencoder.feature_values)
                    for consequent in new_rule['consequents']:
                        association_rules.append({'antecedents': new_rule['antecedents'], 'consequent': consequent})

    logger.debug("%d association rules extracted.", len(association_rules))
    return association_rules


def generate_frequent_itemsets(autoencoder: AutoEncoder, similarity=0.5, max_length=2):
    """
    Generate frequent itemsets using the Aerial+ algorithm.
    """
    if not autoencoder:
        logger.error("A trained Autoencoder has to be provided before extracting frequent items.")
        return None

    logger.debug("Extracting frequent items from the given trained Autoencoder ...")

    frequent_itemsets = []
    input_vector_size = len(autoencoder.feature_values)

    low_support_antecedents = np.array([])

    feature_value_indices = autoencoder.feature_value_indices

    # Initialize input vectors once
    unmarked_features = _initialize_input_vectors(input_vector_size, feature_value_indices)

    # Precompute target indices for softmax
    feature_value_indices = [(cat['start'], cat['end']) for cat in feature_value_indices]
    softmax_ranges = feature_value_indices

    # Iteratively process combinations of increasing size
    for r in range(1, max_length + 1):
        softmax_ranges = [
            (start, end) for (start, end) in softmax_ranges
            if not all(idx in low_support_antecedents for idx in range(start, end))
        ]

        feature_combinations = list(combinations(softmax_ranges, r))  # Generate combinations

        # Vectorized model evaluation batch
        batch_vectors = []
        batch_candidate_antecedent_list = []

        for category_list in feature_combinations:
            test_vectors, candidate_antecedent_list = _mark_features(unmarked_features, list(category_list),
                                                                     low_support_antecedents)
            if len(test_vectors) > 0:
                batch_vectors.extend(test_vectors)
                batch_candidate_antecedent_list.extend(candidate_antecedent_list)
        if batch_vectors:
            batch_vectors = torch.tensor(np.array(batch_vectors), dtype=torch.float32)
            # Perform a single model evaluation for the batch
            implications_batch = autoencoder(batch_vectors, feature_value_indices).detach().numpy()
            for test_vector, implication_probabilities, candidate_antecedents \
                    in zip(batch_vectors, implications_batch, batch_candidate_antecedent_list):
                if len(candidate_antecedents) == 0:
                    continue

                # Identify low-support antecedents
                if any(implication_probabilities[ant] <= similarity for ant in candidate_antecedents):
                    if r == 1:
                        low_support_antecedents = np.append(low_support_antecedents, candidate_antecedents)
                    continue

                # Add to frequent itemsets
                frequent_itemsets.append(
                    [autoencoder.feature_values[idx] for idx in candidate_antecedents]
                )

    logger.debug("%d frequent itemsets extracted.", len(frequent_itemsets))
    return frequent_itemsets


def _mark_features(unmarked_test_vector, features, low_support_antecedents):
    """
    Create a list of test vectors by marking the given features in the unmarked test vector.
    This optimized version processes features in bulk using NumPy operations.
    """
    input_vector_size = unmarked_test_vector.shape[0]

    # Compute valid feature ranges excluding low_support_antecedents
    feature_ranges = [
        np.setdiff1d(np.arange(start, end), low_support_antecedents)
        for (start, end) in features
    ]

    # Create all combinations of feature indices
    combinations = np.array(np.meshgrid(*feature_ranges)).T.reshape(-1, len(features))

    # Initialize test_vectors and candidate_antecedents
    n_combinations = combinations.shape[0]
    test_vectors = np.tile(unmarked_test_vector, (n_combinations, 1))
    candidate_antecedents = [[] for _ in range(n_combinations)]

    # Vectorized marking of test_vectors
    for i, (start, end) in enumerate(features):
        # Get the feature range
        valid_indices = combinations[:, i]

        # Ensure indices are within bounds
        valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < input_vector_size)]

        # Mark test_vectors based on valid indices for the current feature
        for j, idx in enumerate(valid_indices):
            test_vectors[j, start:end] = 0  # Set feature range to 0
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


def _get_rule(antecedents, consequents, feature_values):
    """
    Find the corresponding feature value for the given antecedents and consequent that are indices in test vectors
    :param antecedents: a list of indices in the test vectors marking the antecedent locations
    :param consequents: an index in the test vector marking the consequent location
    :param feature_values: a list of string that keeps track of which neuron in the Autoencoder input corresponds
        to which feature value in the tabular data
    :return:
    """
    rule = {'antecedents': [], 'consequents': []}
    for antecedent in antecedents:
        rule['antecedents'].append(feature_values[antecedent])

    for consequent in consequents:
        rule['consequents'].append(feature_values[consequent])

    return rule
