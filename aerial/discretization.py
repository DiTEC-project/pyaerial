"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

This script includes different discretization methods for tabular data.

References:
    - Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization of continuous features.
      In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.
    - Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification learning.
      In Proceedings of the 13th International Joint Conference on Artificial Intelligence (pp. 1022-1027).
    - Kerber, R. (1992). ChiMerge: Discretization of numeric attributes.
      In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128). AAAI Press.
    - Garcia, S., Luengo, J., Sáez, J. A., Lopez, V., & Herrera, F. (2013). A survey of discretization techniques: Taxonomy and empirical analysis in supervised learning.
      IEEE Transactions on Knowledge and Data Engineering, 25(4), 734-750.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger("aerial")


def equal_frequency_discretization(df: pd.DataFrame, n_bins=5):
    """
    Detect numerical columns automatically and discretize them into n_bins intervals based on equal frequency.
    Intervals are represented as strings.

    This method ensures each bin contains approximately the same number of data points, which can be
    useful when the data distribution is skewed.

    Reference:
        Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization
        of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.

    :param df: tabular data in pandas DataFrame form
    :param n_bins: number of intervals (bins)
    :return: df with discrete columns
    """
    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        try:
            # Use labels=True to get string intervals
            df_discretized[col] = pd.qcut(df[col], q=n_bins, duplicates='drop')
            df_discretized[col] = df_discretized[col].astype(str)
        except ValueError:
            logger.debug(f"Column '{col}' could not be discretized due to insufficient unique values.")

    return df_discretized


def equal_width_discretization(df: pd.DataFrame, n_bins=5):
    """
    Detect numerical columns automatically and discretize them into n_bins intervals based on equal width.
    Intervals are represented as strings.

    This is one of the simplest unsupervised discretization methods that divides the range of values
    into equal-width bins.

    Reference:
        Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization
        of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.

    :param df: tabular data in pandas DataFrame form
    :param n_bins: number of intervals (bins)
    :return: df with discrete columns
    """
    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        try:
            df_discretized[col] = pd.cut(df[col], bins=n_bins)
            df_discretized[col] = df_discretized[col].astype(str)
        except ValueError:
            logger.debug(f"Column '{col}' could not be discretized due to insufficient unique values.")

    return df_discretized


def kmeans_discretization(df: pd.DataFrame, n_bins=5, random_state=42):
    """
    Discretize numerical columns using k-means clustering. Each cluster represents a bin,
    and interval boundaries are created at the midpoints between cluster centers.

    The k-means algorithm groups values into clusters, then creates interval boundaries
    by finding midpoints between consecutive cluster centers. This creates more meaningful
    bins than equal-width or equal-frequency when data has distinct natural clusters.

    Reference:
        Garcia, S., Luengo, J., Sáez, J. A., Lopez, V., & Herrera, F. (2013). A survey of
        discretization techniques: Taxonomy and empirical analysis in supervised learning.
        IEEE Transactions on Knowledge and Data Engineering, 25(4), 734-750.

    :param df: tabular data in pandas DataFrame form
    :param n_bins: number of clusters/bins
    :param random_state: random seed for reproducibility
    :return: df with discrete columns represented as intervals
    :raises ValueError: if all numerical columns fail to discretize
    """
    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        logger.warning("No numerical columns found to discretize.")
        return df_discretized

    successfully_discretized = []
    failed_columns = []

    for col in num_cols:
        try:
            # Remove NaN values for clustering
            valid_mask = df[col].notna()
            if valid_mask.sum() < n_bins:
                logger.debug(f"Column '{col}' has fewer valid values than n_bins. Skipping.")
                failed_columns.append(col)
                continue

            values = df.loc[valid_mask, col].values.reshape(-1, 1)

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=n_bins, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(values)

            # Sort cluster centers and create ordered labels
            sorted_centers_idx = np.argsort(kmeans.cluster_centers_.flatten())
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_centers_idx)}

            # Create interval boundaries based on midpoints between cluster centers
            centers = np.sort(kmeans.cluster_centers_.flatten())

            # Calculate boundaries as midpoints between consecutive centers
            boundaries = [-np.inf]
            for i in range(len(centers) - 1):
                midpoint = (centers[i] + centers[i + 1]) / 2
                boundaries.append(midpoint)
            boundaries.append(np.inf)

            # Use pd.cut to create intervals
            df_discretized[col] = pd.cut(df[col], bins=boundaries, duplicates='drop')
            df_discretized[col] = df_discretized[col].astype(str)
            successfully_discretized.append(col)

        except Exception as e:
            logger.debug(f"Column '{col}' could not be discretized using k-means: {e}")
            failed_columns.append(col)

    # Report results
    if len(successfully_discretized) == 0:
        raise ValueError(
            f"K-means discretization failed for all {len(num_cols)} numerical columns. "
            f"Failed columns: {failed_columns}. "
            f"This may be due to insufficient data points (need at least n_bins={n_bins} values per column)."
        )
    elif len(failed_columns) > 0:
        logger.warning(
            f"K-means discretization succeeded for {len(successfully_discretized)} columns but failed for "
            f"{len(failed_columns)} columns: {failed_columns}"
        )
    else:
        logger.info(f"Successfully discretized {len(successfully_discretized)} numerical columns using k-means.")

    return df_discretized


def entropy_based_discretization(df: pd.DataFrame, target_col: str, n_bins=5):
    """
    Supervised discretization using entropy minimization (MDLP - Minimum Description Length Principle).
    This method uses a decision tree to find optimal split points that minimize entropy with respect
    to the target variable.

    This is a supervised method that requires a target column and creates bins that are most
    informative for predicting the target.

    Reference:
        Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued
        attributes for classification learning. In Proceedings of the 13th International Joint
        Conference on Artificial Intelligence (pp. 1022-1027).

    :param df: tabular data in pandas DataFrame form
    :param target_col: name of the target column for supervised discretization
    :param n_bins: maximum number of bins (decision tree max_leaf_nodes)
    :return: df with discrete columns
    :raises ValueError: if target column not found or all numerical columns fail to discretize
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude target if it's numeric
    num_cols = [col for col in num_cols if col != target_col]

    if len(num_cols) == 0:
        logger.warning("No numerical columns found to discretize (excluding target).")
        return df_discretized

    successfully_discretized = []
    failed_columns = []

    for col in num_cols:
        try:
            # Remove rows with NaN in either column or target
            valid_mask = df[col].notna() & df[target_col].notna()
            if valid_mask.sum() < n_bins:
                logger.debug(f"Column '{col}' has insufficient valid values. Skipping.")
                failed_columns.append(col)
                continue

            X = df.loc[valid_mask, col].values.reshape(-1, 1)
            y = df.loc[valid_mask, target_col].values

            # Use decision tree to find optimal splits
            tree = DecisionTreeClassifier(max_leaf_nodes=n_bins, random_state=42)
            tree.fit(X, y)

            # Get threshold values from the tree
            thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
            thresholds = np.unique(np.sort(thresholds))

            if len(thresholds) == 0:
                logger.debug(f"No splits found for column '{col}'. Using single bin.")
                # Create a single bin covering all values as a range
                min_val = df[col].min()
                max_val = df[col].max()
                bins = [min_val - 0.001, max_val + 0.001]
                df_discretized[col] = pd.cut(df[col], bins=bins, duplicates='drop')
                df_discretized[col] = df_discretized[col].astype(str)
                successfully_discretized.append(col)
                continue

            # Create bins using thresholds
            bins = np.concatenate([[-np.inf], thresholds, [np.inf]])
            df_discretized[col] = pd.cut(df[col], bins=bins, duplicates='drop')
            df_discretized[col] = df_discretized[col].astype(str)
            successfully_discretized.append(col)

        except Exception as e:
            logger.debug(f"Column '{col}' could not be discretized using entropy-based method: {e}")
            failed_columns.append(col)

    # Report results
    if len(successfully_discretized) == 0:
        raise ValueError(
            f"Entropy-based discretization failed for all {len(num_cols)} numerical columns. "
            f"Failed columns: {failed_columns}. "
            f"This may be due to insufficient data or no correlation with the target variable."
        )
    elif len(failed_columns) > 0:
        logger.warning(
            f"Entropy-based discretization succeeded for {len(successfully_discretized)} columns but failed for "
            f"{len(failed_columns)} columns: {failed_columns}"
        )
    else:
        logger.info(f"Successfully discretized {len(successfully_discretized)} numerical columns using entropy-based method.")

    return df_discretized


def chimerge_discretization(df: pd.DataFrame, target_col: str, max_bins=5, significance_level=0.05):
    """
    ChiMerge discretization algorithm that merges adjacent intervals based on chi-square statistic.
    This supervised method starts with many intervals and iteratively merges the most similar
    adjacent pairs until a stopping criterion is met.

    The algorithm computes chi-square statistics between adjacent intervals and merges those with
    the lowest chi-square value (most similar) until the desired number of bins is reached or
    the chi-square threshold is exceeded.

    Reference:
        Kerber, R. (1992). ChiMerge: Discretization of numeric attributes. In Proceedings of
        the tenth national conference on Artificial intelligence (pp. 123-128). AAAI Press.

    :param df: tabular data in pandas DataFrame form
    :param target_col: name of the target column for supervised discretization
    :param max_bins: maximum number of bins to create
    :param significance_level: chi-square significance level for merging (default 0.05)
    :return: df with discrete columns
    :raises ValueError: if target column not found or all numerical columns fail to discretize
    """
    from scipy.stats import chi2

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [col for col in num_cols if col != target_col]

    if len(num_cols) == 0:
        logger.warning("No numerical columns found to discretize (excluding target).")
        return df_discretized

    successfully_discretized = []
    failed_columns = []

    for col in num_cols:
        try:
            # Remove rows with NaN
            valid_mask = df[col].notna() & df[target_col].notna()
            if valid_mask.sum() < max_bins:
                logger.debug(f"Column '{col}' has insufficient valid values. Skipping.")
                failed_columns.append(col)
                continue

            values = df.loc[valid_mask, col].values
            target = df.loc[valid_mask, target_col].values

            # Get unique classes
            classes = np.unique(target)
            n_classes = len(classes)

            # Initialize with sorted unique values as boundaries
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_target = target[sorted_indices]

            # Create initial intervals (one per unique value, up to a reasonable limit)
            unique_values = np.unique(sorted_values)
            if len(unique_values) > 100:
                # Start with percentile-based bins if too many unique values
                percentiles = np.linspace(0, 100, min(100, len(unique_values)))
                boundaries = np.percentile(sorted_values, percentiles)
                boundaries = np.unique(boundaries)
            else:
                boundaries = unique_values

            # Create frequency table for each interval
            def get_interval_frequencies(boundaries, values, target, classes):
                intervals = []
                for i in range(len(boundaries) - 1):
                    mask = (values >= boundaries[i]) & (values < boundaries[i + 1])
                    freq = np.array([np.sum((target[mask] == c)) for c in classes])
                    if np.sum(freq) > 0:
                        intervals.append({
                            'start': boundaries[i],
                            'end': boundaries[i + 1],
                            'freq': freq
                        })
                # Handle last interval (inclusive on both ends)
                mask = values >= boundaries[-1]
                freq = np.array([np.sum((target[mask] == c)) for c in classes])
                if np.sum(freq) > 0:
                    intervals.append({
                        'start': boundaries[-1],
                        'end': np.inf,
                        'freq': freq
                    })
                return intervals

            intervals = get_interval_frequencies(boundaries, sorted_values, sorted_target, classes)

            # Chi-square computation between adjacent intervals
            def compute_chi_square(freq1, freq2):
                # Combine frequencies
                combined = freq1 + freq2
                total1 = np.sum(freq1)
                total2 = np.sum(freq2)
                total = total1 + total2

                if total == 0:
                    return np.inf

                chi_sq = 0
                for i in range(len(freq1)):
                    expected1 = combined[i] * total1 / total
                    expected2 = combined[i] * total2 / total

                    if expected1 > 0:
                        chi_sq += (freq1[i] - expected1) ** 2 / expected1
                    if expected2 > 0:
                        chi_sq += (freq2[i] - expected2) ** 2 / expected2

                return chi_sq

            # Critical chi-square value
            critical_value = chi2.ppf(1 - significance_level, n_classes - 1)

            # Merge intervals until we reach max_bins
            while len(intervals) > max_bins:
                # Find pair with minimum chi-square
                min_chi_sq = np.inf
                min_idx = -1

                for i in range(len(intervals) - 1):
                    chi_sq = compute_chi_square(intervals[i]['freq'], intervals[i + 1]['freq'])
                    if chi_sq < min_chi_sq:
                        min_chi_sq = chi_sq
                        min_idx = i

                # Stop if minimum chi-square exceeds critical value
                if min_chi_sq > critical_value and len(intervals) <= max_bins * 2:
                    break

                # Merge intervals at min_idx and min_idx+1
                if min_idx >= 0:
                    merged = {
                        'start': intervals[min_idx]['start'],
                        'end': intervals[min_idx + 1]['end'],
                        'freq': intervals[min_idx]['freq'] + intervals[min_idx + 1]['freq']
                    }
                    intervals = intervals[:min_idx] + [merged] + intervals[min_idx + 2:]

            # Create bins from intervals
            bin_edges = [intervals[0]['start']]
            for interval in intervals:
                if interval['end'] != np.inf:
                    bin_edges.append(interval['end'])

            bin_edges = [-np.inf] + bin_edges + [np.inf]
            bin_edges = np.unique(bin_edges)

            # Apply discretization
            df_discretized[col] = pd.cut(df[col], bins=bin_edges, duplicates='drop')
            df_discretized[col] = df_discretized[col].astype(str)
            successfully_discretized.append(col)

        except Exception as e:
            logger.debug(f"Column '{col}' could not be discretized using ChiMerge: {e}")
            failed_columns.append(col)

    # Report results
    if len(successfully_discretized) == 0:
        raise ValueError(
            f"ChiMerge discretization failed for all {len(num_cols)} numerical columns. "
            f"Failed columns: {failed_columns}. "
            f"This may be due to insufficient data or no clear class separation."
        )
    elif len(failed_columns) > 0:
        logger.warning(
            f"ChiMerge discretization succeeded for {len(successfully_discretized)} columns but failed for "
            f"{len(failed_columns)} columns: {failed_columns}"
        )
    else:
        logger.info(f"Successfully discretized {len(successfully_discretized)} numerical columns using ChiMerge.")

    return df_discretized


def custom_bins_discretization(df: pd.DataFrame, bins_dict: dict):
    """
    Discretize numerical columns using custom bin edges specified by the user.
    This provides full control over the discretization boundaries.

    Example bins_dict:
        {
            'age': [0, 18, 30, 50, 100],
            'income': [0, 30000, 60000, 100000, np.inf]
        }

    Reference:
        Garcia, S., Luengo, J., Sáez, J. A., Lopez, V., & Herrera, F. (2013). A survey of
        discretization techniques: Taxonomy and empirical analysis in supervised learning.
        IEEE Transactions on Knowledge and Data Engineering, 25(4), 734-750.

    :param df: tabular data in pandas DataFrame form
    :param bins_dict: dictionary mapping column names to lists of bin edges
    :return: df with discrete columns
    """
    df_discretized = df.copy()

    for col, bins in bins_dict.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' is not numeric. Skipping.")
            continue

        try:
            df_discretized[col] = pd.cut(df[col], bins=bins, duplicates='drop')
            df_discretized[col] = df_discretized[col].astype(str)
        except Exception as e:
            logger.error(f"Error discretizing column '{col}' with custom bins: {e}")

    return df_discretized


def quantile_discretization(df: pd.DataFrame, n_bins=5, percentiles=None):
    """
    Discretize numerical columns based on quantiles (percentiles). This is similar to
    equal_frequency_discretization but allows custom percentile specification.

    If percentiles are not provided, uses equal-frequency binning. Otherwise, creates bins
    at the specified percentile values.

    Reference:
        Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization
        of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.

    :param df: tabular data in pandas DataFrame form
    :param n_bins: number of bins (used if percentiles not specified)
    :param percentiles: list of percentile values (e.g., [0, 25, 50, 75, 100])
    :return: df with discrete columns
    """
    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        try:
            if percentiles is not None:
                # Use custom percentiles
                bin_edges = np.percentile(df[col].dropna(), percentiles)
                bin_edges = np.unique(bin_edges)
                df_discretized[col] = pd.cut(df[col], bins=bin_edges, duplicates='drop', include_lowest=True)
            else:
                # Use equal frequency
                df_discretized[col] = pd.qcut(df[col], q=n_bins, duplicates='drop')

            df_discretized[col] = df_discretized[col].astype(str)
        except Exception as e:
            logger.debug(f"Column '{col}' could not be discretized: {e}")

    return df_discretized


def decision_tree_discretization(df: pd.DataFrame, target_col: str, max_depth=3, min_samples_leaf=5):
    """
    Supervised discretization using decision tree regression to find optimal split points.

    This method builds a decision tree regressor for each numerical feature using the target
    variable, then extracts the split points from the tree to create bins. Unlike entropy-based
    discretization which is designed for classification, this method works with both categorical
    and numerical target variables.

    For categorical targets, values are encoded numerically. The decision tree learns which
    splits best separate the target values, providing interpretable discretization boundaries.

    Reference:
        Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and unsupervised discretization
        of continuous features. In Machine Learning Proceedings 1995 (pp. 194-202). Morgan Kaufmann.

    :param df: tabular data in pandas DataFrame form
    :param target_col: name of the target column for supervised discretization
    :param max_depth: maximum depth of the decision tree (controls number of bins)
    :param min_samples_leaf: minimum samples required in each leaf node
    :return: df with discrete columns
    :raises ValueError: if target column not found or all numerical columns fail to discretize
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeRegressor

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude target if it's numeric
    num_cols = [col for col in num_cols if col != target_col]

    if len(num_cols) == 0:
        logger.warning("No numerical columns found to discretize (excluding target).")
        return df_discretized

    successfully_discretized = []
    failed_columns = []

    # Encode target if categorical
    target_values = df[target_col].copy()
    if not pd.api.types.is_numeric_dtype(target_values):
        label_encoder = LabelEncoder()
        target_encoded = label_encoder.fit_transform(target_values.dropna())
        target_values = pd.Series(target_encoded, index=target_values.dropna().index)

    for col in num_cols:
        try:
            # Remove rows with NaN in either column or target
            valid_mask = df[col].notna() & target_values.notna()
            if valid_mask.sum() < min_samples_leaf * 2:
                logger.debug(f"Column '{col}' has insufficient valid values. Skipping.")
                failed_columns.append(col)
                continue

            X = df.loc[valid_mask, col].values.reshape(-1, 1)
            y = target_values.loc[valid_mask].values

            # Build decision tree regressor
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            tree.fit(X, y)

            # Extract threshold values from the tree
            thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
            thresholds = np.unique(np.sort(thresholds))

            if len(thresholds) == 0:
                logger.debug(f"No splits found for column '{col}'. Using single bin.")
                # Create a single bin covering all values as a range
                min_val = df[col].min()
                max_val = df[col].max()
                bins = [min_val - 0.001, max_val + 0.001]
                df_discretized[col] = pd.cut(df[col], bins=bins, duplicates='drop')
                df_discretized[col] = df_discretized[col].astype(str)
                successfully_discretized.append(col)
                continue

            # Create bins using thresholds
            bins = np.concatenate([[-np.inf], thresholds, [np.inf]])
            df_discretized[col] = pd.cut(df[col], bins=bins, duplicates='drop')
            df_discretized[col] = df_discretized[col].astype(str)
            successfully_discretized.append(col)

        except Exception as e:
            logger.debug(f"Column '{col}' could not be discretized using decision tree: {e}")
            failed_columns.append(col)

    # Report results
    if len(successfully_discretized) == 0:
        raise ValueError(
            f"Decision tree discretization failed for all {len(num_cols)} numerical columns. "
            f"Failed columns: {failed_columns}. "
            f"This may be due to insufficient data or no clear patterns with the target variable."
        )
    elif len(failed_columns) > 0:
        logger.warning(
            f"Decision tree discretization succeeded for {len(successfully_discretized)} columns but failed for "
            f"{len(failed_columns)} columns: {failed_columns}"
        )
    else:
        logger.info(f"Successfully discretized {len(successfully_discretized)} numerical columns using decision tree.")

    return df_discretized


def zscore_discretization(df: pd.DataFrame, n_std=1.0):
    """
    Discretize numerical columns based on z-scores (standard deviations from the mean).
    Creates bins centered around the mean with boundaries at multiples of the standard deviation.

    This method is particularly useful for normally distributed data, creating interpretable
    bins based on statistical properties (e.g., values within 1 std, 1-2 std, >2 std from mean).

    Common z-score binning schemes:
    - n_std=1.0: Creates bins at [-inf, μ-2σ, μ-σ, μ+σ, μ+2σ, inf] (5 bins)
    - n_std=0.5: Creates bins at [-inf, μ-σ, μ-0.5σ, μ+0.5σ, μ+σ, inf] (5 bins)

    :param df: tabular data in pandas DataFrame form
    :param n_std: number of standard deviations for bin boundaries (default 1.0)
    :return: df with discrete columns based on z-score bins
    """
    df_discretized = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        try:
            # Calculate mean and std
            mean = df[col].mean()
            std = df[col].std()

            if std == 0:
                logger.debug(f"Column '{col}' has zero standard deviation. Skipping.")
                continue

            # Create bins based on standard deviations
            # Typical bins: very low, low, medium, high, very high
            bins = [
                -np.inf,
                mean - 2 * n_std * std,
                mean - n_std * std,
                mean + n_std * std,
                mean + 2 * n_std * std,
                np.inf
            ]

            df_discretized[col] = pd.cut(df[col], bins=bins, duplicates='drop')
            df_discretized[col] = df_discretized[col].astype(str)

        except Exception as e:
            logger.debug(f"Column '{col}' could not be discretized using z-score: {e}")

    return df_discretized
