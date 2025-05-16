"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

This script include different discretization methods for tabular data
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("aerial")


def equal_frequency_discretization(df: pd.DataFrame, n_bins=5):
    """
    Detect numerical columns automatically and discretize them into n_bins intervals based on equal frequency.
    Intervals are represented as strings.

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
