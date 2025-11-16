"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

Comprehensive tests for discretization methods in aerial/discretization.py
"""
import numpy as np
import pandas as pd
import pytest

from aerial import discretization


class TestEqualFrequencyDiscretization:
    """Test equal frequency (quantile-based) discretization"""

    def test_basic_discretization(self):
        """Test basic equal frequency discretization"""
        df = pd.DataFrame({
            'value': np.arange(100),
            'categorical': ['A'] * 50 + ['B'] * 50
        })

        result = discretization.equal_frequency_discretization(df, n_bins=4)

        # Check that numerical column is discretized
        assert result['value'].dtype == object
        # Check that categorical column is unchanged
        assert all(result['categorical'].isin(['A', 'B']))
        # Check that we have approximately equal frequencies
        assert result['value'].nunique() <= 4

    def test_with_missing_values(self):
        """Test discretization with NaN values"""
        df = pd.DataFrame({
            'value': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })

        result = discretization.equal_frequency_discretization(df, n_bins=3)

        # Check that discretization happened (NaN is converted to string 'nan')
        assert result['value'].dtype == object
        # Check that original NaN location has 'nan' string
        assert result.loc[2, 'value'] == 'nan'

    def test_insufficient_unique_values(self):
        """Test when column has fewer unique values than bins"""
        df = pd.DataFrame({
            'value': [1, 1, 1, 2, 2, 2]
        })

        result = discretization.equal_frequency_discretization(df, n_bins=5)

        # Should handle gracefully
        assert result is not None
        assert len(result) == 6


class TestEqualWidthDiscretization:
    """Test equal width discretization"""

    def test_basic_discretization(self):
        """Test basic equal width discretization"""
        df = pd.DataFrame({
            'value': np.linspace(0, 100, 100),
            'categorical': ['X'] * 100
        })

        result = discretization.equal_width_discretization(df, n_bins=5)

        # Check that numerical column is discretized
        assert result['value'].dtype == object
        # Categorical should be unchanged
        assert all(result['categorical'] == 'X')

    def test_skewed_distribution(self):
        """Test with skewed distribution"""
        # Create exponentially distributed data
        df = pd.DataFrame({
            'value': np.exp(np.linspace(0, 5, 100))
        })

        result = discretization.equal_width_discretization(df, n_bins=4)

        assert result['value'].dtype == object
        assert result['value'].nunique() <= 4


class TestKMeansDiscretization:
    """Test k-means clustering-based discretization"""

    def test_basic_clustering(self):
        """Test basic k-means discretization"""
        # Create data with clear clusters
        df = pd.DataFrame({
            'value': list(range(10)) + list(range(50, 60)) + list(range(100, 110))
        })

        result = discretization.kmeans_discretization(df, n_bins=3, random_state=42)

        # Check discretization occurred
        assert result['value'].dtype == object
        # Should create 3 bins
        assert result['value'].nunique() == 3

    def test_with_missing_values(self):
        """Test k-means with NaN values"""
        df = pd.DataFrame({
            'value': [1.0, 2.0, np.nan, 50.0, 51.0, np.nan, 100.0, 101.0]
        })

        result = discretization.kmeans_discretization(df, n_bins=3, random_state=42)

        # After pd.cut, NaN values are converted to string 'nan'
        # Check that we have 'nan' strings where original NaN values were
        assert result.loc[2, 'value'] == 'nan'
        assert result.loc[5, 'value'] == 'nan'
        # Valid values should be clustered into intervals
        assert result['value'].dtype == object
        # Should have created intervals (not 'nan') for non-NaN values
        non_nan_values = result.loc[[0, 1, 3, 4, 6, 7], 'value']
        assert all(non_nan_values != 'nan')

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility"""
        df = pd.DataFrame({
            'value': np.random.randn(100)
        })

        result1 = discretization.kmeans_discretization(df.copy(), n_bins=3, random_state=42)
        result2 = discretization.kmeans_discretization(df.copy(), n_bins=3, random_state=42)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestEntropyBasedDiscretization:
    """Test supervised entropy-based discretization"""

    def test_basic_supervised_discretization(self):
        """Test basic entropy-based discretization with a target"""
        # Create synthetic data where low values correlate with class A, high with class B
        df = pd.DataFrame({
            'feature1': list(range(20)) + list(range(50, 70)),
            'feature2': np.random.randn(40),
            'target': ['A'] * 20 + ['B'] * 20
        })

        result = discretization.entropy_based_discretization(df, target_col='target', n_bins=3)

        # Numerical columns should be discretized
        assert result['feature1'].dtype == object
        assert result['feature2'].dtype == object
        # Target should be unchanged
        assert all(result['target'].isin(['A', 'B']))

    def test_missing_target_column(self):
        """Test error handling when target column doesn't exist"""
        df = pd.DataFrame({
            'value': range(10)
        })

        # Should raise ValueError when target not found
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            discretization.entropy_based_discretization(df, target_col='nonexistent', n_bins=3)

    def test_multiclass_target(self):
        """Test with multiple target classes"""
        df = pd.DataFrame({
            'value': list(range(30)),
            'target': ['A'] * 10 + ['B'] * 10 + ['C'] * 10
        })

        result = discretization.entropy_based_discretization(df, target_col='target', n_bins=4)

        assert result['value'].dtype == object
        assert result['value'].nunique() <= 4


class TestDecisionTreeDiscretization:
    """Test decision tree-based discretization"""

    def test_basic_decision_tree(self):
        """Test basic decision tree discretization"""
        df = pd.DataFrame({
            'feature': list(range(50)),
            'target': ['A'] * 25 + ['B'] * 25
        })

        result = discretization.decision_tree_discretization(df, target_col='target', max_depth=3)

        # Feature should be discretized
        assert result['feature'].dtype == object
        # Target unchanged
        assert all(result['target'].isin(['A', 'B']))

    def test_with_numerical_target(self):
        """Test decision tree discretization with numerical target"""
        df = pd.DataFrame({
            'feature': list(range(100)),
            'target': list(range(100))  # Numerical target
        })

        result = discretization.decision_tree_discretization(df, target_col='target', max_depth=4)

        # Feature should be discretized
        assert result['feature'].dtype == object
        # Target should remain numerical
        assert pd.api.types.is_numeric_dtype(result['target'])

    def test_with_categorical_target(self):
        """Test decision tree discretization with categorical target"""
        df = pd.DataFrame({
            'feature1': list(range(30)),
            'feature2': np.random.randn(30),
            'target': ['Low'] * 10 + ['Medium'] * 10 + ['High'] * 10
        })

        result = discretization.decision_tree_discretization(df, target_col='target', max_depth=3)

        # Features should be discretized
        assert result['feature1'].dtype == object
        assert result['feature2'].dtype == object
        # Target unchanged
        assert all(result['target'].isin(['Low', 'Medium', 'High']))

    def test_max_depth_parameter(self):
        """Test that max_depth controls complexity"""
        df = pd.DataFrame({
            'feature': list(range(100)),
            'target': ['A'] * 50 + ['B'] * 50
        })

        result_shallow = discretization.decision_tree_discretization(df, target_col='target', max_depth=2)
        result_deep = discretization.decision_tree_discretization(df, target_col='target', max_depth=5)

        # Both should discretize
        assert result_shallow['feature'].dtype == object
        assert result_deep['feature'].dtype == object
        # Deeper tree might create more bins (but not guaranteed)
        assert result_shallow['feature'].nunique() >= 1
        assert result_deep['feature'].nunique() >= 1

    def test_min_samples_leaf_parameter(self):
        """Test min_samples_leaf parameter"""
        df = pd.DataFrame({
            'feature': list(range(100)),
            'target': np.random.choice(['A', 'B', 'C'], 100)
        })

        result = discretization.decision_tree_discretization(
            df, target_col='target', max_depth=4, min_samples_leaf=10
        )

        # Should discretize successfully
        assert result['feature'].dtype == object

    def test_missing_target_column(self):
        """Test error handling when target doesn't exist"""
        df = pd.DataFrame({
            'feature': range(10)
        })

        # Should raise ValueError when target not found
        with pytest.raises(ValueError, match="Target column 'nonexistent' not found"):
            discretization.decision_tree_discretization(df, target_col='nonexistent', max_depth=3)


class TestChiMergeDiscretization:
    """Test ChiMerge discretization algorithm"""

    def test_basic_chimerge(self):
        """Test basic ChiMerge discretization"""
        df = pd.DataFrame({
            'feature': list(range(50)),
            'target': ['A'] * 25 + ['B'] * 25
        })

        result = discretization.chimerge_discretization(df, target_col='target', max_bins=4)

        # Feature should be discretized
        assert result['feature'].dtype == object
        assert result['feature'].nunique() <= 4
        # Target unchanged
        assert all(result['target'].isin(['A', 'B']))

    def test_significance_level(self):
        """Test different significance levels"""
        df = pd.DataFrame({
            'feature': list(range(100)),
            'target': ['A'] * 50 + ['B'] * 50
        })

        result_strict = discretization.chimerge_discretization(
            df, target_col='target', max_bins=10, significance_level=0.01
        )
        result_relaxed = discretization.chimerge_discretization(
            df, target_col='target', max_bins=10, significance_level=0.10
        )

        # Both should discretize
        assert result_strict['feature'].dtype == object
        assert result_relaxed['feature'].dtype == object

    def test_with_multiple_features(self):
        """Test ChiMerge with multiple numerical features"""
        df = pd.DataFrame({
            'feature1': list(range(30)),
            'feature2': list(range(30, 60)),
            'feature3': list(range(60, 90)),
            'target': ['A'] * 10 + ['B'] * 10 + ['C'] * 10
        })

        result = discretization.chimerge_discretization(df, target_col='target', max_bins=3)

        # All numerical features should be discretized
        assert result['feature1'].dtype == object
        assert result['feature2'].dtype == object
        assert result['feature3'].dtype == object


class TestCustomBinsDiscretization:
    """Test custom bin edges discretization"""

    def test_basic_custom_bins(self):
        """Test discretization with custom bin edges"""
        df = pd.DataFrame({
            'age': [5, 15, 25, 35, 45, 55, 65, 75],
            'income': [10000, 25000, 40000, 55000, 70000, 85000, 100000, 120000]
        })

        bins_dict = {
            'age': [0, 18, 30, 50, 100],
            'income': [0, 30000, 60000, 100000, np.inf]
        }

        result = discretization.custom_bins_discretization(df, bins_dict)

        # Both columns should be discretized
        assert result['age'].dtype == object
        assert result['income'].dtype == object
        # Check expected number of unique bins
        assert result['age'].nunique() == 4
        assert result['income'].nunique() == 4

    def test_partial_columns(self):
        """Test custom bins on subset of columns"""
        df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10, 20),
            'col3': range(20, 30)
        })

        bins_dict = {
            'col1': [0, 5, 10]
        }

        result = discretization.custom_bins_discretization(df, bins_dict)

        # Only col1 should be discretized
        assert result['col1'].dtype == object
        assert pd.api.types.is_numeric_dtype(result['col2'])
        assert pd.api.types.is_numeric_dtype(result['col3'])

    def test_nonexistent_column(self):
        """Test handling of nonexistent column in bins_dict"""
        df = pd.DataFrame({
            'value': range(10)
        })

        bins_dict = {
            'nonexistent': [0, 5, 10]
        }

        # Should not raise error, just skip
        result = discretization.custom_bins_discretization(df, bins_dict)
        pd.testing.assert_frame_equal(result, df)


class TestQuantileDiscretization:
    """Test quantile-based discretization"""

    def test_default_quantiles(self):
        """Test with default n_bins (equal frequency)"""
        df = pd.DataFrame({
            'value': range(100)
        })

        result = discretization.quantile_discretization(df, n_bins=4)

        assert result['value'].dtype == object
        assert result['value'].nunique() <= 4

    def test_custom_percentiles(self):
        """Test with custom percentiles"""
        df = pd.DataFrame({
            'value': range(100)
        })

        # Use quartiles
        result = discretization.quantile_discretization(df, percentiles=[0, 25, 50, 75, 100])

        assert result['value'].dtype == object
        assert result['value'].nunique() <= 4

    def test_extreme_percentiles(self):
        """Test with extreme percentiles"""
        df = pd.DataFrame({
            'value': range(100)
        })

        # Use deciles
        result = discretization.quantile_discretization(
            df, percentiles=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        )

        assert result['value'].dtype == object
        assert result['value'].nunique() <= 10


class TestZScoreDiscretization:
    """Test z-score based discretization"""

    def test_basic_zscore(self):
        """Test basic z-score discretization"""
        # Create normally distributed data
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(50, 10, 100)
        })

        result = discretization.zscore_discretization(df, n_std=1.0)

        # Should discretize successfully
        assert result['value'].dtype == object
        # Should have up to 5 bins (very low, low, medium, high, very high)
        assert result['value'].nunique() <= 5

    def test_different_n_std(self):
        """Test with different n_std parameter"""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(0, 1, 100)
        })

        result_1 = discretization.zscore_discretization(df.copy(), n_std=1.0)
        result_05 = discretization.zscore_discretization(df.copy(), n_std=0.5)

        # Both should discretize
        assert result_1['value'].dtype == object
        assert result_05['value'].dtype == object

    def test_zero_std(self):
        """Test handling of zero standard deviation"""
        df = pd.DataFrame({
            'value': [5.0] * 100  # All same values
        })

        result = discretization.zscore_discretization(df, n_std=1.0)

        # Column should remain unchanged (logged and skipped)
        # In this case, pd.cut won't be called, so it stays numeric
        assert len(result) == 100

    def test_output_is_ranges(self):
        """Test that output is interval ranges, not labels"""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 15, 50)
        })

        result = discretization.zscore_discretization(df, n_std=1.0)

        # Check that results are string representations of intervals
        sample_values = result['value'].dropna().unique()
        for val in sample_values[:3]:  # Check a few values
            # Should contain '(' or '[' and ']' indicating intervals
            assert ('(' in val or '[' in val) and ']' in val


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""

    def test_mixed_data_types(self):
        """Test discretization with mixed data types"""
        df = pd.DataFrame({
            'numeric1': np.random.randn(50),
            'numeric2': np.random.uniform(0, 100, 50),
            'categorical1': np.random.choice(['A', 'B', 'C'], 50),
            'categorical2': np.random.choice(['X', 'Y'], 50)
        })

        result = discretization.equal_frequency_discretization(df, n_bins=5)

        # Numerical columns should be discretized
        assert result['numeric1'].dtype == object
        assert result['numeric2'].dtype == object
        # Categorical columns should be unchanged
        assert result['categorical1'].dtype == object
        assert result['categorical2'].dtype == object

    def test_all_methods_on_same_data(self):
        """Test that all unsupervised methods work on the same dataset"""
        df = pd.DataFrame({
            'value': np.random.randn(100)
        })

        # Test all unsupervised methods
        result_freq = discretization.equal_frequency_discretization(df.copy(), n_bins=5)
        result_width = discretization.equal_width_discretization(df.copy(), n_bins=5)
        result_kmeans = discretization.kmeans_discretization(df.copy(), n_bins=5, random_state=42)
        result_quantile = discretization.quantile_discretization(df.copy(), n_bins=5)

        # All should successfully discretize
        assert result_freq['value'].dtype == object
        assert result_width['value'].dtype == object
        assert result_kmeans['value'].dtype == object
        assert result_quantile['value'].dtype == object

    def test_supervised_methods_on_classification_data(self):
        """Test supervised methods on classification-like data"""
        # Create data where feature correlates with target
        np.random.seed(42)
        feature = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(5, 1, 50)
        ])

        df = pd.DataFrame({
            'feature': feature,
            'target': ['Low'] * 50 + ['High'] * 50
        })

        result_entropy = discretization.entropy_based_discretization(df.copy(), 'target', n_bins=4)
        result_chimerge = discretization.chimerge_discretization(df.copy(), 'target', max_bins=4)
        result_decision_tree = discretization.decision_tree_discretization(df.copy(), 'target', max_depth=3)

        # All should discretize successfully
        assert result_entropy['feature'].dtype == object
        assert result_chimerge['feature'].dtype == object
        assert result_decision_tree['feature'].dtype == object

    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        df = pd.DataFrame()

        result = discretization.equal_frequency_discretization(df, n_bins=5)

        # Should return empty dataframe
        assert len(result) == 0

    def test_single_value_column(self):
        """Test discretization of column with single unique value"""
        df = pd.DataFrame({
            'constant': [5.0] * 100
        })

        result = discretization.equal_width_discretization(df, n_bins=5)

        # Should handle gracefully
        assert len(result) == 100

    def test_large_number_of_bins(self):
        """Test with more bins than unique values"""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })

        result = discretization.equal_frequency_discretization(df, n_bins=10)

        # Should create at most 5 bins
        assert result['value'].nunique() <= 5

    def test_preservation_of_index(self):
        """Test that dataframe index is preserved"""
        df = pd.DataFrame({
            'value': range(20)
        }, index=range(100, 120))

        result = discretization.equal_frequency_discretization(df, n_bins=4)

        # Index should be preserved
        assert all(result.index == df.index)

    def test_multiple_numeric_columns(self):
        """Test discretization of multiple numeric columns simultaneously"""
        df = pd.DataFrame({
            'col1': np.random.randn(50),
            'col2': np.random.uniform(0, 100, 50),
            'col3': np.random.exponential(2, 50),
            'col4': np.random.poisson(5, 50)
        })

        result = discretization.equal_frequency_discretization(df, n_bins=4)

        # All columns should be discretized
        for col in ['col1', 'col2', 'col3', 'col4']:
            assert result[col].dtype == object
            assert result[col].nunique() <= 4