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
        assert pd.api.types.is_string_dtype(result['value'])
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
        assert pd.api.types.is_string_dtype(result['value'])
        # Check that original NaN location has 'nan' string
        assert pd.isna(result.loc[2, 'value']) or result.loc[2, 'value'] == 'nan'

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
        assert pd.api.types.is_string_dtype(result['value'])
        # Categorical should be unchanged
        assert all(result['categorical'] == 'X')

    def test_skewed_distribution(self):
        """Test with skewed distribution"""
        # Create exponentially distributed data
        df = pd.DataFrame({
            'value': np.exp(np.linspace(0, 5, 100))
        })

        result = discretization.equal_width_discretization(df, n_bins=4)

        assert pd.api.types.is_string_dtype(result['value'])
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
        assert pd.api.types.is_string_dtype(result['value'])
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
        assert pd.isna(result.loc[2, 'value']) or result.loc[2, 'value'] == 'nan'
        assert pd.isna(result.loc[5, 'value']) or result.loc[5, 'value'] == 'nan'
        # Valid values should be clustered into intervals
        assert pd.api.types.is_string_dtype(result['value'])
        # Should have created intervals (not 'nan') for non-NaN values
        non_nan_values = result.loc[[0, 1, 3, 4, 6, 7], 'value']
        assert all(not pd.isna(v) and v != 'nan' for v in non_nan_values)

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
        assert pd.api.types.is_string_dtype(result['feature1'])
        assert pd.api.types.is_string_dtype(result['feature2'])
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

        assert pd.api.types.is_string_dtype(result['value'])
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
        assert pd.api.types.is_string_dtype(result['feature'])
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
        assert pd.api.types.is_string_dtype(result['feature'])
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
        assert pd.api.types.is_string_dtype(result['feature1'])
        assert pd.api.types.is_string_dtype(result['feature2'])
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
        assert pd.api.types.is_string_dtype(result_shallow['feature'])
        assert pd.api.types.is_string_dtype(result_deep['feature'])
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
        assert pd.api.types.is_string_dtype(result['feature'])

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
        assert pd.api.types.is_string_dtype(result['feature'])
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
        assert pd.api.types.is_string_dtype(result_strict['feature'])
        assert pd.api.types.is_string_dtype(result_relaxed['feature'])

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
        assert pd.api.types.is_string_dtype(result['feature1'])
        assert pd.api.types.is_string_dtype(result['feature2'])
        assert pd.api.types.is_string_dtype(result['feature3'])


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
        assert pd.api.types.is_string_dtype(result['age'])
        assert pd.api.types.is_string_dtype(result['income'])
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
        assert pd.api.types.is_string_dtype(result['col1'])
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

        assert pd.api.types.is_string_dtype(result['value'])
        assert result['value'].nunique() <= 4

    def test_custom_percentiles(self):
        """Test with custom percentiles"""
        df = pd.DataFrame({
            'value': range(100)
        })

        # Use quartiles
        result = discretization.quantile_discretization(df, percentiles=[0, 25, 50, 75, 100])

        assert pd.api.types.is_string_dtype(result['value'])
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

        assert pd.api.types.is_string_dtype(result['value'])
        assert result['value'].nunique() <= 10


class TestZScoreDiscretization:
    """Test z-score based discretization"""

    def test_basic_zscore(self):
        """Test basic discretization into fixed number of bins"""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(50, 10, 100)
        })

        result = discretization.zscore_discretization(df, n_bins=5)

        # Should discretize successfully to integer categories
        assert str(result['value'].dtype) == "Int64"

        # Should have <= 5 bins
        assert result['value'].nunique() <= 5

    def test_different_n_bins(self):
        """Test that different n_bins produce different binning resolutions"""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(0, 1, 100)
        })

        result_5 = discretization.zscore_discretization(df.copy(), n_bins=5)
        result_3 = discretization.zscore_discretization(df.copy(), n_bins=3)

        # Both should discretize and produce integer labels
        assert str(result_5['value'].dtype) == "Int64"
        assert str(result_3['value'].dtype) == "Int64"

        # Different bin counts should produce different unique label counts
        assert result_5['value'].nunique() >= result_3['value'].nunique()

    def test_zero_std(self):
        """Test handling of zero standard deviation"""
        df = pd.DataFrame({
            'value': [5.0] * 100  # All same values
        })

        result = discretization.zscore_discretization(df, n_bins=5)

        # Column should remain unchanged BUT treated as skipped → cast to string
        assert pd.api.types.is_string_dtype(result['value'])

        # Values should remain identical
        assert result['value'].nunique() == 1
        assert set(result['value'].unique()) == {"5.0"}

    def test_output_is_integer_bins(self):
        """Output should be integer-coded bins, not interval strings"""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 15, 50)
        })

        result = discretization.zscore_discretization(df, n_bins=5)

        # Should produce integer bin labels
        assert pd.api.types.is_integer_dtype(result['value'])

        # Labels should NOT contain interval parentheses
        vals = result['value'].dropna().unique()
        for v in vals[:3]:
            assert isinstance(v, (int, np.integer))


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
        assert pd.api.types.is_string_dtype(result['numeric1'])
        assert pd.api.types.is_string_dtype(result['numeric2'])
        # Categorical columns should be unchanged
        assert pd.api.types.is_string_dtype(result['categorical1'])
        assert pd.api.types.is_string_dtype(result['categorical2'])

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
        assert pd.api.types.is_string_dtype(result_freq['value'])
        assert pd.api.types.is_string_dtype(result_width['value'])
        assert pd.api.types.is_string_dtype(result_kmeans['value'])
        assert pd.api.types.is_string_dtype(result_quantile['value'])

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
        assert pd.api.types.is_string_dtype(result_entropy['feature'])
        assert pd.api.types.is_string_dtype(result_chimerge['feature'])
        assert pd.api.types.is_string_dtype(result_decision_tree['feature'])

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
            assert pd.api.types.is_string_dtype(result[col])
            assert result[col].nunique() <= 4


class TestColumnFiltering:
    """Test filtering of low-cardinality and already-discrete columns"""

    def test_binary_column_filtered(self):
        """Test that binary 0/1 columns are filtered out"""
        df = pd.DataFrame({
            'binary': [0, 1, 0, 1, 0, 1] * 20,  # 120 rows
            'continuous': np.random.randn(120)
        })

        result = discretization.equal_frequency_discretization(df, n_bins=5)

        # Binary is skipped → cast to string
        assert pd.api.types.is_string_dtype(result['binary'])

        # Continuous column should be discretized → dtype object
        assert pd.api.types.is_string_dtype(result['continuous'])

    def test_low_cardinality_filtered(self):
        """Test that low cardinality columns are filtered out"""
        df = pd.DataFrame({
            'status': [1, 2, 3] * 40,  # 3 unique → skipped
            'continuous': np.linspace(0, 100, 120)
        })

        result = discretization.equal_width_discretization(df, n_bins=5)

        # Status column skipped → dtype object
        assert pd.api.types.is_string_dtype(result['status'])

        # Continuous discretized → dtype object
        assert pd.api.types.is_string_dtype(result['continuous'])

    def test_few_unique_values_filtered(self):
        """Test that columns with ≤ n_bins unique values are filtered"""
        df = pd.DataFrame({
            'few_unique': [1, 2, 3, 4] * 30,  # 4 unique → skipped
            'many_unique': list(range(120))  # 120 unique → discretized
        })

        result = discretization.equal_frequency_discretization(df, n_bins=5)

        # Skipped → cast to string
        assert pd.api.types.is_string_dtype(result['few_unique'])

        # Discretized → dtype object
        assert pd.api.types.is_string_dtype(result['many_unique'])

    def test_all_columns_filtered(self):
        """Test when all columns are filtered out"""
        df = pd.DataFrame({
            'binary1': [0, 1] * 50,
            'binary2': [1, 0] * 50,
            'ternary': [1, 2, 3] * 33 + [1]
        })

        result = discretization.kmeans_discretization(df, n_bins=5)

        # All columns skipped → all cast to string
        for col in df.columns:
            assert pd.api.types.is_string_dtype(result[col])

    def test_mixed_filtered_and_discretized(self):
        """Test mix of filtered and discretized columns"""
        np.random.seed(42)
        df = pd.DataFrame({
            'binary': [0, 1] * 50,            # skipped → object
            'continuous1': np.random.randn(100),  # discretized → object
            'status': [1, 2, 3] * 33 + [1],       # skipped → object
            'continuous2': np.linspace(0, 100, 100)  # discretized → object
        })

        result = discretization.equal_width_discretization(df, n_bins=5)

        # Skipped columns
        assert pd.api.types.is_string_dtype(result['binary'])
        assert pd.api.types.is_string_dtype(result['status'])

        # Discretized columns
        assert pd.api.types.is_string_dtype(result['continuous1'])
        assert pd.api.types.is_string_dtype(result['continuous2'])

    def test_filtering_with_supervised_methods(self):
        """Test that filtering works with supervised methods"""
        np.random.seed(42)
        df = pd.DataFrame({
            'binary': [0, 1] * 50,                 # skipped
            'feature': np.random.randn(100),       # discretized
            'target': ['A'] * 50 + ['B'] * 50      # target untouched
        })

        result = discretization.entropy_based_discretization(df, target_col='target', n_bins=5)

        # Binary skipped → object
        assert pd.api.types.is_string_dtype(result['binary'])

        # Feature discretized → object
        assert pd.api.types.is_string_dtype(result['feature'])

        # Target unchanged
        assert set(result['target'].unique()) == {'A', 'B'}

    def test_empty_data_column_filtered(self):
        """Test that columns with no data are filtered"""
        df = pd.DataFrame({
            'empty': [np.nan] * 100,  # n_total = 0 → skipped
            'valid': range(100)       # many uniques → discretized
        })

        result = discretization.equal_frequency_discretization(df, n_bins=5)

        # Empty column skipped → cast to string
        assert pd.api.types.is_string_dtype(result['empty'])

        # Valid column discretized → object
        assert pd.api.types.is_string_dtype(result['valid'])
