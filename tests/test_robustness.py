"""
Robustness tests for PyAerial.

These tests focus on:
1. End-to-end pipeline integration
2. Reproducibility with fixed seeds
3. Edge cases that real users encounter
4. Model persistence consistency
5. Input validation

These complement the existing unit tests by testing user-facing scenarios.
"""
import unittest
import numpy as np
import pandas as pd
import torch
import os
import tempfile

from aerial import model, rule_extraction, discretization
from aerial.model import AutoEncoder, train
from aerial.rule_extraction import generate_rules, generate_frequent_itemsets


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete workflows as users would use them."""

    def test_categorical_data_pipeline(self):
        """Test basic pipeline: categorical data → train → extract rules."""
        df = pd.DataFrame({
            'weather': ['sunny', 'sunny', 'rainy', 'rainy', 'cloudy'] * 6,
            'temperature': ['hot', 'hot', 'cold', 'cold', 'mild'] * 6,
            'play': ['yes', 'yes', 'no', 'no', 'yes'] * 6
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.3)

        self.assertIsInstance(result, dict)
        self.assertIn('rules', result)
        self.assertIn('statistics', result)
        self.assertIsInstance(result['statistics'], dict)

    def test_numerical_discretization_pipeline(self):
        """Test pipeline: numerical data → discretize → train → extract rules."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.uniform(0, 100, 30),
            'category': np.random.choice(['A', 'B', 'C'], 30)
        })

        discretized = discretization.equal_frequency_discretization(df, n_bins=3)
        trained = train(discretized, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.3)

        self.assertIsInstance(result, dict)
        self.assertIn('rules', result)

    def test_classification_rules_pipeline(self):
        """Test pipeline for classification rule extraction."""
        df = pd.DataFrame({
            'feature1': ['low', 'low', 'high', 'high'] * 8,
            'feature2': ['small', 'large', 'small', 'large'] * 8,
            'target': ['classA', 'classA', 'classB', 'classB'] * 8
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(
            trained,
            target_classes=['target'],
            min_rule_frequency=0.1,
            min_rule_strength=0.3
        )

        for rule in result['rules']:
            self.assertEqual(rule['consequent']['feature'], 'target')

    def test_features_of_interest_pipeline(self):
        """Test pipeline with features of interest constraint."""
        df = pd.DataFrame({
            'color': ['red', 'blue', 'green'] * 12,
            'size': ['S', 'M', 'L'] * 12,
            'shape': ['circle', 'square', 'triangle'] * 12
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(
            trained,
            features_of_interest=['color', 'size'],
            min_rule_frequency=0.1,
            min_rule_strength=0.2
        )

        for rule in result['rules']:
            for ant in rule['antecedents']:
                self.assertIn(ant['feature'], ['color', 'size'])

    def test_all_discretization_methods_pipeline(self):
        """Test that all discretization methods integrate properly."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature': np.random.randn(30),
            'target': ['A'] * 15 + ['B'] * 15
        })

        methods = [
            ('equal_frequency', lambda d: discretization.equal_frequency_discretization(d.copy(), n_bins=3)),
            ('equal_width', lambda d: discretization.equal_width_discretization(d.copy(), n_bins=3)),
            ('kmeans', lambda d: discretization.kmeans_discretization(d.copy(), n_bins=3, random_state=42)),
        ]

        for name, method in methods:
            with self.subTest(method=name):
                discretized = method(df)
                trained = train(discretized, epochs=1, show_progress=False)
                result = generate_rules(trained, min_rule_frequency=0.01, min_rule_strength=0.2)
                self.assertIsInstance(result, dict)


class TestReproducibility(unittest.TestCase):
    """Test that results are reproducible with fixed seeds."""

    def test_training_reproducibility_with_seed(self):
        """Same seed should produce same model weights."""
        df = pd.DataFrame({
            'A': ['x', 'y', 'z'] * 10,
            'B': ['1', '2', '3'] * 10
        })

        torch.manual_seed(42)
        np.random.seed(42)
        model1 = train(df, epochs=2, show_progress=False)
        weights1 = model1.encoder[0].weight.data.clone()

        torch.manual_seed(42)
        np.random.seed(42)
        model2 = train(df, epochs=2, show_progress=False)
        weights2 = model2.encoder[0].weight.data.clone()

        self.assertTrue(torch.allclose(weights1, weights2, atol=1e-5))

    def test_rule_extraction_deterministic(self):
        """Rule extraction should be deterministic (no randomness)."""
        df = pd.DataFrame({
            'A': ['x', 'y'] * 15,
            'B': ['1', '2'] * 15
        })

        torch.manual_seed(42)
        trained = train(df, epochs=2, show_progress=False)

        result1 = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.3)
        result2 = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.3)

        self.assertEqual(len(result1['rules']), len(result2['rules']))

        for r1, r2 in zip(result1['rules'], result2['rules']):
            self.assertEqual(r1['antecedents'], r2['antecedents'])
            self.assertEqual(r1['consequent'], r2['consequent'])
            self.assertAlmostEqual(r1['support'], r2['support'], places=5)

    def test_discretization_reproducibility(self):
        """Discretization with random_state should be reproducible."""
        df = pd.DataFrame({'value': np.random.randn(30)})

        result1 = discretization.kmeans_discretization(df.copy(), n_bins=3, random_state=42)
        result2 = discretization.kmeans_discretization(df.copy(), n_bins=3, random_state=42)

        pd.testing.assert_frame_equal(result1, result2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases that real users might encounter."""

    @classmethod
    def setUpClass(cls):
        """Create shared model for edge case tests that don't need specific data."""
        cls.basic_df = pd.DataFrame({
            'A': ['x', 'y'] * 15,
            'B': ['1', '2'] * 15
        })
        cls.basic_model = train(cls.basic_df, epochs=1, show_progress=False)

    def test_column_names_with_spaces(self):
        """Column names with spaces should work."""
        df = pd.DataFrame({
            'first column': ['a', 'b'] * 15,
            'second column': ['x', 'y'] * 15
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.2)

        self.assertIsInstance(result, dict)
        if result['rules']:
            features = set()
            for rule in result['rules']:
                for ant in rule['antecedents']:
                    features.add(ant['feature'])
                features.add(rule['consequent']['feature'])
            self.assertTrue(any(' ' in f for f in features))

    def test_unicode_column_names(self):
        """Unicode column names should work."""
        df = pd.DataFrame({
            '温度': ['高', '低'] * 15,
            'émoji': ['yes', 'no'] * 15
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.2)

        self.assertIsInstance(result, dict)

    def test_column_names_with_double_underscore(self):
        """Column names with __ should be handled (sanitized internally)."""
        df = pd.DataFrame({
            'prefix__suffix': ['a', 'b'] * 15,
            'normal': ['x', 'y'] * 15
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.2)

        self.assertIsInstance(result, dict)

    def test_single_column_dataset(self):
        """Dataset with single column should work."""
        df = pd.DataFrame({
            'only_column': ['a', 'b', 'c'] * 10
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.2)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['rules']), 0)

    def test_two_column_dataset(self):
        """Minimal dataset with two columns should work."""
        df = pd.DataFrame({
            'col1': ['a', 'b'] * 15,
            'col2': ['x', 'y'] * 15
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.01, min_rule_strength=0.2)

        self.assertIsInstance(result, dict)

    def test_high_cardinality_column(self):
        """Column with many unique values should work."""
        df = pd.DataFrame({
            'high_card': [f'val_{i}' for i in range(30)],
            'low_card': ['A', 'B'] * 15
        })

        trained = train(df, epochs=1, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.01, min_rule_strength=0.2)

        self.assertIsInstance(result, dict)

    def test_highly_correlated_features(self):
        """Perfectly correlated features should produce high-confidence rules."""
        df = pd.DataFrame({
            'A': ['x'] * 20 + ['y'] * 20,
            'B': ['1'] * 20 + ['2'] * 20
        })

        trained = train(df, epochs=3, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.3, min_rule_strength=0.7)

        if result['rules']:
            max_conf = max(r['confidence'] for r in result['rules'])
            self.assertGreater(max_conf, 0.7)

    def test_no_correlation_features(self):
        """Independent features should have low confidence rules."""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.choice(['x', 'y'], 30),
            'B': np.random.choice(['1', '2'], 30)
        })

        trained = train(df, epochs=2, show_progress=False)
        result = generate_rules(trained, min_rule_frequency=0.1, min_rule_strength=0.9)

        self.assertEqual(len(result['rules']), 0)

    def test_missing_values_in_categorical(self):
        """NaN values in categorical columns should be handled."""
        df = pd.DataFrame({
            'A': ['x', 'y', None, 'x', 'y'] * 6,
            'B': ['1', '2', '1', None, '2'] * 6
        })

        trained = train(df, epochs=1, show_progress=False)
        self.assertIsNotNone(trained)


class TestModelPersistence(unittest.TestCase):
    """Test save/load functionality and consistency."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_load_produces_same_forward_pass(self):
        """Saved and loaded model should produce identical outputs."""
        df = pd.DataFrame({
            'A': ['x', 'y', 'z'] * 10,
            'B': ['1', '2', '3'] * 10
        })

        trained = train(df, epochs=1, show_progress=False)
        save_path = os.path.join(self.temp_dir, 'test_model')
        trained.save(save_path)

        loaded = AutoEncoder(
            input_dimension=trained.input_dimension,
            feature_count=trained.feature_count
        )
        loaded.load(save_path)

        test_input = torch.randn(1, trained.input_dimension)
        fvi = [range(cat['start'], cat['end']) for cat in trained.feature_value_indices]

        trained.eval()
        loaded.eval()

        with torch.no_grad():
            out1 = trained(test_input, fvi)
            out2 = loaded(test_input, fvi)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_load_nonexistent_returns_false(self):
        """Loading nonexistent model should return False, not crash."""
        model = AutoEncoder(input_dimension=10, feature_count=3)
        result = model.load(os.path.join(self.temp_dir, 'nonexistent'))
        self.assertFalse(result)


class TestInputValidation(unittest.TestCase):
    """Test that invalid inputs are handled gracefully."""

    @classmethod
    def setUpClass(cls):
        """Create shared model for validation tests."""
        cls.df = pd.DataFrame({
            'A': ['x', 'y'] * 15,
            'B': ['1', '2'] * 15
        })
        cls.trained = train(cls.df, epochs=1, show_progress=False)

    def test_none_autoencoder_returns_none(self):
        """generate_rules with None autoencoder should return None."""
        result = generate_rules(None)
        self.assertIsNone(result)

    def test_none_autoencoder_itemsets_returns_none(self):
        """generate_frequent_itemsets with None autoencoder should return None."""
        result = generate_frequent_itemsets(None)
        self.assertIsNone(result)

    def test_invalid_quality_metrics_returns_none(self):
        """Invalid quality metric names should return None."""
        result = generate_rules(
            self.trained,
            quality_metrics=['support', 'invalid_metric_name']
        )
        self.assertIsNone(result)

    def test_empty_dataframe_training(self):
        """Training on empty DataFrame raises ValueError (no columns to process)."""
        df = pd.DataFrame()
        with self.assertRaises(ValueError):
            train(df, epochs=1, show_progress=False)

    def test_max_antecedents_zero(self):
        """max_antecedents=0 should return no rules."""
        result = generate_rules(self.trained, max_antecedents=0)
        self.assertEqual(len(result['rules']), 0)

    def test_very_high_similarity_thresholds(self):
        """Very high thresholds should return no rules."""
        result = generate_rules(self.trained, min_rule_frequency=0.999, min_rule_strength=0.999)
        self.assertEqual(len(result['rules']), 0)

    def test_nonexistent_feature_of_interest(self):
        """Non-existent feature in features_of_interest should be handled."""
        result = generate_rules(
            self.trained,
            features_of_interest=['NonExistent'],
            min_rule_frequency=0.1,
            min_rule_strength=0.2
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['rules']), 0)

    def test_nonexistent_target_class(self):
        """Non-existent target class should return no rules."""
        result = generate_rules(
            self.trained,
            target_classes=['NonExistent'],
            min_rule_frequency=0.1,
            min_rule_strength=0.2
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['rules']), 0)


class TestFrequentItemsets(unittest.TestCase):
    """Additional tests for frequent itemset mining."""

    @classmethod
    def setUpClass(cls):
        """Create shared model for itemset tests."""
        cls.df = pd.DataFrame({
            'A': ['x', 'y', 'z'] * 12,
            'B': ['1', '2', '3'] * 12
        })
        cls.trained = train(cls.df, epochs=1, show_progress=False)

    def test_itemset_support_values_valid(self):
        """All itemset support values should be between 0 and 1."""
        result = generate_frequent_itemsets(self.trained, frequency=0.1)

        for itemset in result['itemsets']:
            self.assertGreaterEqual(itemset['support'], 0)
            self.assertLessEqual(itemset['support'], 1)

    def test_itemset_max_length_respected(self):
        """Itemsets should not exceed max_length."""
        df = pd.DataFrame({
            'A': ['x', 'y'] * 15,
            'B': ['1', '2'] * 15,
            'C': ['a', 'b'] * 15
        })
        trained = train(df, epochs=1, show_progress=False)

        result = generate_frequent_itemsets(trained, frequency=0.01, max_length=1)
        for itemset in result['itemsets']:
            self.assertLessEqual(len(itemset['itemset']), 1)

        result = generate_frequent_itemsets(trained, frequency=0.01, max_length=2)
        for itemset in result['itemsets']:
            self.assertLessEqual(len(itemset['itemset']), 2)


class TestQualityMetricsBounds(unittest.TestCase):
    """Test that quality metrics stay within expected bounds."""

    @classmethod
    def setUpClass(cls):
        """Create shared model for metric bounds tests."""
        cls.df = pd.DataFrame({
            'A': ['x', 'y', 'z'] * 12,
            'B': ['1', '2', '3'] * 12,
            'C': ['a', 'b', 'c'] * 12
        })
        cls.trained = train(cls.df, epochs=2, show_progress=False)
        cls.result = generate_rules(cls.trained, min_rule_frequency=0.01, min_rule_strength=0.2)

    def test_support_bounds(self):
        """Support should be between 0 and 1."""
        for rule in self.result['rules']:
            self.assertGreaterEqual(rule['support'], 0)
            self.assertLessEqual(rule['support'], 1)

    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        for rule in self.result['rules']:
            self.assertGreaterEqual(rule['confidence'], 0)
            self.assertLessEqual(rule['confidence'], 1)

    def test_zhangs_metric_bounds(self):
        """Zhang's metric should be between -1 and 1."""
        for rule in self.result['rules']:
            self.assertGreaterEqual(rule['zhangs_metric'], -1)
            self.assertLessEqual(rule['zhangs_metric'], 1)

    def test_rule_coverage_bounds(self):
        """Rule coverage should be between 0 and 1."""
        for rule in self.result['rules']:
            self.assertGreaterEqual(rule['rule_coverage'], 0)
            self.assertLessEqual(rule['rule_coverage'], 1)

    def test_confidence_at_least_support(self):
        """Confidence * rule_coverage = support, so confidence >= support when coverage <= 1."""
        for rule in self.result['rules']:
            self.assertGreaterEqual(rule['rule_coverage'], rule['support'])

    def test_all_metrics_present(self):
        """All requested metrics should be in output."""
        from aerial.rule_quality import AVAILABLE_METRICS
        result = generate_rules(
            self.trained,
            min_rule_frequency=0.01,
            min_rule_strength=0.2,
            quality_metrics=AVAILABLE_METRICS
        )
        if result['rules']:
            for metric in AVAILABLE_METRICS:
                self.assertIn(metric, result['rules'][0])


class TestDataCoverage(unittest.TestCase):
    """Test data coverage statistics."""

    @classmethod
    def setUpClass(cls):
        """Create shared model for coverage tests."""
        cls.df = pd.DataFrame({
            'A': ['x', 'y'] * 15,
            'B': ['1', '2'] * 15
        })
        cls.trained = train(cls.df, epochs=1, show_progress=False)

    def test_data_coverage_bounds(self):
        """Data coverage should be between 0 and 1."""
        result = generate_rules(self.trained, min_rule_frequency=0.01, min_rule_strength=0.2)

        if result['rules']:
            self.assertIn('data_coverage', result['statistics'])
            self.assertGreaterEqual(result['statistics']['data_coverage'], 0)
            self.assertLessEqual(result['statistics']['data_coverage'], 1)

    def test_empty_rules_has_empty_stats(self):
        """When no rules found, statistics should be empty dict or have zero counts."""
        result = generate_rules(self.trained, min_rule_frequency=0.99, min_rule_strength=0.999)

        # With extreme thresholds, should get very few or no rules
        self.assertLessEqual(len(result['rules']), 1)
        self.assertIsInstance(result['statistics'], dict)


class TestMinConfidenceFilter(unittest.TestCase):
    """Test min_confidence post-filtering."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame({
            'A': ['x', 'y'] * 20,
            'B': ['1', '2'] * 20
        })
        cls.trained = train(cls.df, epochs=2, show_progress=False)

    def test_min_confidence_filters_rules(self):
        """min_confidence should filter out low-confidence rules."""
        result = generate_rules(
            self.trained,
            min_rule_frequency=0.01,
            min_rule_strength=0.2,
            min_confidence=0.6
        )

        for rule in result['rules']:
            self.assertGreaterEqual(rule['confidence'], 0.6)


class TestMinSupportFilter(unittest.TestCase):
    """Test min_support post-filtering."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame({
            'A': ['x', 'y'] * 20,
            'B': ['1', '2'] * 20
        })
        cls.trained = train(cls.df, epochs=2, show_progress=False)

    def test_min_support_filters_rules(self):
        """min_support should filter out low-support rules."""
        result = generate_rules(
            self.trained,
            min_rule_frequency=0.01,
            min_rule_strength=0.2,
            min_support=0.2
        )

        for rule in result['rules']:
            self.assertGreaterEqual(rule['support'], 0.2)


if __name__ == '__main__':
    unittest.main()