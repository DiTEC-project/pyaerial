import unittest
import numpy as np
import pandas as pd

from aerial.model import train
from aerial.rule_extraction import generate_rules, generate_frequent_itemsets
from aerial.rule_quality import (
    calculate_interestingness,
    calculate_yulesq,
    calculate_lift,
    calculate_conviction,
    calculate_zhangs_metric,
    _calculate_rule_quality_from_indices,
    _calculate_itemset_support_from_indices,
    DEFAULT_RULE_METRICS,
    AVAILABLE_METRICS
)


class TestRuleQualityFunctions(unittest.TestCase):
    """Test individual rule quality calculation functions"""

    def test_calculate_lift(self):
        """Test lift calculation"""
        support = 0.5
        confidence = 0.75
        lift = calculate_lift(support, confidence)
        self.assertAlmostEqual(lift, 1.5, places=2)

    def test_calculate_lift_edge_case(self):
        """Test lift with edge case values"""
        support = 0.3
        confidence = 0.9
        lift = calculate_lift(support, confidence)
        self.assertAlmostEqual(lift, 3.0, places=2)

    def test_calculate_conviction(self):
        """Test conviction calculation"""
        support = 0.5
        confidence = 0.75
        conviction = calculate_conviction(support, confidence)
        # conviction = (1 - 0.5) / (1 - 0.75) = 0.5 / 0.25 = 2.0
        self.assertAlmostEqual(conviction, 2.0, places=2)

    def test_calculate_conviction_high_confidence(self):
        """Test conviction with high confidence"""
        support = 0.6
        confidence = 0.95
        conviction = calculate_conviction(support, confidence)
        # conviction = (1 - 0.6) / (1 - 0.95) = 0.4 / 0.05 = 8.0
        self.assertAlmostEqual(conviction, 8.0, places=2)

    def test_calculate_zhangs_metric(self):
        """Test Zhang's metric calculation"""
        support = 0.4
        support_ant = 0.5
        support_cons = 0.6
        zhangs = calculate_zhangs_metric(support, support_ant, support_cons)
        # numerator = 0.4 - 0.5 * 0.6 = 0.4 - 0.3 = 0.1
        # denominator = max(0.4 * (1 - 0.5), 0.5 * (0.6 - 0.4)) = max(0.2, 0.1) = 0.2
        # zhangs = 0.1 / 0.2 = 0.5
        self.assertAlmostEqual(zhangs, 0.5, places=2)

    def test_calculate_zhangs_metric_independence(self):
        """Test Zhang's metric when antecedent and consequent are independent"""
        support = 0.3
        support_ant = 0.5
        support_cons = 0.6
        zhangs = calculate_zhangs_metric(support, support_ant, support_cons)
        # support_ant * support_cons = 0.5 * 0.6 = 0.3 (equal to support, so independent)
        self.assertAlmostEqual(zhangs, 0.0, places=2)

    def test_calculate_interestingness(self):
        """Test interestingness calculation"""
        confidence = 0.8
        support = 0.5
        rhs_support = 0.625
        input_length = 100
        # interestingness = 0.8 * (0.5 / 0.625) * (1 - 0.5/100)
        #                 = 0.8 * 0.8 * 0.995 = 0.6368
        interest = calculate_interestingness(confidence, support, rhs_support, input_length)
        self.assertAlmostEqual(interest, 0.6368, places=3)

    def test_calculate_yulesq(self):
        """Test Yule's Q calculation"""
        full_count = 50
        not_ant_not_con = 30
        con_not_ant = 10
        ant_not_con = 10
        # ad = 50 * 30 = 1500
        # bc = 10 * 10 = 100
        # yulesq = (1500 - 100) / (1500 + 100) = 1400 / 1600 = 0.875
        yulesq = calculate_yulesq(full_count, not_ant_not_con, con_not_ant, ant_not_con)
        self.assertAlmostEqual(yulesq, 0.875, places=3)

    def test_calculate_yulesq_perfect_association(self):
        """Test Yule's Q with perfect positive association"""
        full_count = 100
        not_ant_not_con = 100
        con_not_ant = 0
        ant_not_con = 0
        # ad = 100 * 100 = 10000, bc = 0
        # yulesq = (10000 - 0) / (10000 + 0) = 1.0
        yulesq = calculate_yulesq(full_count, not_ant_not_con, con_not_ant, ant_not_con)
        self.assertAlmostEqual(yulesq, 1.0, places=3)


class TestRuleQualityFromIndices(unittest.TestCase):
    """Test optimized rule quality calculation using integer indices"""

    def setUp(self):
        """Create sample transaction data"""
        # Create a deterministic dataset for predictable test results
        self.transactions = pd.DataFrame({
            'Color': ['Red', 'Red', 'Blue', 'Blue', 'Red', 'Blue'],
            'Size': ['S', 'S', 'M', 'M', 'S', 'M'],
        })
        self.model = train(self.transactions, epochs=2)
        self.transaction_array = self.model.input_vectors.to_numpy()
        self.num_transactions = len(self.transaction_array)

    def test_calculate_rule_quality_with_default_metrics(self):
        """Test rule quality calculation with default metrics"""
        # Find indices for Color=Red and Size=S
        color_red_idx = self.model.feature_values.index('Color__Red')
        size_s_idx = self.model.feature_values.index('Size__S')

        antecedent_indices = [color_red_idx]
        consequent_index = size_s_idx

        result = _calculate_rule_quality_from_indices(
            antecedent_indices,
            consequent_index,
            self.transaction_array,
            self.num_transactions,
            DEFAULT_RULE_METRICS
        )

        # Check that default metrics are present
        self.assertIn('support', result)
        self.assertIn('confidence', result)
        self.assertIn('zhangs_metric', result)
        self.assertIn('rule_coverage', result)
        self.assertIn('_antecedent_mask', result)

        # Verify the values are floats and within valid ranges
        self.assertIsInstance(result['support'], float)
        self.assertIsInstance(result['confidence'], float)
        self.assertIsInstance(result['zhangs_metric'], float)

        # Support and confidence must be between 0 and 1
        self.assertTrue(0 <= result['support'] <= 1)
        self.assertTrue(0 <= result['confidence'] <= 1)
        # Zhang's metric must be between -1 and 1
        self.assertTrue(-1 <= result['zhangs_metric'] <= 1)

    def test_calculate_rule_quality_with_all_metrics(self):
        """Test rule quality calculation with all available metrics"""
        antecedent_indices = [0]
        consequent_index = 1

        result = _calculate_rule_quality_from_indices(
            antecedent_indices,
            consequent_index,
            self.transaction_array,
            self.num_transactions,
            AVAILABLE_METRICS
        )

        # Check that all metrics are present
        for metric in AVAILABLE_METRICS:
            self.assertIn(metric, result, f"Metric '{metric}' not found in result")

    def test_calculate_itemset_support(self):
        """Test itemset support calculation"""
        # Test with a simple itemset
        itemset_indices = [0, 1]

        support = _calculate_itemset_support_from_indices(
            itemset_indices,
            self.transaction_array,
            self.num_transactions
        )

        self.assertIsInstance(support, float)
        self.assertTrue(0 <= support <= 1)

    def test_calculate_itemset_support_single_item(self):
        """Test itemset support calculation with single item"""
        itemset_indices = [0]

        support = _calculate_itemset_support_from_indices(
            itemset_indices,
            self.transaction_array,
            self.num_transactions
        )

        # Calculate expected support manually
        expected_support = np.sum(self.transaction_array[:, 0] == 1) / self.num_transactions
        self.assertAlmostEqual(support, round(expected_support, 3), places=3)

    def test_empty_antecedents(self):
        """Test with empty antecedents"""
        antecedent_indices = []
        consequent_index = 0

        result = _calculate_rule_quality_from_indices(
            antecedent_indices,
            consequent_index,
            self.transaction_array,
            self.num_transactions,
            ['support', 'confidence']
        )

        self.assertIn('support', result)
        self.assertIn('confidence', result)

        # With empty antecedents, support equals consequent support
        expected_support = np.sum(self.transaction_array[:, consequent_index] == 1) / self.num_transactions
        self.assertAlmostEqual(result['support'], round(expected_support, 3), places=3)


class TestIntegratedRuleMining(unittest.TestCase):
    """Test integrated rule mining with automatic quality calculation"""

    def setUp(self):
        """Create sample transactions and train model"""
        self.transactions = pd.DataFrame({
            'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green'],
            'Size': ['S', 'M', 'L', 'S', 'L', 'M'],
            'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Square', 'Triangle']
        })
        self.model = train(self.transactions, epochs=2)

    def test_generate_rules_returns_dict_with_rules_and_stats(self):
        """Test that generate_rules returns proper structure"""
        result = generate_rules(self.model, ant_similarity=0.001, cons_similarity=0.001)

        self.assertIsInstance(result, dict)
        self.assertIn('rules', result)
        self.assertIn('statistics', result)

    def test_rules_have_quality_metrics(self):
        """Test that generated rules include quality metrics"""
        result = generate_rules(self.model, ant_similarity=0.001, cons_similarity=0.001)

        if len(result['rules']) > 0:
            rule = result['rules'][0]
            # Check structure
            self.assertIn('antecedents', rule)
            self.assertIn('consequent', rule)

            # Check default quality metrics
            self.assertIn('support', rule)
            self.assertIn('confidence', rule)
            self.assertIn('zhangs_metric', rule)
            self.assertIn('rule_coverage', rule)

            # Verify values are in valid ranges
            self.assertTrue(0 <= rule['support'] <= 1)
            self.assertTrue(0 <= rule['confidence'] <= 1)
            self.assertTrue(-1 <= rule['zhangs_metric'] <= 1)
            self.assertTrue(0 <= rule['rule_coverage'] <= 1)

    def test_statistics_calculated(self):
        """Test that aggregate statistics are calculated"""
        result = generate_rules(self.model, ant_similarity=0.001, cons_similarity=0.001)

        stats = result['statistics']
        if len(result['rules']) > 0:
            self.assertIn('rule_count', stats)
            self.assertIn('average_support', stats)
            self.assertIn('average_confidence', stats)

            # Verify rule_count matches actual number of rules
            self.assertEqual(stats['rule_count'], len(result['rules']))

    def test_custom_quality_metrics(self):
        """Test generating rules with custom quality metrics"""
        custom_metrics = ['support', 'confidence', 'lift', 'conviction']
        result = generate_rules(
            self.model,
            ant_similarity=0.001,
            cons_similarity=0.001,
            quality_metrics=custom_metrics
        )

        if len(result['rules']) > 0:
            rule = result['rules'][0]
            for metric in custom_metrics:
                self.assertIn(metric, rule, f"Custom metric '{metric}' not found in rule")

    def test_all_available_quality_metrics(self):
        """Test generating rules with all available quality metrics"""
        result = generate_rules(
            self.model,
            ant_similarity=0.001,
            cons_similarity=0.001,
            quality_metrics=AVAILABLE_METRICS
        )

        if len(result['rules']) > 0:
            rule = result['rules'][0]
            for metric in AVAILABLE_METRICS:
                self.assertIn(metric, rule, f"Metric '{metric}' not found in rule")

    def test_generate_frequent_itemsets_with_support(self):
        """Test that frequent itemsets include support values"""
        result = generate_frequent_itemsets(self.model, similarity=0.001)

        self.assertIsInstance(result, dict)
        self.assertIn('itemsets', result)
        self.assertIn('statistics', result)

        if len(result['itemsets']) > 0:
            itemset = result['itemsets'][0]
            self.assertIn('itemset', itemset)
            self.assertIn('support', itemset)

            # Check support is valid
            self.assertTrue(0 <= itemset['support'] <= 1)

    def test_itemset_statistics(self):
        """Test that itemset statistics are calculated"""
        result = generate_frequent_itemsets(self.model, similarity=0.001)

        stats = result['statistics']
        if len(result['itemsets']) > 0:
            self.assertIn('itemset_count', stats)
            self.assertIn('average_support', stats)

            # Verify itemset_count matches
            self.assertEqual(stats['itemset_count'], len(result['itemsets']))

            # Verify average support is calculated correctly
            avg_support = sum(item['support'] for item in result['itemsets']) / len(result['itemsets'])
            self.assertAlmostEqual(stats['average_support'], round(avg_support, 3), places=3)

    def test_invalid_quality_metrics(self):
        """Test that invalid quality metrics are rejected"""
        result = generate_rules(
            self.model,
            quality_metrics=['support', 'invalid_metric']
        )

        # Should return None on invalid metrics
        self.assertIsNone(result)

    def test_empty_result_structure(self):
        """Test structure when no rules are found"""
        # Use very high thresholds to get no rules
        result = generate_rules(self.model, ant_similarity=0.99, cons_similarity=0.99)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result['rules']), 0)
        self.assertIsInstance(result['statistics'], dict)

    def test_rule_coverage_calculation(self):
        """Test that rule_coverage (antecedent support) is calculated correctly"""
        result = generate_rules(self.model, ant_similarity=0.001, cons_similarity=0.001)

        if len(result['rules']) > 0:
            for rule in result['rules']:
                # rule_coverage must be present
                self.assertIn('rule_coverage', rule)

                # rule_coverage should be >= support (coverage of LHS >= coverage of LHS & RHS)
                self.assertGreaterEqual(rule['rule_coverage'], rule['support'])

                # rule_coverage must be a valid probability
                self.assertTrue(0 <= rule['rule_coverage'] <= 1)

                # Verify it's a float with proper rounding
                self.assertIsInstance(rule['rule_coverage'], float)

    def test_rule_coverage_always_included(self):
        """Test that rule_coverage is always included regardless of quality_metrics parameter"""
        # Test with different quality metrics configurations
        test_configs = [
            ['support', 'confidence'],
            ['lift', 'conviction'],
            ['zhangs_metric'],
            AVAILABLE_METRICS
        ]

        for quality_metrics in test_configs:
            result = generate_rules(
                self.model,
                ant_similarity=0.001,
                cons_similarity=0.001,
                quality_metrics=quality_metrics
            )

            if len(result['rules']) > 0:
                rule = result['rules'][0]
                self.assertIn('rule_coverage', rule,
                             f"rule_coverage missing when quality_metrics={quality_metrics}")


class TestBackwardCompatibility(unittest.TestCase):
    """Test that the API changes don't break existing functionality"""

    def setUp(self):
        """Create sample data"""
        self.transactions = pd.DataFrame({
            'A': ['1', '2', '1', '2'],
            'B': ['X', 'Y', 'X', 'Y']
        })
        self.model = train(self.transactions, epochs=1)

    def test_rule_structure_preserved(self):
        """Test that rule dictionary structure is preserved"""
        result = generate_rules(self.model, ant_similarity=0.001, cons_similarity=0.001)

        if len(result['rules']) > 0:
            rule = result['rules'][0]

            # Check antecedents format
            self.assertIsInstance(rule['antecedents'], list)
            for ant in rule['antecedents']:
                self.assertIn('feature', ant)
                self.assertIn('value', ant)

            # Check consequent format
            self.assertIsInstance(rule['consequent'], dict)
            self.assertIn('feature', rule['consequent'])
            self.assertIn('value', rule['consequent'])

    def test_default_metrics_constant(self):
        """Test that DEFAULT_RULE_METRICS contains expected values"""
        expected_defaults = ['support', 'confidence', 'zhangs_metric']
        self.assertEqual(DEFAULT_RULE_METRICS, expected_defaults)

    def test_available_metrics_constant(self):
        """Test that AVAILABLE_METRICS contains all expected metrics"""
        expected_metrics = ['support', 'confidence', 'zhangs_metric', 'lift', 'conviction', 'yulesq', 'interestingness']
        self.assertEqual(AVAILABLE_METRICS, expected_metrics)


class TestParallelization(unittest.TestCase):
    """Test parallelization support in rule quality calculation"""

    def setUp(self):
        """Create sample data"""
        self.transactions = pd.DataFrame({
            'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green'] * 10,
            'Size': ['S', 'M', 'L', 'S', 'L', 'M'] * 10,
            'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Square', 'Triangle'] * 10
        })
        self.model = train(self.transactions, epochs=2)

    def test_num_workers_parameter_rules(self):
        """Test that num_workers parameter is accepted by generate_rules"""
        result_sequential = generate_rules(
            self.model,
            ant_similarity=0.001,
            cons_similarity=0.001,
            num_workers=1
        )

        result_parallel = generate_rules(
            self.model,
            ant_similarity=0.001,
            cons_similarity=0.001,
            num_workers=2
        )

        # Both should succeed and return the same structure
        self.assertIsInstance(result_sequential, dict)
        self.assertIsInstance(result_parallel, dict)

        # Results should be consistent (same number of rules)
        if len(result_sequential['rules']) > 0 and len(result_parallel['rules']) > 0:
            self.assertEqual(len(result_sequential['rules']), len(result_parallel['rules']))

    def test_num_workers_parameter_itemsets(self):
        """Test that num_workers parameter is accepted by generate_frequent_itemsets"""
        result_sequential = generate_frequent_itemsets(
            self.model,
            similarity=0.001,
            num_workers=1
        )

        result_parallel = generate_frequent_itemsets(
            self.model,
            similarity=0.001,
            num_workers=2
        )

        # Both should succeed and return the same structure
        self.assertIsInstance(result_sequential, dict)
        self.assertIsInstance(result_parallel, dict)

        # Results should be consistent (same number of itemsets)
        if len(result_sequential['itemsets']) > 0 and len(result_parallel['itemsets']) > 0:
            self.assertEqual(len(result_sequential['itemsets']), len(result_parallel['itemsets']))

    def test_parallel_results_match_sequential(self):
        """Test that parallel processing produces same results as sequential"""
        result_seq = generate_rules(
            self.model,
            ant_similarity=0.001,
            cons_similarity=0.001,
            num_workers=1
        )

        result_par = generate_rules(
            self.model,
            ant_similarity=0.001,
            cons_similarity=0.001,
            num_workers=4
        )

        if len(result_seq['rules']) > 0:
            # Same number of rules
            self.assertEqual(len(result_seq['rules']), len(result_par['rules']))

            # Statistics should match
            self.assertEqual(result_seq['statistics']['rule_count'],
                           result_par['statistics']['rule_count'])


if __name__ == "__main__":
    unittest.main()