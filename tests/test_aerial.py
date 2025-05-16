import unittest
import pandas as pd

from aerial.model import AutoEncoder
from aerial.data_preparation import _one_hot_encoding_with_feature_tracking
from aerial.model import train
from aerial.rule_extraction import (
    generate_rules,
    generate_frequent_itemsets,
)


class TestAerialFunctions(unittest.TestCase):
    def setUp(self):
        """Create sample transactions and train an AutoEncoder"""
        self.transactions = pd.DataFrame({
            'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red'],
            'Size': ['S', 'M', 'L', 'S', 'L'],
            'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Square']
        })
        self.model = train(self.transactions, epochs=5)

    def test_one_hot_encoding_with_feature_tracking(self):
        """Test one-hot vector creation from transactions"""
        vector_list, feature_value_indices = _one_hot_encoding_with_feature_tracking(self.transactions)
        self.assertIsInstance(vector_list, pd.DataFrame)
        self.assertIsInstance(feature_value_indices, list)
        self.assertEqual(vector_list.shape[0], len(self.transactions))

    def test_train_autoencoder(self):
        """Test training an autoencoder model"""
        model = train(self.transactions, epochs=2)
        self.assertIsInstance(model, AutoEncoder)

    def test_generate_rules(self):
        """Test rule generation"""
        rules = generate_rules(self.model, ant_similarity=0.4, cons_similarity=0.6)
        self.assertIsInstance(rules, list)
        if rules:
            self.assertIn('antecedents', rules[0])
            self.assertIn('consequent', rules[0])

    def test_generate_frequent_itemsets(self):
        """Test frequent itemset generation"""
        itemsets = generate_frequent_itemsets(self.model, similarity=0.4)
        self.assertIsInstance(itemsets, list)
        if itemsets:
            self.assertIsInstance(itemsets[0], list)


if __name__ == "__main__":
    unittest.main()
