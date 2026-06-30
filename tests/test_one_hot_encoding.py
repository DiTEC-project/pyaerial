import unittest
import pandas as pd
from aerial.data_preparation import _one_hot_encoding_with_feature_tracking


class TestOneHotEncoding(unittest.TestCase):

    def test_all_categorical(self):
        df = pd.DataFrame({
            'Color': pd.Series(['Red', 'Green', 'Blue'], dtype='category'),
            'Size': pd.Series(['S', 'M', 'L'], dtype='category')
        })
        encoded, _ = _one_hot_encoding_with_feature_tracking(df)

        self.assertIsNotNone(encoded)
        self.assertIn('Color__Red', encoded.columns)
        self.assertIn('Size__S', encoded.columns)
        self.assertEqual(encoded.shape[1], 6)  # 3 colors + 3 sizes

    def test_categorical_plus_low_cardinality_numeric(self):
        df = pd.DataFrame({
            'Color': ['Red', 'Green', 'Blue'],
            'Rating': [1, 2, 3]  # <= 10 unique → categorical
        })
        encoded, _ = _one_hot_encoding_with_feature_tracking(df)

        self.assertIsNotNone(encoded)
        self.assertIn('Color__Red', encoded.columns)
        self.assertIn('Rating__1', encoded.columns)
        self.assertIn('Rating__2', encoded.columns)

    def test_high_cardinality_numeric_is_discretized(self):
        df = pd.DataFrame({
            'Color': ['Red', 'Green', 'Blue', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'],
            'Weight': list(range(12))  # 12 unique values > 10 threshold → auto discretized
        })

        encoded, _ = _one_hot_encoding_with_feature_tracking(df)

        self.assertIsNotNone(encoded)
        self.assertIn('Color__Red', encoded.columns)
        # Weight should appear as interval bins, not raw integers
        self.assertTrue(any(col.startswith('Weight__') for col in encoded.columns))
        self.assertFalse(any(col == f'Weight__{i}' for i in range(12) for col in encoded.columns))


if __name__ == '__main__':
    unittest.main()
