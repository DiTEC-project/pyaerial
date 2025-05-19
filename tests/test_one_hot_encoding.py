import pandas as pd
import pytest
from unittest.mock import patch

from aerial.data_preparation import _one_hot_encoding_with_feature_tracking


def test_all_categorical():
    df = pd.DataFrame({
        'Color': pd.Series(['Red', 'Green', 'Blue'], dtype='category'),
        'Size': pd.Series(['S', 'M', 'L'], dtype='category')
    })
    encoded = _one_hot_encoding_with_feature_tracking(df)[0]

    assert encoded is not None
    assert 'Color__Red' in encoded.columns
    assert 'Size__S' in encoded.columns
    assert encoded.shape[1] == 6  # 3 colors + 3 sizes


def test_categorical_plus_low_cardinality_numeric():
    df = pd.DataFrame({
        'Color': ['Red', 'Green', 'Blue'],
        'Rating': [1, 2, 3]  # <= 10 unique → categorical
    })
    encoded = _one_hot_encoding_with_feature_tracking(df)[0]

    assert encoded is not None
    assert 'Color__Red' in encoded.columns
    assert 'Rating__1' in encoded.columns
    assert 'Rating__2' in encoded.columns
    assert pd.api.types.is_integer_dtype(encoded['Rating__1'])


def test_high_cardinality_numeric_triggers_error_and_none_return():
    df = pd.DataFrame({
        'Color': ['Red', 'Green', 'Blue', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue', 'Red', 'Green', 'Blue'],
        'Weight': list(range(12))  # 12 unique values > 10 threshold, treated as numerical, should return None
    })

    result = _one_hot_encoding_with_feature_tracking(df)[0]
    assert result is None
