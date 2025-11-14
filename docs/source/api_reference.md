# API Reference

This section lists the important classes and functions as part of the Aerial package.

## Model Module

### AutoEncoder

```python
AutoEncoder(input_dimension, feature_count, layer_dims=None)
```

Constructs an autoencoder designed for association rule mining on tabular data, based on the Neurosymbolic Association
Rule Mining method.

**Parameters**:

- `input_dimension` (int): Number of input features after one-hot encoding.
- `feature_count` (int): Original number of categorical features in the dataset.
- `layer_dims` (list of int, optional): User-specified hidden layer dimensions. If not provided, the model calculates a
  default architecture using a logarithmic reduction strategy (base 16).

**Behavior**:

- Automatically builds an under-complete autoencoder with a bottleneck at the original feature count.
- If no layer_dims are provided, the architecture is determined by reducing the input dimension using a geometric
  progression and creates `log₁₆(input_dimension)` layers in total.
- Uses Xavier initialization for weights and sets all biases to zero.
- Applies Tanh activation functions between layers, except the final encoder and decoder layers.

### train

```python
train(
    transactions,
    autoencoder=None,
    noise_factor=0.5,
    lr=5e-3,
    epochs=2,
    batch_size=2,
    loss_function=torch.nn.BCELoss(),
    num_workers=1,
    layer_dims=None,
    device=None,
    patience=20,
    delta=1e-4
)
```

Given a categorical tabular data in Pandas dataframe form, it one-hot encodes the data, vectorizes the one-hot encoded
version by also keeping track of start and end indices of vectors per column, and then trains the AutoEncoder model
using the one-hot encoded version.

If there are numerical features with less than 10 cardinality, it treats them as categorical features. If the
cardinality is more than 10, then it throws an error.

**Parameters**:

- `transactions` (pd.DataFrame): Tabular input data for training.
- `autoencoder` (AutoEncoder, optional): A preconstructed autoencoder instance. If not provided, one is created
  automatically.
- `noise_factor` (float, default=0.5): Controls the amount of Gaussian noise added to inputs during training (denoising effect).
- `lr` (float, default=5e-3): Learning rate for the Adam optimizer.
- `epochs` (int, default=2): Number of training epochs.
- `batch_size` (int, default=2): Number of samples per training batch.
- `loss_function` (torch.nn.Module, default=torch.nn.BCELoss()): Loss function to apply.
- `num_workers` (int, default=1): Number of subprocesses used for data loading.
- `layer_dims` (list of int, optional): Custom hidden layer dimensions for autoencoder construction.
- `device` (str, optional): Name of the device to run the Autoencoder model on, e.g., "cuda", "cpu" etc. The device option that is
  set here will also be used in the rule extraction stage. If not specified, uses CUDA if available, otherwise CPU.
- `patience` (int, default=20): Number of epochs to wait for improvement before early stopping.
- `delta` (float, default=1e-4): Minimum change in loss to qualify as an improvement for early stopping.

**Returns**: A trained instance of the AutoEncoder.

## Rule Extraction Module

### generate_rules

```python
generate_rules(
    autoencoder,
    features_of_interest=None,
    ant_similarity=0.5,
    cons_similarity=0.8,
    max_antecedents=2,
    target_classes=None,
    quality_metrics=['support', 'confidence', 'zhangs_metric'],
    num_workers=1
)
```

Extracts association rules from a trained AutoEncoder using the Aerial algorithm, with quality metrics calculated automatically using optimized batch processing.

**Parameters**:

- `autoencoder` (AutoEncoder): A trained autoencoder instance.
- `features_of_interest` (list, optional): only look for rules that have these features of interest on the antecedent
  side. Accepted form `["feature1", "feature2", {"feature3": "value1"}, ...]`, either a feature name as str, or specific
  value of a feature in object form
- `ant_similarity` (float, optional): Minimum similarity threshold for an antecedent to be considered frequent.
  Default=0.5
- `cons_similarity` (float, optional): Minimum probability threshold for a feature to qualify as a rule consequent.
  Default=0.8
- `max_antecedents` (int, optional): Maximum number of features allowed in the rule antecedent. Default=2
- `target_classes` (list, optional): When set, restricts rule consequents to the specified class(es) (constraint-based
  rule mining). The format of the list is same as the list format of `features_of_interest`.
- `quality_metrics` (list, optional): Quality metrics to calculate for each rule. Default=['support', 'confidence', 'zhangs_metric'].
  Available metrics: 'support', 'confidence', 'lift', 'conviction', 'zhangs_metric', 'yulesq', 'interestingness'
- `num_workers` (int, optional): Number of parallel workers for quality metric calculation. Default=1 for sequential processing.
  **Note**: Parallelization is automatically disabled for fewer than 1000 rules due to overhead costs. Set to 4-8 for datasets generating 1000+ rules.

**Returns**:

A dictionary containing:

```python
{
    "rules": [
        {
            "antecedents": [...],
            "consequent": {...},
            "support": 0.702,
            "confidence": 0.943,
            "zhangs_metric": 0.69,
            "rule_coverage": 0.744,
            ...  # additional metrics based on quality_metrics parameter
        },
        ...
    ],
    "statistics": {
        "rule_count": 15,
        "average_support": 0.448,
        "average_confidence": 0.881,
        "average_coverage": 0.860,
        "data_coverage": 0.923,
        "average_zhangs_metric": 0.318,
        ...  # additional statistics for calculated metrics
    }
}
```

**Example**:

```python
result = rule_extraction.generate_rules(trained_autoencoder)
print(f"Found {result['statistics']['rule_count']} rules")
for rule in result['rules']:
    print(f"Support: {rule['support']}, Confidence: {rule['confidence']}")
```

### generate_frequent_itemsets

```python
generate_frequent_itemsets(
    autoencoder,
    features_of_interest=None,
    similarity=0.5,
    max_length=2,
    num_workers=1
)
```

Generates frequent itemsets from a trained AutoEncoder using the same Aerial+ mechanism, with support values calculated automatically using optimized batch processing.

**Parameters**:

- `autoencoder` (AutoEncoder): A trained autoencoder instance.
- `features_of_interest` (list, Optional): only look for rules that have these features of interest on the antecedent
  side. Accepted form `["feature1", "feature2", {"feature3": "value1"}, ...]`, either a feature name as str, or specific
  value of a feature in object form
- `similarity` (float, Optional): Minimum similarity threshold for an itemset to be considered frequent. Default=0.5
- `max_length` (int, Optional): Maximum number of items in each itemset. Default=2
- `num_workers` (int, optional): Number of parallel workers for support calculation. Default=1 for sequential processing.
  **Note**: Parallelization is automatically disabled for fewer than 1000 itemsets due to overhead costs. Set to 4-8 for datasets generating 1000+ itemsets.

**Returns**:

A dictionary containing:

```python
{
    "itemsets": [
        {
            "itemset": [{'feature': 'gender', 'value': 'Male'}, {'feature': 'income', 'value': 'High'}],
            "support": 0.524
        },
        {
            "itemset": [{'feature': 'age', 'value': '30-39'}],
            "support": 0.451
        },
        ...
    ],
    "statistics": {
        "itemset_count": 15,
        "average_support": 0.295
    }
}
```

**Example**:

```python
result = rule_extraction.generate_frequent_itemsets(trained_autoencoder)
print(f"Found {result['statistics']['itemset_count']} itemsets")
for item in result['itemsets']:
    print(f"Support: {item['support']}")
```

## Rule Quality Module

**Note**: Quality metrics are now calculated automatically by `generate_rules()` and `generate_frequent_itemsets()`. The functions below are maintained for backward compatibility and advanced use cases.

### Available Quality Metrics

The `rule_quality` module provides the following constants and metrics:

- `AVAILABLE_METRICS`: List of all supported quality metrics
  - 'support', 'confidence', 'lift', 'conviction', 'zhangs_metric', 'yulesq', 'interestingness'
- `DEFAULT_RULE_METRICS`: Default metrics calculated by `generate_rules()`
  - ['support', 'confidence', 'zhangs_metric']

### Internal Helper Functions

The following helper functions are used internally by PyAerial for calculating individual quality metrics:

- `calculate_lift(support, confidence)`: Calculate lift metric
- `calculate_conviction(support, confidence)`: Calculate conviction metric
- `calculate_zhangs_metric(support, support_ant, support_cons)`: Calculate Zhang's metric
- `calculate_yulesq(full_count, not_ant_not_con, con_not_ant, ant_not_con)`: Calculate Yule's Q
- `calculate_interestingness(confidence, support, rhs_support, input_length)`: Calculate interestingness

These are low-level functions used by `generate_rules()` and `generate_frequent_itemsets()`. Most users should use the high-level functions instead.

## Discretization Module

### equal_frequency_discretization

```python
equal_frequency_discretization(df: pd.DataFrame, n_bins = 5)
```

Discretizes all numerical columns into equal-frequency bins and encodes the resulting intervals as string labels.

**Parameters**:

- `df`: A pandas DataFrame containing tabular data.
- `n_bins`: Number of intervals (bins) to create.

**Returns**: A modified DataFrame with numerical columns replaced by string-encoded interval bins.

### equal_width_discretization

```python
equal_width_discretization(df: pd.DataFrame, n_bins = 5)
```

Discretizes all numerical columns into equal-width bins and encodes the resulting intervals as string labels.

**Parameters**:

- `df`: A pandas DataFrame containing tabular data.
- `n_bins`: Number of intervals (bins) to create.

**Returns**: A modified DataFrame with numerical columns replaced by string-encoded interval bins.
