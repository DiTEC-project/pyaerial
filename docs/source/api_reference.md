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
    epochs=1,
    batch_size=2,
    loss_function=torch.nn.BCELoss(),
    num_workers=1,
    layer_dims=None,
    device=None
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
- `noise_factor` (float): Controls the amount of Gaussian noise added to inputs during training (denoising effect).
- `lr` (float): Learning rate for the Adam optimizer.
- `epochs` (int): Number of training epochs.
- `batch_size` (int): Number of samples per training batch.
- `loss_function` (torch.nn.Module): Loss function to apply (default is BCELoss).
- `num_workers` (int): Number of subprocesses used for data loading.
- `layer_dims` (list of int, optional): Custom hidden layer dimensions for autoencoder construction.
- `device` (str): Name of the device to run the Autoencoder model on, e.g., "cuda", "cpu" etc. The device option that is
  set here will also be used in the rule extraction stage.

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
    target_classes=None
)
```

Extracts association rules from a trained AutoEncoder using the Aerial algorithm.

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

**Returns**:

A list of extracted rules in the form:

```python
[
    {"antecedents": [...], "consequent": ...},
    ...
]
```

### generate_frequent_itemsets

```python
generate_frequent_itemsets(
    autoencoder,
    features_of_interest=None,
    similarity=0.5,
    max_length=2
)
```

Generates frequent itemsets from a trained AutoEncoder using the same Aerial+ mechanism.

**Parameters**:

- `autoencoder` (AutoEncoder): A trained autoencoder instance.
- `features_of_interest` (list, Optional): only look for rules that have these features of interest on the antecedent
  side. Accepted form `["feature1", "feature2", {"feature3": "value1"}, ...]`, either a feature name as str, or specific
  value of a feature in object form
- `similarity` (float, Optional): Minimum similarity threshold for an itemset to be considered frequent. Default=0.8
- `max_length` (int, Optional): Maximum number of items in each itemset. Default=2

**Returns**:

A list of frequent itemsets, where each itemset is a list of dictionaries with 'feature' and 'value' keys:

```python
[
    [{'feature': 'gender', 'value': 'Male'}, {'feature': 'income', 'value': 'High'}],
    [{'feature': 'age', 'value': '30-39'}],
    ...
]
```

## Rule Quality Module

### calculate_basic_rule_stats

```python
calculate_basic_rule_stats(rules, transactions, num_workers)
```

Computes support and confidence for a list of rules using parallel processing.

**Parameters**:

- `rules`: List of rule dictionaries with 'antecedents' and 'consequent'.
- `transactions`: A pandas DataFrame of one-hot encoded transactions.
- `num_workers`: Number of parallel workers

**Returns**: A list of rules enriched with support and confidence values.

### calculate_freq_item_support

```python
calculate_freq_item_support(freq_items, transactions, max_workers=1)
```

Calculates the support for a list of frequent itemsets using optimized vectorized operations with parallel processing support.

**Parameters**:

- `freq_items`: List of itemsets (each itemset is a list of dictionaries with 'feature' and 'value' keys).
- `transactions`: A pandas DataFrame of categorical data.
- `max_workers`: Number of parallel workers (via joblib). Default=1. Set higher for faster computation on large datasets.

**Returns**:
- A list of dictionaries, each containing 'itemset' and 'support' keys
- Average support across all itemsets

Example return format:
```python
[
    {
        'itemset': [{'feature': 'age', 'value': '30-39'}],
        'support': 0.524
    },
    {
        'itemset': [{'feature': 'menopause', 'value': 'ge40'}],
        'support': 0.451
    },
    ...
], 0.295
```

### calculate_rule_stats

```python
calculate_rule_stats(rules, transactions, max_workers=1)
```

Evaluates rules with extended metrics including: Support, Confidence, Zhang's Metric, Dataset Coverage.

Runs in parallel with joblib.

**Parameters**:

- `rules`: List of rule dictionaries.
- `transactions`: One-hot encoded pandas DataFrame.
- `max_workers`: Number of parallel threads (via joblib).

**Returns**:

- A dictionary of average metrics (support, confidence, zhangs_metric, coverage)
- A list of updated rules

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
