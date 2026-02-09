# Parameter Tuning Guide

> **📝 Note on Parameter Names:** The parameters `min_rule_frequency` and `min_rule_strength` correspond to `ant_similarity` and `con_similarity` in the original [Aerial](https://proceedings.mlr.press/v284/karabulut25a.html) and [PyAerial](https://doi.org/10.1016/j.softx.2025.102341) papers.

> **🖥️ CPU Performance:** PyAerial runs fast on CPU. GPU acceleration is optional and only beneficial for very large datasets.

## Golden Rules

1. Association rule mining is a **knowledge discovery** task.
2. Knowledge discovery is **unsupervised**.
3. There are **NO GROUND TRUTHS** or **best rules** in knowledge discovery, unless an explicit objective is given.

**Therefore**, we need to tell the algorithm what kind of patterns/rules we are looking for by setting the correct
parameters!

## Default Parameter Values

Aerial uses these defaults when you don't specify parameters:

### Core Rule Extraction Parameters ⭐

| Parameter         | Default | What it Controls                                               |
|-------------------|---------|----------------------------------------------------------------|
| `min_rule_frequency`  | `0.5`   | How frequent patterns must be (analogous to minimum support)   |
| `min_rule_strength`   | `0.8`   | How reliable rules must be (confidence + association strength) |
| `max_antecedents` | `2`     | Maximum complexity (number of conditions per rule)             |

### Training Parameters

| Parameter      | Default       | What it Controls                                                                 |
|----------------|---------------|----------------------------------------------------------------------------------|
| `epochs`       | `2`           | Training iterations (shorter training = fewer, higher-quality rules)             |
| `show_progress`| `True`        | Show a progress bar during training                                              |
| `layer_dims`   | Auto-selected | Autoencoder architecture (smaller = stronger compression = higher quality rules) |

> **💡 Tip:** Train less for fewer, higher-quality rules. The default `epochs=2` works well for most datasets. Only increase epochs if you're not getting enough rules for your use case.

### Filtering Parameters

| Parameter        | Default | What it Controls                                   |
|------------------|---------|---------------------------------------------------|
| `min_confidence` | `None`  | Post-filter rules with confidence below this value |
| `min_support`    | `None`  | Post-filter rules with support below this value    |

### Other Parameters

| Parameter              | Default                                      | What it Controls                                     |
|------------------------|----------------------------------------------|------------------------------------------------------|
| `features_of_interest` | `None`                                       | Focus mining on specific features (item constraints) |
| `target_classes`       | `None`                                       | Restrict rules to predict specific class labels      |
| `quality_metrics`      | `['support', 'confidence', 'zhangs_metric']` | Which metrics to calculate                           |
| `num_workers`          | `1`                                          | Parallel processing (set to 4-8 for 1000+ rules)     |
| `lr`                   | `5e-3`                                       | Learning rate                                        |
| `batch_size`           | Auto                                         | Training batch size                                  |
| `device`               | Auto                                         | `'cuda'` for GPU or `'cpu'`                          |

**💡 Tip:** Start with defaults, then adjust the 3 core parameters based on your goals below.

## Quick Parameter Reference

| I want...                                | Set these parameters                          | Example                                                        |
|------------------------------------------|-----------------------------------------------|----------------------------------------------------------------|
| High-quality rules only (post-filtering) | `min_confidence=0.7, min_support=0.1`         | `generate_rules(model, min_confidence=0.7, min_support=0.1)`   |
| High support rules                       | `min_rule_frequency=0.7` (or higher)              | `generate_rules(model, min_rule_frequency=0.7)`                    |
| Low support rules                        | `min_rule_frequency=0.1` (or lower)               | `generate_rules(model, min_rule_frequency=0.1)`                    |
| High confidence rules                    | `min_rule_strength=0.8` (or higher)               | `generate_rules(model, min_rule_strength=0.8)`                     |
| Low confidence rules                     | `min_rule_strength=0.3` (or lower)                | `generate_rules(model, min_rule_strength=0.3)`                     |
| Fewer rules                              | Increase `min_rule_frequency` and `min_rule_strength` | `generate_rules(model, min_rule_frequency=0.6, min_rule_strength=0.8)` |
| More rules                               | Decrease `min_rule_frequency` and `min_rule_strength` | `generate_rules(model, min_rule_frequency=0.2, min_rule_strength=0.5)` |
| Strong associations                      | `min_rule_strength=0.8` (or higher)               | `generate_rules(model, min_rule_strength=0.8)`                     |
| Complex patterns                         | `max_antecedents=3` (or higher)               | `generate_rules(model, max_antecedents=3)`                     |
| More training (more rules)               | `epochs=5` or higher                          | `model.train(data, epochs=5)`                                  |

**Note** that `min_confidence` and `min_support` are post-processing filters.

**💡 Tip: Still not getting the results you want?** See
the [Advanced Tuning](configuration.md#advanced-training-and-architecture-tuning) to learn how
to tune the architecture and training parameters and [Debugging section](configuration.md#debugging) for troubleshooting
tips.

## How to Set Parameters for Specific Goals

### Getting High Support Rules

**Support** measures how frequently a pattern appears in your data. High support rules represent common patterns
and more generalizable rules.

**Parameters to adjust:**

- Increase `min_rule_frequency` to 0.6, 0.7, or higher
- The antecedent similarity threshold is analogous to minimum support in traditional ARM methods

**What to expect:**

- Fewer rules overall
- Rules covering larger portions of your dataset
- More general patterns

**Example:**

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features
trained_autoencoder = model.train(breast_cancer)

# Get high support rules
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.7  # High threshold for common patterns
)
```

### Getting Low Support Rules

**Low support rules** can reveal rare but potentially interesting patterns in your data.

**Parameters to adjust:**

- Decrease `min_rule_frequency` to 0.1, 0.05, or lower
- You may also need to adjust `max_antecedents` if discovering complex rare patterns

**What to expect:**

- More rules overall
- Rules covering smaller portions of your dataset
- Potentially longer execution time

**Performance considerations:**

- Unlike traditional ARM methods, Aerial's neural network performs the same operations regardless of threshold values
- Lower thresholds don't increase search space, but they do result in more candidate rules to filter
- Longer execution time comes from processing and validating more candidate patterns
- Start with moderate values (e.g., 0.3) and gradually decrease

**Example:**

```python
# Get low support (rare pattern) rules
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.1  # Low threshold for rare patterns
)
```

### Controlling the Number of Rules

The number of rules is affected by multiple parameters working together.

**To get fewer rules:**

- Increase both `min_rule_frequency` (e.g., 0.6-0.8) and `min_rule_strength` (e.g., 0.7-0.9)
- Decrease `max_antecedents` to 1 or 2

**To get more rules:**

- Decrease both `min_rule_frequency` (e.g., 0.1-0.3) and `min_rule_strength` (e.g., 0.3-0.6)
- Increase `max_antecedents` to 3 or higher

**Parameter interaction:**

- `min_rule_frequency`: Controls how many antecedent patterns are considered
- `min_rule_strength`: Filters rules based on confidence and association strength
- `max_antecedents`: Limits complexity of patterns

**Example:**

```python
# For a concise set of strong rules
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.6,
    min_rule_strength=0.8,
    max_antecedents=2
)

# For a comprehensive exploration
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.2,
    min_rule_strength=0.5,
    max_antecedents=3
)
```

### Getting High Confidence Rules

**Confidence** measures how often the rule's prediction is correct. High confidence rules are more reliable.

**Parameters to adjust:**

- Increase `min_rule_strength` to 0.8, 0.9, or higher
- The consequent similarity threshold combines confidence and association strength

**What to expect:**

- Fewer rules overall
- More reliable predictions
- Stronger if-then relationships

**Trade-offs:**

- Very high thresholds (>0.9) may result in very few or no rules
- Start with 0.7-0.8 and adjust based on results

**Example:**

```python
# Get high confidence rules
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_strength=0.8  # High threshold for reliable rules
)
```

### Getting Low Confidence Rules

**Low confidence rules** can still be useful for exploratory analysis or finding weak associations.

**Parameters to adjust:**

- Decrease `min_rule_strength` to 0.5, 0.4, or lower

**What to expect:**

- More rules overall
- Weaker if-then relationships
- May include spurious correlations

**When to use:**

- Exploratory data analysis
- When you want to see all possible patterns
- When combined with other filtering criteria

**Example:**

```python
# Get low confidence rules for exploration
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_strength=0.4  # Lower threshold for exploration
)
```

### Getting Rules with High Association Strength

**Association strength** (Zhang's metric) measures the correlation between antecedent and consequent, accounting for the
prevalence of both.

**Parameters to adjust:**

- Increase `min_rule_strength` to 0.7 or higher
- The consequent similarity threshold incorporates association strength

**Difference from confidence:**

- Confidence: P(consequent|antecedent)
- Association strength: Accounts for how common both antecedent and consequent are
- High association strength rules are less likely to be coincidental

**Example:**

```python
# Get rules with strong associations
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_strength=0.7  # Ensures strong correlations
)

# Check Zhang's metric in results
for rule in result['rules']:
    print(f"Zhang's metric: {rule['zhangs_metric']}")
```

### Getting Rules with Low Association Strength

**Low association strength** rules may indicate overfitting or spurious correlations.

**Parameters to adjust:**

- Decrease `min_rule_strength` below 0.5

**Common causes:**

- Over-training the autoencoder
- Too many parameters in the neural network
- Data doesn't have strong patterns

**When you get low association strength unexpectedly:**

- Reduce training epochs
- Use a simpler autoencoder architecture
- Check if your data has meaningful patterns

**Example:**

```python
# If getting low Zhang's metric unexpectedly, try:
trained_autoencoder = model.train(
    breast_cancer,
    epochs=2,  # Reduce from default to prevent overfitting
    layer_dims=[4, 2]  # Simpler architecture
)
```

## Common Scenarios

### Scenario 0: Don't Know What Parameters to Use

Start with defaults and filter results. Training uses smart defaults (epochs=2, auto batch_size).

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features

# Training uses smart defaults (epochs=2 for fewer and higher quality rules)
trained_autoencoder = model.train(breast_cancer)

# Extract rules with default thresholds, filter to keep high-quality ones
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_confidence=0.6
)
```

### Scenario 1: Finding Rare but Strong Patterns

Perfect for discovering uncommon but highly reliable associations.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.05,  # Low support for rare patterns
    min_rule_strength=0.8,  # High confidence for strong rules
    max_antecedents=2  # Moderate complexity
)
```

### Scenario 2: Quick Overview of Main Patterns

Get a concise summary of the most prominent patterns.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.5,  # Higher support for common patterns
    min_rule_strength=0.7,  # Good confidence
    max_antecedents=2  # Limit complexity for interpretability
)
```

### Scenario 3: Comprehensive Exploration

Discover all possible patterns for in-depth analysis.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.1,  # Low support to catch rare patterns
    min_rule_strength=0.5,  # Moderate confidence
    max_antecedents=3  # Allow complex patterns
)
```

### Scenario 4: Classification Rules

Extract rules for predictive modeling with high reliability.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    target_classes=["Class"],  # Specify class label column
    min_rule_strength=0.7,  # High confidence for predictions
    min_rule_frequency=0.3  # Allow diverse patterns
)
```

### Scenario 5: Focused Mining with Item Constraints

Mine rules focusing only on specific features of interest instead of the entire feature space.

```python
# Define which features to focus on
features_of_interest = [
    "age",  # All values of 'age' feature
    {"menopause": "premeno"},  # Only 'premeno' value of 'menopause'
    "tumor-size",  # All values of 'tumor-size'
    {"node-caps": "yes"}  # Only 'yes' value of 'node-caps'
]

result = rule_extraction.generate_rules(
    trained_autoencoder,
    features_of_interest,  # Focus mining on specified features
    min_rule_frequency=0.3,  # Moderate support
    min_rule_strength=0.6  # Moderate confidence
)
```

**Use when:**

- You have domain knowledge about important features
- You want to reduce rule explosion by focusing on key variables
- You're investigating relationships involving specific attributes
- You need faster execution by limiting the search space

**Note:** Features of interest appear only on the antecedent (left) side of rules. To constrain the consequent (right)
side, use `target_classes` (see Scenario 4).

## Understanding Parameter Effects

For a deeper understanding of how each parameter affects rule quality, see this detailed blog
post: [Scalable Knowledge Discovery with PyAerial](https://erkankarabulut.github.io/blog/uva-dsc-seminar-scalable-knowledge-discovery/)

### Parameter Summary

| Parameter         | Analogous to (in traditional ARM)   | Effect when increased                 | Effect when decreased            |
|-------------------|-------------------------------------|---------------------------------------|----------------------------------|
| `min_rule_frequency`  | Minimum support                     | Fewer, higher support rules           | More, lower support rules        |
| `min_rule_strength` | Minimum confidence + Zhang's metric | Fewer, higher confidence rules        | More, lower confidence rules     |
| `max_antecedents` | Maximum itemset size                | More complex patterns, longer runtime | Simpler patterns, faster runtime |

## Next Steps

- See [Advanced: Training and Architecture Tuning](configuration.md#advanced-training-and-architecture-tuning) to learn how the architecture
and training parameters impact rule quality.
- See [Debugging](configuration.md#debugging) if you encounter issues
- Check [User Guide](user_guide.md) for usage examples
- Review [API Reference](api_reference.md) for complete parameter documentation