# Parameter Tuning Guide

## Golden Rules

1. Association rule mining is a **knowledge discovery** task.
2. Knowledge discovery is **unsupervised**.
3. There are **NO GROUND TRUTHS** or **best rules** in knowledge discovery, unless an explicit objective is given.

**Therefore**, we need to tell the algorithm what kind of patterns/rules we are looking for by setting the correct parameters!

## Default Parameter Values

Aerial uses these defaults when you don't specify parameters:

### Core Rule Extraction Parameters ⭐

| Parameter | Default | What it Controls |
|-----------|---------|------------------|
| `ant_similarity` | `0.5` | How frequent patterns must be (like minimum support) |
| `cons_similarity` | `0.8` | How reliable rules must be (confidence + association strength) |
| `max_antecedents` | `2` | Maximum complexity (number of conditions per rule) |

### Training Parameters

| Parameter | Default | What it Controls |
|-----------|---------|------------------|
| `epochs` | `2` | Training iterations (⚠️ more ≠ better, often causes overfitting) |
| `layer_dims` | Auto-selected | Autoencoder architecture (smaller = stronger compression = higher quality rules) |

### Other Parameters

| Parameter | Default | What it Controls |
|-----------|---------|------------------|
| `features_of_interest` | `None` | Focus mining on specific features (item constraints) |
| `target_classes` | `None` | Restrict rules to predict specific class labels |
| `quality_metrics` | `['support', 'confidence', 'zhangs_metric']` | Which metrics to calculate |
| `num_workers` | `1` | Parallel processing (set to 4-8 for 1000+ rules) |
| `lr` | `5e-3` | Learning rate |
| `batch_size` | `2` | Training batch size |
| `device` | Auto | `'cuda'` for GPU or `'cpu'` |

**💡 Tip:** Start with defaults, then adjust the 3 core parameters based on your goals below.

## Quick Parameter Reference

| I want... | Set these parameters | Leave at default | Example |
|-----------|---------------------|------------------|---------|
| High support rules | `ant_similarity=0.7` (or higher) | `cons_similarity`, `max_antecedents` | `generate_rules(model, ant_similarity=0.7)` |
| Low support rules | `ant_similarity=0.1` (or lower) | `cons_similarity`, `max_antecedents` | `generate_rules(model, ant_similarity=0.1)` |
| High confidence rules | `cons_similarity=0.8` (or higher) | `ant_similarity`, `max_antecedents` | `generate_rules(model, cons_similarity=0.8)` |
| Low confidence rules | `cons_similarity=0.3` (or lower) | `ant_similarity`, `max_antecedents` | `generate_rules(model, cons_similarity=0.3)` |
| Fewer rules | Increase `ant_similarity` and `cons_similarity` | `max_antecedents` | `generate_rules(model, ant_similarity=0.6, cons_similarity=0.8)` |
| More rules | Decrease `ant_similarity` and `cons_similarity` | `max_antecedents` | `generate_rules(model, ant_similarity=0.2, cons_similarity=0.5)` |
| Strong associations | `cons_similarity=0.8` (or higher) | `ant_similarity`, `max_antecedents` | `generate_rules(model, cons_similarity=0.8)` |
| Complex patterns | `max_antecedents=3` (or higher) | `ant_similarity`, `cons_similarity` | `generate_rules(model, max_antecedents=3)` |

**Still not getting the results you want?** See the [Debugging section](configuration.md#debugging) for troubleshooting tips.

## How to Set Parameters for Specific Goals

### Getting High Support Rules

**Support** measures how frequently a pattern appears in your data. High support rules represent common patterns.

**Parameters to adjust:**
- Increase `ant_similarity` to 0.6, 0.7, or higher
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
    ant_similarity=0.7  # High threshold for common patterns
)
```

### Getting Low Support Rules

**Low support rules** can reveal rare but potentially interesting patterns in your data.

**Parameters to adjust:**
- Decrease `ant_similarity` to 0.1, 0.05, or lower
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
    ant_similarity=0.1  # Low threshold for rare patterns
)
```

### Controlling the Number of Rules

The number of rules is affected by multiple parameters working together.

**To get fewer rules:**
- Increase both `ant_similarity` (e.g., 0.6-0.8) and `cons_similarity` (e.g., 0.7-0.9)
- Decrease `max_antecedents` to 1 or 2

**To get more rules:**
- Decrease both `ant_similarity` (e.g., 0.1-0.3) and `cons_similarity` (e.g., 0.3-0.6)
- Increase `max_antecedents` to 3 or higher

**Parameter interaction:**
- `ant_similarity`: Controls how many antecedent patterns are considered
- `cons_similarity`: Filters rules based on confidence and association strength
- `max_antecedents`: Limits complexity of patterns

**Example:**
```python
# For a concise set of strong rules
result = rule_extraction.generate_rules(
    trained_autoencoder,
    ant_similarity=0.6,
    cons_similarity=0.8,
    max_antecedents=2
)

# For a comprehensive exploration
result = rule_extraction.generate_rules(
    trained_autoencoder,
    ant_similarity=0.2,
    cons_similarity=0.5,
    max_antecedents=3
)
```

### Getting High Confidence Rules

**Confidence** measures how often the rule's prediction is correct. High confidence rules are more reliable.

**Parameters to adjust:**
- Increase `cons_similarity` to 0.8, 0.9, or higher
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
    cons_similarity=0.8  # High threshold for reliable rules
)
```

### Getting Low Confidence Rules

**Low confidence rules** can still be useful for exploratory analysis or finding weak associations.

**Parameters to adjust:**
- Decrease `cons_similarity` to 0.5, 0.4, or lower

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
    cons_similarity=0.4  # Lower threshold for exploration
)
```

### Getting Rules with High Association Strength

**Association strength** (Zhang's metric) measures the correlation between antecedent and consequent, accounting for the prevalence of both.

**Parameters to adjust:**
- Increase `cons_similarity` to 0.7 or higher
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
    cons_similarity=0.7  # Ensures strong correlations
)

# Check Zhang's metric in results
for rule in result['rules']:
    print(f"Zhang's metric: {rule['zhangs_metric']}")
```

### Getting Rules with Low Association Strength

**Low association strength** rules may indicate overfitting or spurious correlations.

**Parameters to adjust:**
- Decrease `cons_similarity` below 0.5

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

### Scenario 1: Finding Rare but Strong Patterns
Perfect for discovering uncommon but highly reliable associations.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    ant_similarity=0.05,      # Low support for rare patterns
    cons_similarity=0.8,      # High confidence for strong rules
    max_antecedents=2         # Moderate complexity
)
```

### Scenario 2: Quick Overview of Main Patterns
Get a concise summary of the most prominent patterns.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    ant_similarity=0.5,       # Higher support for common patterns
    cons_similarity=0.7,      # Good confidence
    max_antecedents=2         # Limit complexity for interpretability
)
```

### Scenario 3: Comprehensive Exploration
Discover all possible patterns for in-depth analysis.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    ant_similarity=0.1,       # Low support to catch rare patterns
    cons_similarity=0.5,      # Moderate confidence
    max_antecedents=3         # Allow complex patterns
)
```

### Scenario 4: Classification Rules
Extract rules for predictive modeling with high reliability.

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    target_classes=["Class"],  # Specify class label column
    cons_similarity=0.7,       # High confidence for predictions
    ant_similarity=0.3         # Allow diverse patterns
)
```

### Scenario 5: Focused Mining with Item Constraints
Mine rules focusing only on specific features of interest instead of the entire feature space.

```python
# Define which features to focus on
features_of_interest = [
    "age",                      # All values of 'age' feature
    {"menopause": "premeno"},   # Only 'premeno' value of 'menopause'
    "tumor-size",               # All values of 'tumor-size'
    {"node-caps": "yes"}        # Only 'yes' value of 'node-caps'
]

result = rule_extraction.generate_rules(
    trained_autoencoder,
    features_of_interest,       # Focus mining on specified features
    ant_similarity=0.3,         # Moderate support
    cons_similarity=0.6         # Moderate confidence
)
```

**Use when:**
- You have domain knowledge about important features
- You want to reduce rule explosion by focusing on key variables
- You're investigating relationships involving specific attributes
- You need faster execution by limiting the search space

**Note:** Features of interest appear only on the antecedent (left) side of rules. To constrain the consequent (right) side, use `target_classes` (see Scenario 4).

## Understanding Parameter Effects

For a deeper understanding of how each parameter affects rule quality, see this detailed blog post: [Scalable Knowledge Discovery with PyAerial](https://erkankarabulut.github.io/blog/uva-dsc-seminar-scalable-knowledge-discovery/)

### Parameter Summary

| Parameter | Analogous to (in traditional ARM) | Effect when increased | Effect when decreased |
|-----------|----------------------------------|----------------------|----------------------|
| `ant_similarity` | Minimum support | Fewer, higher support rules | More, lower support rules |
| `cons_similarity` | Minimum confidence + Zhang's metric | Fewer, higher confidence rules | More, lower confidence rules |
| `max_antecedents` | Maximum itemset size | More complex patterns, longer runtime | Simpler patterns, faster runtime |

## Next Steps

- See [Debugging](configuration.md#debugging) if you encounter issues
- Check [User Guide](user_guide.md) for usage examples
- Review [API Reference](api_reference.md) for complete parameter documentation