# Getting Started

## Installation

You can easily install **pyaerial** using pip:

```bash
pip install pyaerial
```

> **Note:** Examples in the documentation use `ucimlrepo` to fetch sample datasets. Install it to run the examples:
> ```bash
> pip install ucimlrepo
> ```

> **Data Requirements:** PyAerial works with **categorical data**. You don't need to one-hot encode your data—PyAerial handles encoding automatically.

## Tested Platforms

- **Ubuntu 24.04 LTS**
- **macOS Monterey 12.6.7**
- Python 3.9, 3.10, 3.11 and 3.12

## Quick Start

Here's a simple example to get you started with PyAerial:

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

# Load a categorical tabular dataset from the UCI ML repository
breast_cancer = fetch_ucirepo(id=14).data.features

# Train an autoencoder on the loaded table
trained_autoencoder = model.train(breast_cancer)

# Extract association rules with quality metrics calculated automatically
result = rule_extraction.generate_rules(trained_autoencoder)

# Access rules and statistics
if len(result['rules']) > 0:
    print(f"Overall statistics: {result['statistics']}\n")
    print(f"Sample rule: {result['rules'][0]}")
```

### Output

Following is the partial output of above code:

```python
>>> Output:
breast_cancer dataset:
     age menopause tumor-size inv-nodes  ... deg-malig  breast breast-quad irradiat
0  30-39   premeno      30-34       0-2  ...         3    left    left_low       no
1  40-49   premeno      20-24       0-2  ...         2   right    right_up       no
2  40-49   premeno      20-24       0-2  ...         2    left    left_low       no
                                         ...

Overall statistics: {
   "rule_count": 15,
   "average_support": 0.448,
   "average_confidence": 0.881,
   "average_coverage": 0.860,
   "data_coverage": 0.923,
   "average_zhangs_metric": 0.318
}

Sample rule:
{
   "antecedents": [
      {"feature": "inv-nodes", "value": "0-2"}
   ],
   "consequent": {"feature": "node-caps", "value": "no"},
   "support": 0.702,
   "confidence": 0.943,
   "zhangs_metric": 0.69,
   "rule_coverage": 0.744
}
```

**Interpretation:** When `inv-nodes` is between `0-2`, there's 94.3% confidence that `node-caps` equals `no`, covering 70.2% of the dataset.

**Quality metrics explained:**

- **Support**: How often this pattern appears in the data (rule frequency)
- **Confidence**: How often the prediction is correct (rule reliability)
- **Zhang's Metric**: Strength of the correlation between antecedent and consequent
- **Rule Coverage**: Proportion of transactions containing the antecedents (left-hand side coverage)

---

**Can't get the results you're looking for?**
See the [Parameter Tuning Guide](parameter_guide.md) to learn how to adjust parameters for your specific needs.

---

## What's Next?

- Explore the [User Guide](user_guide.md) for detailed usage examples
- Learn how to [tune parameters](parameter_guide.md) for different use cases
- Configure [GPU usage and logging](configuration.md)
- Check the [API Reference](api_reference.md) for complete function documentation
- Understand [How Aerial Works](research.md) in depth


If you encounter issues, please create an issue in
our [GitHub repository](https://github.com/DiTEC-project/pyaerial/issues), or directly contact the contributors.
