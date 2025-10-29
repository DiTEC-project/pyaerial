# Getting Started

## Installation

You can easily install **pyaerial** using pip:

```bash
pip install pyaerial
```

## Tested Platforms

- **Ubuntu 24.04 LTS**
- **macOS Monterey 12.6.7**
- Python 3.9, 3.10, 3.11 and 3.12

## Quick Start

Here's a simple example to get you started with PyAerial:

```python
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

# Load a categorical tabular dataset from the UCI ML repository
breast_cancer = fetch_ucirepo(id=14).data.features

# Train an autoencoder on the loaded table
trained_autoencoder = model.train(breast_cancer)

# Extract association rules from the autoencoder
association_rules = rule_extraction.generate_rules(trained_autoencoder)

# Calculate rule quality statistics (support, confidence, zhangs metric) for each rule
if len(association_rules) > 0:
    stats, association_rules = rule_quality.calculate_rule_stats(
        association_rules,
        trained_autoencoder.input_vectors
    )
    print(stats, association_rules[:1])
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

Overall rule quality statistics: {
   "rule_count":15,
   "average_support":  0.448,
   "average_confidence": 0.881,
   "average_coverage": 0.860,
   "average_zhangs_metric": 0.318
}

Sample rule:
{
   "antecedents":[
      "inv-nodes__0-2" # meaning column "inv-nodes" has the value between "0-2"
   ],
   "consequent":"node-caps__no", # meaing column "node-caps" has the value "no"
   "support": 0.702,
   "confidence": 0.943,
   "zhangs_metric": 0.69
}
```

## What's Next?

- Explore the [User Guide](user_guide.md) for detailed usage examples
- Check the [API Reference](api_reference.md) for complete function documentation
- Learn about [Advanced Topics](advanced_topics.md) like GPU usage and debugging
- Understand [How Aerial Works](research.md) in depth

If you encounter issues, please see the [Debugging section](advanced_topics.md#debugging) or create an issue in our [GitHub repository](https://github.com/DiTEC-project/pyaerial/issues).
