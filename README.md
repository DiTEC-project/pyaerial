# pyaerial: scalable association rule mining

---
<div align="center">

  <img src="https://img.shields.io/badge/python-3.9%2C3.10%2C3.11%2C3.12-blue" alt="Python Versions">

  <img src="https://img.shields.io/pypi/v/pyaerial.svg" alt="PyPI Version">

  <img src="https://static.pepy.tech/badge/pyaerial" alt="Downloads">

  <img src="https://github.com/DiTEC-project/pyaerial/actions/workflows/tests.yml/badge.svg" alt="Build Status">

  <img src="https://readthedocs.org/projects/pyaerial/badge/?version=latest" alt="Documentation Status">

  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">

  <img src="https://img.shields.io/github/stars/DiTEC-project/pyaerial.svg?style=social&label=Stars" alt="GitHub Stars">
</div>

---
<p align="center">
  <a href="#installation">📥 Install</a> |
  <a href="#quick-start">🚀 Quick Start</a> |
  <a href="#features">✨ Features</a> |
  <a href="https://pyaerial.readthedocs.io">📚 Documentation</a> |
  <a href="https://github.com/DiTEC-project/pyaerial/releases">📋 Releases</a> |
  <a href="#citation">📄 Cite</a> |
  <a href="#contribute">🤝 Contribute</a> |
  <a href="LICENSE">🔑 License</a>
</p>

PyAerial finds human-readable IF-THEN rules in tabular data:

```
Congressional voting records dataset:
IF adoption-of-the-budget-resolution=No AND physician-fee-freeze=Yes THEN Class=republican  (support=0.32, confidence=0.96, zhangs=0.90)
IF adoption-of-the-budget-resolution=y AND physician-fee-freeze=n THEN Class=democrat  (support=0.50, confidence=1.00, zhangs=0.78)

Iris dataset (plants) - numerical:
IF sepal width=(2, 2.9] AND petal width=(1.6, 2.5] THEN class=Iris-virginica  (support=0.12, confidence=1.00, zhangs=0.76)
IF petal length=(1, 2.63] AND petal width=(0.1, 0.87] THEN class=Iris-setosa  (support=0.33, confidence=1.00, zhangs=1.00)

Mushroom dataset:
IF odor=none AND gill-size=broad THEN poisonous=No  (support=0.40, confidence=0.98, zhangs=0.79)
IF gill-spacing=close AND stalk-surface-above-ring=silky THEN edibility=poisonous  (support=0.27, confidence=1.00, zhangs=0.71)

Datasets are from the UCI ML Repository.
```

It is the Python implementation of **Aerial**, a scalable neurosymbolic association rule miner: an under-complete
Autoencoder learns a compact representation of the data, and rules are extracted from the trained model. This avoids
the **rule explosion** and **execution time** problems of exhaustive miners (Apriori, FP-Growth, ECLAT), making rule
mining practical on large datasets such as health records, retail baskets, and sensor data, wherever you want
interpretable patterns next to black-box models.

Learn more about the architecture, training, and rule extraction in our paper:
[Neurosymbolic Association Rule Mining from Tabular Data](https://proceedings.mlr.press/v284/karabulut25a.html)

---

## Why PyAerial?

|                                        | PyAerial                                                             | Exhaustive miners (e.g., Mlxtend, SPMF)        |
|----------------------------------------|----------------------------------------------------------------------|------------------------------------------------|
| Execution time on large data           | 100-1000x faster, also on CPU                                        | Grows steeply with columns and thresholds      |
| Number of rules                        | Concise, high-quality set with full data coverage                    | Rule explosion (easily millions)               |
| Input format                           | pandas DataFrame, one-hot encoding handled internally               | Manual one-hot encoding or custom text formats |
| Rule quality metrics                   | Calculated automatically (support, confidence, Zhang's metric, ...)  | Requires extra steps                           |
| Item constraints, classification rules | Built-in                                                             | Limited or unavailable                         |
| GPU support                            | Optional                                                             | Not available                                  |

For comprehensive benchmarks against Mlxtend, SPMF and other ARM tools, see our software paper:
[PyAerial: Scalable association rule mining from tabular data](https://doi.org/10.1016/j.softx.2025.102341)
(SoftwareX, 2025)

<div align="center">
  <img src="https://raw.githubusercontent.com/DiTEC-project/pyaerial/main/docs/source/_static/assets/benchmark.png" alt="PyAerial performance comparison" width="700">
  <p><i>Execution time comparison across datasets of varying sizes. PyAerial scales linearly while traditional methods (e.g., Mlxtend, SPMF) exhibit exponential growth.</i></p>
</div>

---

## Installation

```bash
pip install pyaerial
```

> **Note:** Examples in the documentation use `ucimlrepo` to fetch sample datasets. Install it to run the examples:
> ```bash
> pip install ucimlrepo
> ```

> **Data Requirements:** PyAerial works with **categorical data**. Numerical columns must be discretized first, using
> the built-in *discretization* module. There is no need to one-hot encode your data; PyAerial handles that
> automatically.

---

## Quick Start

Or try it directly in your browser: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiTEC-project/pyaerial/blob/main/notebooks/quickstart.ipynb)

### Basic Association Rule Mining

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

# Load a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

# Train an autoencoder on the loaded table
trained_autoencoder = model.train(breast_cancer)

# Extract association rules with quality metrics calculated automatically
result = rule_extraction.generate_rules(trained_autoencoder, min_rule_frequency=0.1, min_rule_strength=0.8)

print(f"Overall statistics: {result['statistics']}\n")
print(f"Sample rule: {result['rules'][0]}")
```

**Output:**

```python
Overall statistics: {
    "rule_count": 15,
    "average_support": 0.448,
    "average_confidence": 0.881,
    "average_coverage": 0.860,
    "data_coverage": 0.923,
    "average_zhangs_metric": 0.318
}

Sample rule: {
    "antecedents": [{"feature": "inv-nodes", "value": "0-2"}],
    "consequent": {"feature": "node-caps", "value": "no"},
    "support": 0.702,
    "confidence": 0.943,
    "zhangs_metric": 0.69,
    "rule_coverage": 0.744
}
```

**Interpretation:** When `inv-nodes` is between `0-2`, there's 94.3% confidence that `node-caps` equals `no`, covering
70.2% of the dataset.

**Quality metrics explained:**

- **Support**: Frequency of the rule in the dataset (how often the pattern occurs)
- **Confidence**: How often the consequent is true when antecedent is true (rule reliability)
- **Zhang's Metric**: Correlation measure between antecedent and consequent (-1 to 1; positive values indicate positive
  correlation)
- **Rule Coverage**: Proportion of transactions containing the antecedents
- **Data Coverage** (in statistics): Overall proportion of the dataset covered by at least one rule

Rules are plain dictionaries, so working with them is straightforward:

```python
for rule in result['rules']:
    antecedents = " AND ".join(f"{a['feature']}={a['value']}" for a in rule['antecedents'])
    consequent = f"{rule['consequent']['feature']}={rule['consequent']['value']}"
    print(f"IF {antecedents} THEN {consequent} (support: {rule['support']:.2f}, conf: {rule['confidence']:.2f})")
```

### Working with Numerical Data

For datasets with numerical columns, use PyAerial's built-in discretization methods:

```python
from aerial import model, rule_extraction, discretization
from ucimlrepo import fetch_ucirepo

# Load a numerical dataset (e.g., Iris)
iris = fetch_ucirepo(id=53).data.features

# Discretize numerical columns into categorical bins
# Before: sepal_length = 5.1, 4.9, 7.0, ...  After: sepal_length = (4.8, 5.5], (4.8, 5.5], (6.4, 7.9], ...
iris_discretized = discretization.equal_frequency_discretization(iris, n_bins=3)

# Train and extract rules as usual
trained_autoencoder = model.train(iris_discretized, epochs=10)
result = rule_extraction.generate_rules(trained_autoencoder, min_rule_frequency=0.1)
```

Eight discretization methods are available: unsupervised (equal-frequency, equal-width, k-means, quantile, custom
bins) and supervised (entropy-based, ChiMerge, decision tree), each documented with academic references in
the [User Guide](https://pyaerial.readthedocs.io/en/latest/user_guide.html#running-aerial-for-numerical-values).

### More Recipes

| Goal                                             | How                                                                              | Details                                                                                                          |
|--------------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Focus mining on features of interest             | `generate_rules(model, features_of_interest=["age", {"menopause": "premeno"}])`  | [Item constraints](https://pyaerial.readthedocs.io/en/latest/user_guide.html#specifying-item-constraints)          |
| Classification rules (class label as consequent) | `generate_rules(model, target_classes=["Class"])`                                | [Classification rules](https://pyaerial.readthedocs.io/en/latest/user_guide.html#learning-classification-rules)    |
| Keep only high-quality rules                     | `generate_rules(model, filter_min_confidence=0.7, filter_min_support=0.1)`       | [Parameter guide](https://pyaerial.readthedocs.io/en/latest/parameter_guide.html)                                  |
| Frequent itemsets instead of rules               | `generate_frequent_itemsets(model)`                                              | [API reference](https://pyaerial.readthedocs.io/en/latest/api_reference.html#generate_frequent_itemsets)           |
| Antecedents of unlimited length                  | `generate_rules(model, max_antecedents=None)`                                    | [Parameter guide](https://pyaerial.readthedocs.io/en/latest/parameter_guide.html)                                  |
| Visualize rules                                  | Via [NiaARM](https://github.com/firefly-cpp/NiaARM#visualization)                | [User guide](https://pyaerial.readthedocs.io/en/latest/user_guide.html)                                            |

**Can't get the results you're looking for?**

- [Parameter Tuning Guide](https://pyaerial.readthedocs.io/en/latest/parameter_guide.html): quick reference for high/low support, confidence, and rule count
- [Troubleshooting](https://pyaerial.readthedocs.io/en/latest/parameter_guide.html#when-things-dont-work): what to do when Aerial doesn't find rules or takes too long
- [Training and Rule Quality](https://pyaerial.readthedocs.io/en/latest/parameter_guide.html#training-parameters-and-rule-quality): training duration and architecture optimization

> **Note on Parameter Names:** The parameters `min_rule_frequency` and `min_rule_strength` correspond to `ant_similarity` and `cons_similarity` in the original [Aerial](https://proceedings.mlr.press/v284/karabulut25a.html) and [PyAerial](https://doi.org/10.1016/j.softx.2025.102341) papers.

---

## Features

**Rule mining**

- Scalable association rule mining without rule explosion, with full data coverage
- Frequent itemset mining with the same neural approach
- Item constraints (features of interest) and classification rules (target classes)
- Quality metrics calculated automatically: support, confidence, Zhang's metric, lift, conviction, Yule's Q, interestingness, leverage

**Data handling**

- Direct pandas DataFrame input; one-hot encoding handled internally
- Eight discretization methods for numerical columns (unsupervised and supervised)

**Performance**

- Fast on CPU; optional GPU acceleration for very large datasets
- Post-filters (`filter_min_confidence`, `filter_min_support`) for high-quality rule sets

**Customization and integration**

- Customizable Autoencoder architecture and training (epochs, learning rate, batch size, masking window)
- Rule visualization via [NiaARM](https://github.com/firefly-cpp/NiaARM#visualization); interpretable inference via [imodels](https://github.com/csinva/imodels)

---

## How Aerial Works

Aerial employs a three-stage neurosymbolic pipeline:

**1. Data Preparation.** Categorical data is one-hot encoded while tracking feature relationships; numerical columns
are pre-discretized. The encoded values form the input vectors of the Autoencoder.

**2. Autoencoder Training.** An under-complete Autoencoder is trained with a masking mechanism: each batch randomly
corrupts a subset of features to a uniform "unknown" distribution, and the network learns to reconstruct them from the
remaining unmasked features. Masking mirrors the antecedent to consequent query pattern used during rule extraction, an
improvement over the Gaussian-noise-based denoising of the original
[Aerial+ paper](https://proceedings.mlr.press/v284/karabulut25a.html).

**3. Rule Extraction.** Rules are extracted by querying the trained Autoencoder with test vectors:

1. Test vectors are created with equal probabilities across categories
2. Specific feature values are marked as antecedents while others remain at baseline
3. Forward runs through the network produce implication probabilities
4. Rules are extracted when implication probabilities exceed the rule frequency and strength thresholds
5. Antecedent combinations are searched with an FP-Growth-style growth strategy: only combinations whose estimated
   frequency passes the threshold are extended further, so forward runs scale with the number of frequent
   combinations rather than all possible combinations (`max_antecedents=None` mines antecedents of unlimited length)
6. Quality metrics (support, confidence, coverage, Zhang's metric, etc.) are calculated automatically using vectorized
   operations

Aerial+ replaces the counting operation of classical rule mining with the Autoencoder's implication probabilities, so
in principle the search strategy of any rule miner can run on top of it. PyAerial adopts FP-Growth's, as it is among
the fastest rule miners.

<div align="center">
  <img src="https://raw.githubusercontent.com/DiTEC-project/pyaerial/main/docs/source/_static/assets/example.png" alt="Rule extraction example" width="600">
  <p><i>Example: Rule extraction process using weather and beverage features</i></p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/DiTEC-project/pyaerial/main/docs/source/_static/assets/pipeline.png" alt="Aerial pipeline" width="700">
  <p><i>Complete three-stage pipeline: data preparation → training → rule extraction</i></p>
</div>

For the architecture, theoretical foundations, and experimental results, see
[How Aerial Works](https://pyaerial.readthedocs.io/en/latest/research.html) in the documentation and the
[paper](https://proceedings.mlr.press/v284/karabulut25a.html).

---

## Documentation

**[Read the full documentation on ReadTheDocs](https://pyaerial.readthedocs.io)** |
**[Release notes on GitHub Releases](https://github.com/DiTEC-project/pyaerial/releases)**

- [Getting Started](https://pyaerial.readthedocs.io/en/latest/getting_started.html): installation and basic usage
- [User Guide](https://pyaerial.readthedocs.io/en/latest/user_guide.html): detailed examples covering all features
- [Parameter Tuning Guide](https://pyaerial.readthedocs.io/en/latest/parameter_guide.html): defaults, tuning for specific goals, troubleshooting
- [Configuration](https://pyaerial.readthedocs.io/en/latest/configuration.html): GPU usage, logging, custom Autoencoders
- [API Reference](https://pyaerial.readthedocs.io/en/latest/api_reference.html): complete function and class documentation
- [How Aerial Works](https://pyaerial.readthedocs.io/en/latest/research.html): the neurosymbolic architecture and algorithm

---

## Citation

If you use PyAerial in your work, please cite our research and software papers:

```bibtex
@InProceedings{pmlr-v284-karabulut25a,
  title         = {Neurosymbolic Association Rule Mining from Tabular Data},
  author        = {Karabulut, Erkan and Groth, Paul and Degeler, Victoria},
  booktitle     = {Proceedings of The 19th International Conference on Neurosymbolic Learning and Reasoning},
  pages         = {565--588},
  year          = {2025},
  editor        = {H. Gilpin, Leilani and Giunchiglia, Eleonora and Hitzler, Pascal and van Krieken, Emile},
  volume        = {284},
  series        = {Proceedings of Machine Learning Research},
  month         = {08--10 Sep},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v284/karabulut25a.html}
}

@article{pyaerial,
  title         = {PyAerial: Scalable association rule mining from tabular data},
  journal       = {SoftwareX},
  volume        = {31},
  pages         = {102341},
  year          = {2025},
  issn          = {2352-7110},
  doi           = {https://doi.org/10.1016/j.softx.2025.102341},
  author        = {Erkan Karabulut and Paul Groth and Victoria Degeler},
}
```

---

## Contact

For questions, suggestions, or collaborations: **Erkan Karabulut**, e.karabulut@uva.nl or erkankkarabulut@gmail.com

---

## Contribute

Contributions are welcome: report bugs or suggest features by opening an issue, improve the documentation, or submit a
pull request on [GitHub](https://github.com/DiTEC-project/pyaerial).

### Contributors

<a href="https://github.com/DiTEC-project/pyaerial/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DiTEC-project/pyaerial" />
</a>

*Made with [contrib.rocks](https://contrib.rocks).*

---

## License

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file for details.
