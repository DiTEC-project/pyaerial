# pyaerial: scalable association rule mining

---
<div align="center">

  <img src="https://img.shields.io/badge/python-3.9%2C3.10%2C3.11%2C3.12-blue" alt="Python Versions">

  <img src="https://img.shields.io/pypi/v/pyaerial.svg" alt="PyPI Version">

  <img src="https://github.com/DiTEC-project/pyaerial/actions/workflows/tests.yml/badge.svg" alt="Build Status">

  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">

  <img src="https://img.shields.io/github/stars/DiTEC-project/pyaerial.svg?style=social&label=Stars" alt="GitHub Stars">

  <img src="https://img.shields.io/github/last-commit/DiTEC-project/pyaerial" alt="Last commit">


  <img src="https://img.shields.io/badge/Ubuntu-24.04%20LTS-orange" alt="Tested on Ubuntu 24.04 LTS">

  <img src="https://img.shields.io/badge/macOS-Monterey%2012.6.7-lightgrey" alt="Tested on MacOS Monterey 12.6.7">
</div>

---
<p align="center">
  <a href="#installation">📥 Install</a> |
  <a href="#quick-start">🚀 Quick Start</a> |
  <a href="#features">✨ Features</a> |
  <a href="https://pyaerial.readthedocs.io">📚 Documentation</a> |
  <a href="#citation">📄 Cite</a> |
  <a href="#contribute">🤝 Contribute</a> |
  <a href="LICENSE">🔑 License</a>
</p>

PyAerial is a **Python implementation** of the Aerial scalable neurosymbolic association rule miner for tabular data. It
utilizes an under-complete denoising Autoencoder to learn a compact representation of tabular data, and extracts a
concise set of high-quality association rules with full data coverage.

Unlike traditional exhaustive methods (e.g., Apriori, FP-Growth), Aerial addresses the **rule explosion** problem by
learning neural representations and extracting only the most relevant patterns, making it suitable for large-scale
datasets. PyAerial supports **GPU acceleration**, **numerical data discretization**, **item constraints**, and
**classification rule extraction** and **rule visualization**
via [NiaARM](https://github.com/firefly-cpp/NiaARM?tab=readme-ov-file#visualization) library (see [Features](#features)
for complete list).

Learn more about the architecture, training, and rule extraction in our paper:
[Neurosymbolic Association Rule Mining from Tabular Data](https://proceedings.mlr.press/v284/karabulut25a.html)

---

## Installation

Install PyAerial using pip:

```bash
pip install pyaerial
```

---

## Quick Start

```python
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

# Load a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

# Train an autoencoder on the loaded table
trained_autoencoder = model.train(breast_cancer)

# Extract association rules from the autoencoder
association_rules = rule_extraction.generate_rules(trained_autoencoder)

# Calculate rule quality statistics
if len(association_rules) > 0:
    stats, association_rules = rule_quality.calculate_rule_stats(
        association_rules,
        trained_autoencoder.input_vectors
    )
    print(f"Extracted {stats['rule_count']} rules\n")

    # Display a sample rule
    sample_rule = association_rules[0]
    print(f"Sample Rule: {sample_rule}")
```

**Output:**

```python
Extracted
15
rules

Sample
Rule: {
    "antecedents": ["inv-nodes__0-2"],
    "consequent": "node-caps__no",
    "support": 0.702,
    "confidence": 0.943,
    "zhangs_metric": 0.69
}
```

**Interpretation:**
This rule indicates that when the `inv-nodes` feature has a value between `0-2`, there is a strong likelihood (94.3%
confidence) that `node-caps` equals `no`. The rule covers 70.2% of the dataset.

**Quality metrics explained:**

- **Support**: Frequency of the rule in the dataset (how often the pattern occurs)
- **Confidence**: How often the consequent is true when antecedent is true (rule reliability)
- **Zhang's Metric**: Correlation measure between antecedent and consequent (-1 to 1; positive values indicate positive
  correlation)

**Overall statistics across all 15 rules:**

```python
{
    "rule_count": 15,
    "average_support": 0.448,
    "average_confidence": 0.881,
    "average_coverage": 0.860,
    "average_zhangs_metric": 0.318
}
```

**Can't get results you looked for?**
See [_Debugging_](https://pyaerial.readthedocs.io/en/latest/advanced_topics.html#debugging) in our
documentation.

---

## Features

PyAerial provides a comprehensive toolkit for association rule mining with advanced capabilities:

- **Scalable Rule Mining** - Efficiently mine association rules from large tabular datasets without rule explosion
- **Frequent Itemset Mining** - Generate frequent itemsets using the same neural approach
- **ARM with Item Constraints** - Focus rule mining on specific features of interest
- **Classification Rules** - Extract rules with target class labels for interpretable inference
- **Numerical Data Support** - Built-in discretization methods (equal-frequency, equal-width)
- **Customizable Architectures** - Fine-tune autoencoder layers and dimensions for optimal performance
- **GPU Acceleration** - Leverage CUDA for faster training on large datasets
- **Quality Metrics** - Comprehensive rule evaluation (support, confidence, coverage, Zhang's metric)
- **Rule Visualization** - Integrate with NiaARM for scatter plots and visual analysis
- **Flexible Training** - Adjust epochs, learning rate, batch size, and noise factors

---

## How Aerial Works?

Aerial employs a three-stage neurosymbolic pipeline to extract high-quality association rules from tabular data:

### 1. Data Preparation

Categorical data is one-hot encoded while tracking feature relationships. Numerical columns require pre-discretization (
equal-frequency or equal-width methods available). The encoded values are transformed into vector format for neural
processing.

### 2. Autoencoder Training

An under-complete denoising autoencoder learns a compact representation of the data:

- **Architecture**: Logarithmic reduction (base 16) automatically configures layers, or use custom dimensions
- **Bottleneck design**: The encoder compresses input to the original feature count, forcing the network to learn
  meaningful associations
- **Denoising mechanism**: Random noise during training improves robustness and generalization

<div align="center">
  <img src="https://raw.githubusercontent.com/DiTEC-project/pyaerial/main/example.png" alt="Rule extraction example" width="600">
  <p><i>Example: Rule extraction process using weather and beverage features</i></p>
</div>

### 3. Rule Extraction

Rules emerge from analyzing the trained autoencoder using test vectors:

1. Test vectors are created with equal probabilities across categories
2. Specific features are set to 1 (antecedents) while others remain at baseline
3. Forward passes through the network produce output probabilities
4. Rules are extracted when probabilities exceed similarity thresholds
5. Quality metrics (support, confidence, coverage, Zhang's metric) are calculated

<div align="center">
  <img src="https://raw.githubusercontent.com/DiTEC-project/pyaerial/main/pipeline.png" alt="Aerial pipeline" width="700">
  <p><i>Complete three-stage pipeline: data preparation → training → rule extraction</i></p>
</div>

**Learn more:** For detailed explanations of the architecture, theoretical foundations, and experimental results, see
our paper:
[Neurosymbolic Association Rule Mining from Tabular Data](https://proceedings.mlr.press/v284/karabulut25a.html)

---

## Documentation

For detailed usage examples, API reference, and advanced topics, visit our comprehensive documentation:

**[📚 Read the full documentation on ReadTheDocs](https://pyaerial.readthedocs.io)**

Documentation includes:

- **Getting Started** - Installation and basic usage
- **User Guide** - Detailed examples for all features
- **API Reference** - Complete function and class documentation
- **Advanced Topics** - GPU usage, debugging, visualization
- **How It Works** - Understanding Aerial's architecture and algorithm

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

For questions, suggestions, or collaborations, please contact:

**Erkan Karabulut**
📧 e.karabulut@uva.nl
📧 erkankkarabulut@gmail.com

---

## Contribute

We welcome contributions from the community! Whether you're fixing bugs, adding new features, improving documentation,
or sharing feedback, your help is appreciated.

**How to contribute:**

- 🐛 **Report bugs** - Open an issue describing the problem
- 💡 **Suggest features** - Share your ideas for improvements
- 📝 **Improve docs** - Help us make the documentation clearer
- 🔧 **Submit PRs** - Fork the repo and create a pull request
- 💬 **Share feedback** - Contact us with your experience using PyAerial

Feel free to open an issue or pull request on [GitHub](https://github.com/DiTEC-project/pyaerial), or reach out
directly!

### Contributors

All contributors to this project are recognized and appreciated! The profiles of contributors will be listed here:

<a href="https://github.com/DiTEC-project/pyaerial/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DiTEC-project/pyaerial" />
</a>

*Made with [contrib.rocks](https://contrib.rocks).*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.