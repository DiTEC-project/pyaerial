# pyaerial: scalable association rule mining

------------------------------

This is a Python implementation of the Aerial scalable neurosymbolic association rule miner for tabular data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
    - [Association rule mining from categorical tabular data](#1-association-rule-mining-from-categorical-tabular-data)
    - [Setting Aerial parameters](#2-setting-aerial-parameters)
    - [Fine-tuning Autoencoder architecture and dimensions](#3-fine-tuning-autoencoder-architecture-and-dimensions)
    - [Running Aerial for numerical values](#4-running-aerial-for-numerical-values)
    - [Frequent itemset mining with Aerial](#5-frequent-itemset-mining-with-aerial)
    - [Using Aerial for rule-based classification for interpretable inference](#6-using-aerial-for-rule-based-classification-for-interpretable-inference)
    - [Fine-tuning the training parameters](#7-fine-tuning-the-training-parameters)
    - [Setting the log levels](#8-setting-the-log-levels)
- [Functions Overview](#functions-overview)
- [Citation](#citation)
- [Contact](#contact)
- [Contributing](#contributing)

---

## Introduction

Aerial is a scalable Neurosymbolic association rule mining (ARM) method for tabular data. It aims to address the rule
explosion and execution time problems in ARM and it is fully compatible with the existing solutions. Aerial
first creates a neural representation of a given tabular data using an Autoencoder, and then extracts association rules
from the neural representation.

See our paper for the details of Autoencoder architecture, training and rule extraction
algorithm [Neurosymbolic Association Rule Mining from Tabular Data](https://arxiv.org/abs/2504.19354).
If you use Aerial in your work, please [cite](#citation) our paper.

---

## Installation

You can easily install **pyaerial** using pip:

```bash
pip install pyaerial
```

## Usage

This section exemplifies the usage of Aerial with and without hyperparameter tuning.

### 1. Association rule mining from categorical tabular data

```
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

# load a categorical tabular dataset from the UCI ML repository
breast_cancer = fetch_ucirepo(id=14).data.features

# train an autoencoder on the loaded table
trained_autoencoder = model.train(breast_cancer)

# extract association rules from the autoencoder
association_rules = rule_extraction.generate_rules(trained_autoencoder)

# calculate rule quality statistics (support, confidence, zhangs metric) for each rule
if len(association_rules) > 0:
    stats, association_rules = rule_quality.calculate_rule_stats(association_rules, trained_autoencoder.input_vectors)
    print(stats, association_rules[:1])
```

Following is the partial output of above code:

```
>>> Output:
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

### 2. Setting Aerial parameters

Aerial has 3 key parameters; antecedent and consequent similarity threshold, and antecedent length.

As shown in the paper, higher antecedent thresholds results in lower number of higher support rules, while
higher consequent thresholds results in lower number of higher confidence rules.

These 3 parameters can be set using the `generate_rules` function:

```
import pandas as pd
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features

trained_autoencoder = model.train(table_with_labels)

# hyperparameters of aerial can be set using the generate_rules function
association_rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.5, cons_similarity=0.8, max_antecedents=2)
...
```

### 3. Fine-tuning Autoencoder architecture and dimensions

Aerial uses an under-complete Autoencoder and in default, it decides automatically how many layers to use and the
dimensions of each layer (see [Functions and Classes](#functions-and-classes), Autoencoder).

Alternatively, you can specify the number of layers and dimensions in the `train` method to improve performance.

```
from aerial import model, rule_extraction, rule_quality

...
# layer_dims=[2] specifies that there is gonna be 1 hidden layer with a dimension of 2
trained_autoencoder = model.train(breast_cancer, layer_dims=[2]) 
...
```

### 4. Running Aerial for numerical values

Discretizing numerical values is required before running Aerial. We provide 2 discretization methods as part of
the [`discretization.py`](aerial/discretization.py) script; equal-frequency and equal-width discretization.

```
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# load a numerical tabular data
iris = fetch_ucirepo(id=53).data.features

# find and discretize numerical columns 
iris_discretized = discretization.equal_frequency_discretization(iris, n_bins=10)

trained_autoencoder = model.train(iris_discretized, layer_dims=[19], epochs=5)

association_rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.1, cons_similarity=0.5)
```

Following is the partial iris dataset content before and after the discretization:

```
>>> Output:
# before discretization
   sepal length  sepal width  petal length  petal width
0           5.1          3.5           1.4          0.2
1           4.9          3.0           1.4          0.2
...

# after discretization
  sepal length  sepal width  petal length   petal width
0  (5.0, 5.27]  (3.4, 3.61]  (0.999, 1.4]  (0.099, 0.2]
1   (4.8, 5.0]   (2.8, 3.0]  (0.999, 1.4]  (0.099, 0.2]
...
```

### 5. Frequent itemset mining with Aerial

Aerial can also be used for frequent itemset mining besides association rules.

```
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

# categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features
trained_autoencoder = model.train(breast_cancer, epochs=5, lr=1e-3)

# extract frequent itemsets
frequent_itemsets = rule_extraction.generate_frequent_itemsets(trained_autoencoder)

# calculate support values of the frequent itemsets
support_values, average_support = rule_quality.calculate_freq_item_support(frequent_itemsets, breast_cancer)
```

Note that we pass the original dataset (`breast_cancer`) to the `calculate_freq_item_support()` in this case. The
following is a sample output:

```
>>> Output:

Frequent itemsets: 
{('menopause__premeno',): 0.524, ('menopause__ge40',): 0.451, ... }

Average support: 0.295
```

### 6. Using Aerial for rule-based classification for interpretable inference

Aerial can be used to learn rules with a class label on the consequent side, which can later be used for inference
either by themselves or as part of rule list or rule set classifiers (e.g.,
from [imodels](https://github.com/csinva/imodels) repository).

This is done by setting `target_class` parameter of the `generate_rules` function. This parameter refers to the class
label column of the tabular data.

```
import pandas as pd
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

# categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14)
labels = breast_cancer.data.targets
breast_cancer = breast_cancer.data.features

# merge labels column with the actual table 
table_with_labels = pd.concat([breast_cancer, labels], axis=1)

trained_autoencoder = model.train(table_with_labels)

# generate rules with a target class, this learns rules that has the "target_class" column (in this case this column is called "Class") on the consequent side
association_rules = rule_extraction.generate_rules(trained_autoencoder, target_class="Class", cons_similarity=0.5)

if len(association_rules) > 0:
    stats, association_rules = rule_quality.calculate_rule_stats(association_rules, trained_autoencoder.input_vectors)
```

Sample output showing rules with class labels on the right hand side:

```
>>> Output:

{
   "antecedents":[
      "menopause__premeno"
   ],
   "consequent":"Class__no-recurrence-events", # consequent has the class label (column) named "Class" with the value "no-recurrence-events"
   "support":np.float64(0.35664335664335667),
   "confidence":np.float64(0.68),
   "zhangs_metric":np.float64(-0.06585858585858577)
}
```

### 7. Fine-tuning the training parameters

The [`train()`](aerial/model.py) function allows programmers to specify various training parameters:

- autoencoder: You can implement your own Autoencoder and use it for ARM as part of Aerial
- noise_factor `default=0.5`: amount of random noise (`+-`) added to each neuron of the denoising Autoencoder
  before the training process
- lr `default=5e-3`: learning rate
- epochs `default=1`: number of training epochs
- batch_size `default=2`: number of batches to train
- loss_function `default=torch.nn.BCELoss()`: loss function
- num_workers `default=1`: number of workers for parallel execution

```
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

# increasing epochs to 5, note that longer training may lead to overfitting which results in rules with low association strength (zhangs' metric)
trained_autoencoder = model.train(breast_cancer, epochs=5, lr=1e-3)

association_rules = rule_extraction.generate_rules(trained_autoencoder)
if len(association_rules) > 0:
    stats, association_rules = rule_quality.calculate_rule_stats(association_rules, trained_autoencoder.input_vectors)
```

### 8. Setting the log levels

Aerial source code prints extra debug statements notifying the beginning and ending of major
functions such as the training process or rule extraction. The log levels can be changed as follows:

```
import logging
import aerial

# setting the log levels to DEBUG level
aerial.setup_logging(logging.DEBUG)
...
```

## Functions overview

This section lists the important classes and functions as part of the Aerial package.

### AutoEncoder(input_dimension, feature_count, layer_dims=None)

Part of the [`model.py`](aerial/model.py) script. Constructs an autoencoder designed for association rule mining on
tabular data, based on the Neurosymbolic Association
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

### train function

    train(
        transactions,
        autoencoder=None,
        noise_factor=0.5,
        lr=5e-3,
        epochs=1,
        batch_size=2,
        loss_function=torch.nn.BCELoss(),
        num_workers=1,
        layer_dims=None
    )

Part of the [`model.py`](aerial/model.py) script. Trains the AutoEncoder model using one-hot encoded tabular transaction
data.

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

- `layer_dims` (list of int, optional): Custom hidden layer dimensions for autoencoder construction (if applicable).

**Returns**: A trained instance of the AutoEncoder.

### generate_rules

    generate_rules(
        autoencoder,
        ant_similarity=0.5,
        cons_similarity=0.8,
        max_antecedents=2,
        target_class=None
    )

Part of the [`rule_extraction.py`](aerial/rule_extraction.py) script. Extracts association rules from a trained
AutoEncoder using the Aerial algorithm.

**Parameters**:

- `autoencoder` (AutoEncoder): A trained autoencoder instance.

- `ant_similarity` (float): Minimum similarity threshold for an antecedent to be considered frequent.

- `cons_similarity` (float): Minimum probability threshold for a feature to qualify as a rule consequent.

- `max_antecedents` (int): Maximum number of features allowed in the rule antecedent.

- `target_class` (str, optional): When set, restricts rule consequents to the specified class (constraint-based rule
  mining).

**Returns**:

    A list of extracted rules in the form:

    [
        {"antecedents": [...], "consequent": ...},
        ...
    ]

### generate_frequent_itemsets

    generate_frequent_itemsets(
        autoencoder,
        similarity=0.5,
        max_length=2
    )

Part of the [`rule_extraction.py`](aerial/rule_extraction.py) script. Generates frequent itemsets from a trained
AutoEncoder using the same Aerial+ mechanism.

**Parameters**:

- `autoencoder` (AutoEncoder): A trained autoencoder instance.

- `similarity` (float): Minimum similarity threshold for an itemset to be considered frequent.

- `max_length` (int): Maximum number of items in each itemset.

**Returns**:

    A list of frequent itemsets, where each itemset is a list of string features:

    [
        [...],  # e.g., ['gender=Male', 'income=High']
        ...
    ]

### equal_frequency_discretization

    equal_frequency_discretization(df: pd.DataFrame, n_bins=10)

Discretizes all numerical columns into equal-frequency bins and encodes the resulting intervals as string labels.

**Parameters**:

- `df`: A pandas DataFrame containing tabular data.

- `n_bins`: Number of intervals (bins) to create.

**Returns**: A modified DataFrame with numerical columns replaced by string-encoded interval bins.

### equal_width_discretization

`equal_width_discretization(df: pd.DataFrame, n_bins=10)`

Discretizes all numerical columns into equal-width bins and encodes the resulting intervals as string labels.

**Parameters**:

- `df`: A pandas DataFrame containing tabular data.

- `n_bins`: Number of intervals (bins) to create.

**Returns**: A modified DataFrame with numerical columns replaced by string-encoded interval bins.

### calculate_basic_rule_stats

`calculate_basic_rule_stats(rules, transactions)`

Computes support and confidence for a list of rules using parallel processing.

**Parameters**:

- `rules`: List of rule dictionaries with 'antecedents' and 'consequent'.

- `transactions`: A pandas DataFrame of one-hot encoded transactions.

**Returns**: A list of rules enriched with support and confidence values.

### calculate_freq_item_support

`calculate_freq_item_support(freq_items, transactions)`

Calculates the support for a list of frequent itemsets.

**Parameters**:

- `freq_items`: List of itemsets (list of strings in "feature__value" format).

- `transactions`: A pandas DataFrame of categorical data.

**Returns**: A dictionary of itemset supports and their average support.

### calculate_rule_stats

`calculate_rule_stats(rules, transactions, max_workers=1)`

Evaluates rules with extended metrics including: Support, Confidence, Zhang’s Metric, Dataset Coverage.

Runs in parallel with joblib.

**Parameters**:

- `rules`: List of rule dictionaries.

- `transactions`: One-hot encoded pandas DataFrame.

- `max_workers`: Number of parallel threads (via joblib).

**Returns**:

- A dictionary of average metrics (support, confidence, zhangs_metric, coverage)

- A list of updated rules

## Citation

If you use pyaerial in your work, please cite the following paper:

```
@misc{karabulut2025neurosymbolic,
    title={Neurosymbolic Association Rule Mining from Tabular Data},
    author={Erkan Karabulut and Paul Groth and Victoria Degeler},
    year={2025},
    eprint={2504.19354},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Contact

For questions, suggestions, or collaborations, please contact:

    Erkan Karabulut
    📧 e.karabulut@uva.nl
    📧 erkankkarabulut@gmail.com

## Contributing

Contributions, feedback, and issue reports are very welcome!

Feel free to open a pull request or create an issue if you have ideas for improvements.

