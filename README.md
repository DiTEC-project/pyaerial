# pyaerial: scalable association rule mining

------------------------------

This is a Python implementation of the Aerial scalable neurosymbolic association rule miner for tabular data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
    - [Association rule mining from categorical tabular data](#1-association-rule-mining-from-categorical-tabular-data)
    - [Specifying item constraints](#2-specifying-item-constraints)
    - [Setting Aerial parameters](#3-setting-aerial-parameters)
    - [Fine-tuning Autoencoder architecture and dimensions](#4-fine-tuning-autoencoder-architecture-and-dimensions)
    - [Running Aerial for numerical values](#5-running-aerial-for-numerical-values)
    - [Frequent itemset mining with Aerial](#6-frequent-itemset-mining-with-aerial)
    - [Using Aerial for rule-based classification for interpretable inference](#7-using-aerial-for-rule-based-classification-for-interpretable-inference)
    - [Fine-tuning the training parameters](#8-fine-tuning-the-training-parameters)
    - [Setting the log levels](#9-setting-the-log-levels)
- [How Aerial works?](#how-aerial-works)
- [How to debug Aerial?](#how-to-debug-aerial)
    - [What to do when Aerial does not learn any rules?](#what-to-do-when-aerial-does-not-learn-any-rules)
    - [What to do when Aerial takes too much time and learns too many rules?](#what-to-do-when-aerial-takes-too-much-time-and-learns-too-many-rules)
    - [What to do if Aerial produces error messages?](#what-to-do-if-aerial-produces-error-messages)
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

See [How Aerial works?](#how-aerial-works) section or our paper for the details of Autoencoder architecture, training
and rule extraction algorithm
[Neurosymbolic Association Rule Mining from Tabular Data](https://arxiv.org/abs/2504.19354). If you use Aerial in your
work, please [cite](#citation) our paper.

---

## Installation

You can easily install **pyaerial** using pip:

```bash
pip install pyaerial
```

## Usage Examples

This section exemplifies the usage of Aerial with and without hyperparameter tuning.

If you encounter issues such as Aerial can't learn rules, or takes too much time to terminate, please
see [How to debug Aerial?](#how-to-debug-aerial) section.

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

### 2. Specifying item constraints

Instead of performing rule extraction on all features, Aerial allows you to extract rules only for
features of interest. This is called ARM with item constraints.

In ARM with item constraints, the antecedent side of the rules will contain the items of interest. However, the
consequent side of the rules may still contain other feature values.

`features_of_interest` parameter of
`generate_rules()` can be used to do that (also valid for `generate_frequent_itemsets()`, see below).

```
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

# categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

trained_autoencoder = model.train(breast_cancer)

# features of interest, either a feature with its all values (e.g., "age") or with its certain values (e.g., premeno value of menopause feature is the only feature value of interest)
features_of_interest = ["age", {"menopause": 'premeno'}, 'tumor-size', 'inv-nodes', {"node-caps": "yes"}]

association_rules = rule_extraction.generate_rules(trained_autoencoder, features_of_interest, cons_similarity=0.5)
```

The output rules will only contain features of interest on the antecedent side:

```
>>> Output:
association_rules: [
   {
      "antecedents":[
         "menopause__premeno"
      ],
      "consequent":"node-caps__no",
      ...
   },
   {
      "antecedents":[
         "menopause__premeno"
      ],
      "consequent":"breast__right",
      ...
   },
   ...
]
```

### 3. Setting Aerial parameters

Aerial has 3 key parameters; antecedent and consequent similarity threshold, and antecedent length.

As shown in the paper, higher antecedent thresholds results in lower number of higher support rules, while
higher consequent thresholds results in lower number of higher confidence rules.

These 3 parameters can be set using the `generate_rules` function:

```
import pandas as pd
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features

trained_autoencoder = model.train(breast_cancer)

# hyperparameters of aerial can be set using the generate_rules function
association_rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.5, cons_similarity=0.8, max_antecedents=2)
...
```

### 4. Fine-tuning Autoencoder architecture and dimensions

Aerial uses an under-complete Autoencoder and in default, it decides automatically how many layers to use and the
dimensions of each layer (see [Functions Overview](#functions-overview), Autoencoder).

Alternatively, you can specify the number of layers and dimensions in the `train` method to improve performance.

```
from aerial import model, rule_extraction, rule_quality

...
# layer_dims=[4, 2] specifies that there are gonna be 2 hidden layers with the dimensions 4 and 2, for encoder and decoder
trained_autoencoder = model.train(breast_cancer, layer_dims=[4, 2]) 
...
```

### 5. Running Aerial for numerical values

Discretizing numerical values is required before running Aerial. We provide 2 discretization methods as part of
the [`discretization.py`](aerial/discretization.py) script; equal-frequency and equal-width discretization. However,
Aerial can work with any other discretization method of your choice as well.

```
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# load a numerical tabular data
iris = fetch_ucirepo(id=53).data.features

# find and discretize numerical columns 
iris_discretized = discretization.equal_frequency_discretization(iris, n_bins=5)

trained_autoencoder = model.train(iris_discretized, epochs=10)

association_rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.05, cons_similarity=0.8)
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

### 6. Frequent itemset mining with Aerial

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

### 7. Using Aerial for rule-based classification for interpretable inference

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

### 8. Fine-tuning the training parameters

The [`train()`](aerial/model.py) function allows programmers to specify various training parameters:

- autoencoder: You can implement your own Autoencoder and use it for ARM as part of Aerial, as long as the last layer
  matches the original version (see our paper or the source code, [`model.py`](aerial/model.py))
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

### 9. Setting the log levels

Aerial source code prints extra debug statements notifying the beginning and ending of major
functions such as the training process or rule extraction. The log levels can be changed as follows:

```
import logging
import aerial

# setting the log levels to DEBUG level
aerial.setup_logging(logging.DEBUG)
...
```

### 10. Running Aerial on GPU

The `device` parameter in `train()` can be used to run Aerial on GPU. Note that Aerial only uses a shallow
Autoencoder and therefore can also run on CPU without a major performance hindrance.

Furthermore, Aerial will also use the device specified in `train()` function for rule extraction, e.g., when
performing forward runs on the trained Autoencoder with the test vectors.

```
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

# run Aerial on GPU
trained_autoencoder = model.train(breast_cancer, device="cuda")

# during the rule extraction stage, Aerial will continue to use the device specified above
association_rules = rule_extraction.generate_rules(trained_autoencoder)
...
```

## How Aerial works?

The figure below shows the pipeline of operations for Aerial in 3 main stages.

![Aerial neurosymbolic association rule mining pipeline](pipeline.png)

1. **Data preparation.**
    1. Tabular data is first one-hot encoded. This is done
       using [`data_preparation.py:_one_hot_encoding_with_feature_tracking()`](aerial/data_preparation.py).
    2. One-hot encoded value are then converted to vector format in the [`model.py:train()`](aerial/model.py).
    3. If the tabular data contains numerical columns, they are pre-discretized as exemplified
       in [Running Aerial for numerical values](#4-running-aerial-for-numerical-values).
2. **Training stage.**
    1. An under-complete Autoencoder with either default automatically-picked number of layers and dimension (based on
       the dataset size and dimension) is constructed, or user-specified layers and dimension. (
       see [Autoencoder](#autoencoder--inputdimension-featurecount-layerdimsnone-))
    2. All the training parameters can be customized including number of epochs, batch size, learning rate etc. (
       see [train()](#train-function))
    3. An Autoencoder is then trained with a denoising mechanism to learn associations between input features. The full
       Autoencoder architecture is given in our [paper](https://arxiv.org/abs/2504.19354).
3. **Rule extraction stage.**
    1. Association rules are then extracted from the trained Autoencoder using Aerial's rule extraction algorithm (
       see [rule_extraction.py:generate_rules()](#generaterules)). Below figure shows an example rule extraction
       process.
    2. **Example**. Assume `$weather$` and `$beverage$` are features with categories `{cold, warm}`
       and `{tea, coffee, soda}` respectively.
       The first step is to initialize a test vector of size 5 corresponding to 5 possible categories with equal
       probabilities per feature, `[0.5, 0.5, 0.33, 0.33, 0.33]`. Then we mark `$weather(warm)$` by assigning 1
       to `warm` and 0 to `cold`, `[1, 0, 0.33, 0.33, 0.33]`, and call the resulting vector a *test vector*.

       Assume that after a forward run, `[0.7, 0.3, 0.04, 0.1, 0.86]` is received as the output probabilities. Since
       the probability of `$p_{weather(warm)} = 0.7$` is bigger than the given antecedent similarity
       threshold (`$\tau_a = 0.5$`), and `$p_{beverage(soda)} = 0.86$` probability is higher than the consequent
       similarity threshold (`$\tau_c = 0.8$`), we conclude with `$weather(warm) \rightarrow beverage(soda)$`.

   ![Aerial rule extraction example](example.png)

    3. Frequent itemsets (instead of rules) can also be
       extracted ([rule_extraction.py:generate_frequent_itemsets()](#generatefrequentitemsets)).
    4. Finally various quality criteria is then calculated for each rule as well as an overall average values, .e.g,
       support, confidence, coverage, association strength (zhangs' metric). This is done
       in [rule_quality.py](aerial/rule_quality.py). See [calculate_rule_stats()](#calculaterulestats)
       and [calculate_basic_rule_stats()](#calculatebasicrulestats)

## How to debug Aerial?

To be able to debug Aerial, this section explains how each of the parameters of Aerial can impact the number and the
quality of the rules learned.

### What to do when Aerial does not learn any rules?

Following are some recommendations when Aerial can not find rules, assuming that the data preparation is done
correctly (e.g., the data is discretized).

- **Longer training.** Increasing the number of epochs can make Aerial capture associations better. However,
  training for too long may lead to overfitting, which means non-informative rules with low association strength.
- **Adding more parameters.** Increasing the number of layers and/or dimension of the layers can again allow Aerial
  to discover associations that was not possible with lower number of parameters. This may require training longer as
  well.
- **Reducing antecedent similarity threshold.** Antecedent similarity threshold in Aerial is synonymous to minimum
  support threshold in exhaustive ARM methods. Reducing antecedent similarity threshold will result in more rules with
  potentially lower support.
- **Reducing consequent similarity threshold.** Consequent similarity threshold of Aerial is synoynmous to minimum
  confidence threshold in exhaustive ARM methods. Reducing this threshold will result in more rules with potentially
  lower confidence.

### What to do when Aerial takes too much time and learns too many rules?

Similar to any other ARM algorithm, when performing knowledge discovery by learning rules, it could be the case
that the input parameters of the algorithm results in a huge search space and that the underlying hardware does not
allow terminating in a reasonable time.

To overcome this, we suggest starting with smaller search spaces and gradually increasing. In the scope of
Aerial, this can be done as follows:

1. Start with `max_antecedents=2`, observe the execution time and usefulness of the rules you learned. Then gradually
   increase this number if necessary for the task you want to achieve.
2. Start with `ant_similarity=0.5`, or even higher if necessary. A high antecedent similarity means you start
   discovering the most prominent patterns in the data first, that are usually easier to discover. This parameter is
   synonymous with the minimum support threshold of exhaustive ARM methods such as Apriori or FP-Growth (but not the
   same).
3. Do not set low `cons_similarity`. The consequent similarity is synonymous to a combination of minimum confidence
   and zhang's metric thresholds. There is no reason to set this parameter low, e.g., lower than 0.5. Similar
   to `ant_similarity`, start with a high number such as `0.9` and then gradually decrease if necessary.
4. Train less or use less parameters. If Aerial does not terminate for an unreasonable duration, it could also mean that
   the model over-fitted the data and is finding many non-informative rules which increase the execution time. To
   prevent that, start with smaller number of epochs and parameters. For datasets where the number of rows `n` is much
   bigger than the number columns `d`, such that `n >> d`, usually training for 2 epochs with 2 layers of decreasing
   dimensions per encoder and decoder is enough.
5. Another alternative is to apply ideas from the ARM rule explosion literature. One of the ideas is to learn rules for
   items of interest rather than all items (columns). This can be done with Aerial as it is exemplified
   in [Specifying item constraints](#2-specifying-item-constraints) section.
6. If the dataset is big and you needed to create a deeper neural network with many parameters, use GPU rather than a
   CPU. Please see [Running Aerial on GPU](#10-running-aerial-on-gpu) section for details.

Note that it is also always possible that there are no prominent patterns in the data to discover.

### What to do if Aerial produces error messages?

Please create an issue in this repository with the error message and/or send an email to e.karabulut@uva.nl.

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
        layer_dims=None,
        device=None
    )

Part of the [`model.py`](aerial/model.py) script. Given a categorical tabular data in Pandas dataframe form, it one-hot
encodes the data, vectorizes the one-hot encoded version by also keeping track of start and end indices of vectors per
column, and then trains the AutoEncoder model using the one-hot encoded version.

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

- `layer_dims` (list of int, optional): Custom hidden layer dimensions for autoencoder construction (if applicable).

- `device` (str): Name of the device to run the Autoencoder model on, e.g., "cuda", "cpu" etc. The device option that is
  set here will also be used in the rule extraction stage.

**Returns**: A trained instance of the AutoEncoder.

### generate_rules

    generate_rules(
        autoencoder,
        features_of_interest=None,
        ant_similarity=0.5,
        cons_similarity=0.8,
        max_antecedents=2,
        target_class=None
    )

Part of the [`rule_extraction.py`](aerial/rule_extraction.py) script. Extracts association rules from a trained
AutoEncoder using the Aerial algorithm.

**Parameters**:

- `autoencoder` (AutoEncoder): A trained autoencoder instance.

- `features_of_interest` (list): only look for rules that have these features of interest on the antecedent side
  accepted form ["feature1", "feature2", {"feature3": "value1}, ...], either a feature name as str, or specific value
  of a feature in object form

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
        features_of_interest=None,
        similarity=0.5,
        max_length=2
    )

Part of the [`rule_extraction.py`](aerial/rule_extraction.py) script. Generates frequent itemsets from a trained
AutoEncoder using the same Aerial+ mechanism.

**Parameters**:

- `autoencoder` (AutoEncoder): A trained autoencoder instance.

- `features_of_interest` (list): only look for rules that have these features of interest on the antecedent side
  accepted form ["feature1", "feature2", {"feature3": "value1}, ...], either a feature name as str, or specific value
  of a feature in object form

- `similarity` (float): Minimum similarity threshold for an itemset to be considered frequent.

- `max_length` (int): Maximum number of items in each itemset.

**Returns**:

    A list of frequent itemsets, where each itemset is a list of string features:

    [
        [...],  # e.g., ['gender=Male', 'income=High']
        ...
    ]

### equal_frequency_discretization

    equal_frequency_discretization(df: pd.DataFrame, n_bins=5)

Discretizes all numerical columns into equal-frequency bins and encodes the resulting intervals as string labels.

**Parameters**:

- `df`: A pandas DataFrame containing tabular data.

- `n_bins`: Number of intervals (bins) to create.

**Returns**: A modified DataFrame with numerical columns replaced by string-encoded interval bins.

### equal_width_discretization

`equal_width_discretization(df: pd.DataFrame, n_bins=5)`

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

