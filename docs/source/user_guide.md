# User Guide

This section provides detailed examples of using Aerial with various configurations and use cases.

If you encounter issues such as Aerial can't learn rules, or takes too much time to terminate, please see the [Debugging section](advanced_topics.md#debugging).

## Basic Usage Examples

### 1. Association Rule Mining from Categorical Tabular Data

```python
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

# load a categorical tabular dataset from the UCI ML repository
breast_cancer = fetch_ucirepo(id=14).data.features

# train an autoencoder on the loaded table
trained_autoencoder = model.train(breast_cancer)

# extract association rules with quality metrics calculated automatically
result = rule_extraction.generate_rules(trained_autoencoder)

# access rules and statistics
if len(result['rules']) > 0:
    print(result['statistics'])
    print(result['rules'][0])
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

**Working with rules:**

Rules are returned in a structured dictionary format with quality metrics included:

```python
# Accessing rule components and quality metrics
for rule in result['rules']:
    # Access antecedent features
    for ant in rule['antecedents']:
        feature_name = ant['feature']  # e.g., "inv-nodes"
        feature_value = ant['value']   # e.g., "0-2"

    # Access consequent
    cons_feature = rule['consequent']['feature']  # e.g., "node-caps"
    cons_value = rule['consequent']['value']      # e.g., "no"

    # Access quality metrics (automatically calculated)
    support = rule['support']
    confidence = rule['confidence']
    zhangs_metric = rule['zhangs_metric']
    rule_coverage = rule['rule_coverage']  # antecedent support
```

### 2. Specifying Item Constraints

Instead of performing rule extraction on all features, Aerial allows you to extract rules only for features of interest. This is called ARM with item constraints.

In ARM with item constraints, the antecedent side of the rules will contain the items of interest. However, the consequent side of the rules may still contain other feature values (to restrict the consequent side as well, see [Using Aerial for Rule-Based Classification](#7-using-aerial-for-rule-based-classification-for-interpretable-inference)).

`features_of_interest` parameter of `generate_rules()` can be used to do that (also valid for `generate_frequent_itemsets()`, see below).

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

# categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

trained_autoencoder = model.train(breast_cancer)

# features of interest, either a feature with its all values (e.g., "age") or with its certain values (e.g., premeno value of menopause feature is the only feature value of interest)
features_of_interest = ["age", {"menopause": 'premeno'}, 'tumor-size', 'inv-nodes', {"node-caps": "yes"}]

result = rule_extraction.generate_rules(trained_autoencoder, features_of_interest, cons_similarity=0.5)
```

The output rules will only contain features of interest on the antecedent side:

```
>>> Output:
result['rules']: [
   {
      "antecedents": [
         {"feature": "menopause", "value": "premeno"}
      ],
      "consequent": {"feature": "node-caps", "value": "no"},
      "support": 0.357,
      "confidence": 0.68,
      "zhangs_metric": -0.066,
      "rule_coverage": 0.525
   },
   {
      "antecedents": [
         {"feature": "menopause", "value": "premeno"}
      ],
      "consequent": {"feature": "breast", "value": "right"},
      "support": 0.245,
      "confidence": 0.72,
      "zhangs_metric": 0.124,
      "rule_coverage": 0.525
   },
   ...
]
```

### 3. Setting Aerial Parameters

Aerial has 3 key parameters; antecedent and consequent similarity threshold, and antecedent length.

As shown in the paper, higher antecedent thresholds results in lower number of higher support rules, while higher consequent thresholds results in lower number of higher confidence rules.

These 3 parameters can be set using the `generate_rules` function:

```python
import pandas as pd
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features

trained_autoencoder = model.train(breast_cancer)

# hyperparameters of aerial can be set using the generate_rules function
association_rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.5, cons_similarity=0.8, max_antecedents=2)
...
```

### 4. Fine-tuning Autoencoder Architecture and Dimensions

Aerial uses an under-complete Autoencoder and in default, it decides automatically how many layers to use and the dimensions of each layer (see [API Reference](api_reference.md)).

Alternatively, you can specify the number of layers and dimensions in the `train` method to improve performance.

```python
from aerial import model, rule_extraction, rule_quality

...
# layer_dims=[4, 2] specifies that there are gonna be 2 hidden layers with the dimensions 4 and 2, for encoder and decoder
trained_autoencoder = model.train(breast_cancer, layer_dims=[4, 2])
...
```

### 5. Running Aerial for Numerical Values

Discretizing numerical values is required before running Aerial. We provide 2 discretization methods as part of the `discretization.py` script; equal-frequency and equal-width discretization. However, Aerial can work with any other discretization method of your choice as well.

```python
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# load a numerical tabular data
iris = fetch_ucirepo(id=53).data.features

# find and discretize numerical columns
iris_discretized = discretization.equal_frequency_discretization(iris, n_bins=3)

trained_autoencoder = model.train(iris_discretized, epochs=10)

result = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.1, cons_similarity=0.8)
print(f"Found {result['statistics']['rule_count']} rules")
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

### 6. Frequent Itemset Mining with Aerial

Aerial can also be used for frequent itemset mining besides association rules.

```python
from aerial import model, rule_extraction, rule_quality
from ucimlrepo import fetch_ucirepo

# categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features
trained_autoencoder = model.train(breast_cancer, epochs=5, lr=1e-3)

# extract frequent itemsets with support values calculated automatically
result = rule_extraction.generate_frequent_itemsets(trained_autoencoder)

# access itemsets and statistics
print(f"Found {result['statistics']['itemset_count']} itemsets")
print(f"Average support: {result['statistics']['average_support']}")
```

The following is a sample output:

```
>>> Output:

Found 15 itemsets
Average support: 0.295

Itemsets with support values:
[
   {
      'itemset': [{'feature': 'menopause', 'value': 'premeno'}],
      'support': 0.524
   },
   {
      'itemset': [{'feature': 'menopause', 'value': 'ge40'}],
      'support': 0.451
   },
   {
      'itemset': [{'feature': 'menopause', 'value': 'premeno'}, {'feature': 'age', 'value': '30-39'}],
      'support': 0.312
   },
   ...
]
```

### 7. Using Aerial for Rule-Based Classification for Interpretable Inference

Aerial can be used to learn rules with a class label on the consequent side, which can later be used for inference either by themselves or as part of rule list or rule set classifiers (e.g., from [imodels](https://github.com/csinva/imodels) repository).

This is done by setting `target_classes` parameter of the `generate_rules` function. This parameter refers to the class label(s) column of the tabular data.

As shown in [Specifying Item Constraints](#2-specifying-item-constraints), we can also specify multiple target classes and/or their specific values. `["Class1", {"Class2": "value2"}]` array specifies that we are interested in all values of `Class1` and specifically `value2` of `Class2` in the consequent side of the rules.

```python
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

# generate rules with a target class(es), this learns rules that has the "target_classes" column (in this case this column is called "Class") on the consequent side
result = rule_extraction.generate_rules(trained_autoencoder, target_classes=["Class"], cons_similarity=0.5)

if len(result['rules']) > 0:
    print(f"Generated {result['statistics']['rule_count']} classification rules")
    print(f"Average confidence: {result['statistics']['average_confidence']}")
```

Sample output showing rules with class labels on the right hand side:

```
>>> Output:

Generated 12 classification rules
Average confidence: 0.742

Sample rule:
{
   "antecedents": [
      {"feature": "menopause", "value": "premeno"}
   ],
   "consequent": {"feature": "Class", "value": "no-recurrence-events"},
   "support": 0.357,
   "confidence": 0.68,
   "zhangs_metric": -0.066,
   "rule_coverage": 0.525
}
```

### 8. Fine-tuning the Training Parameters

The `train()` function allows programmers to specify various training parameters:

- `autoencoder`: You can implement your own Autoencoder and use it for ARM as part of Aerial, as long as the last layer matches the original version (see our paper or the source code)
- `noise_factor` (default=0.5): amount of random noise (`+-`) added to each neuron of the denoising Autoencoder before the training process
- `lr` (default=5e-3): learning rate
- `epochs` (default=1): number of training epochs
- `batch_size` (default=2): number of batches to train
- `loss_function` (default=torch.nn.BCELoss()): loss function
- `num_workers` (default=1): number of workers for parallel execution

```python
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

# increasing epochs to 5, note that longer training may lead to overfitting which results in rules with low association strength (zhangs' metric)
trained_autoencoder = model.train(breast_cancer, epochs=5, lr=1e-3)

result = rule_extraction.generate_rules(trained_autoencoder)
if len(result['rules']) > 0:
    print(f"Found {result['statistics']['rule_count']} rules")
    print(f"Average Zhang's metric: {result['statistics']['average_zhangs_metric']}")
```

### 9. Setting the Log Levels

Aerial source code prints extra debug statements notifying the beginning and ending of major functions such as the training process or rule extraction. The log levels can be changed as follows:

```python
import logging
import aerial

# setting the log levels to DEBUG level
aerial.setup_logging(logging.DEBUG)
...
```

### 10. Running Aerial on GPU

The `device` parameter in `train()` can be used to run Aerial on GPU. Note that Aerial only uses a shallow Autoencoder and therefore can also run on CPU without a major performance hindrance.

Furthermore, Aerial will also use the device specified in `train()` function for rule extraction, e.g., when performing forward runs on the trained Autoencoder with the test vectors.

```python
from aerial import model, rule_extraction, rule_quality, discretization
from ucimlrepo import fetch_ucirepo

# a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

# run Aerial on GPU
trained_autoencoder = model.train(breast_cancer, device="cuda")

# during the rule extraction stage, Aerial will continue to use the device specified above
result = rule_extraction.generate_rules(trained_autoencoder)
print(f"Mined {result['statistics']['rule_count']} rules on GPU")
```

### 11. Visualizing Association Rules

Rules learned by PyAerial can be visualized using [NiaARM](https://github.com/firefly-cpp/NiaARM) library. In the following, `visualizable_rule_list()` function converts PyAerial's rule format to NiaARM `RuleList()` format. And then visualizes the rules on a scatter plot using the visualization module of NiaARM

```python
...
from niaarm.visualize import scatter_plot
from niaarm import RuleList, Feature, Rule

def visualizable_rule_list(aerial_result: dict, dataset: pd.DataFrame):
    rule_list = RuleList()
    for rule in aerial_result['rules']:
        # Convert dictionary format to NiaARM Feature format
        antecedents = [Feature(ant['feature'], "cat", categories=[ant['value']]) for ant in rule["antecedents"]]
        consequent = Feature(rule["consequent"]['feature'], "cat", categories=[rule["consequent"]['value']])
        rule_list.append(Rule(antecedents, [consequent], transactions=dataset))
    return rule_list

# learn rules with PyAerial as before
breast_cancer = fetch_ucirepo(id=14).data.features
trained_autoencoder = model.train(breast_cancer)
result = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.1)

# get rules in NiaARM RuleList format
visualizable_rules = visualizable_rule_list(result, breast_cancer)
figure = scatter_plot(rules=visualizable_rules, metrics=('support', 'confidence', 'lift'), interactive=False)
figure.show()
```

Visualization of the PyAerial rules as a scatter plot showing their quality metrics:

![visualization.png](../../visualization.png)

Please see NiaARM for more visualization options: https://github.com/firefly-cpp/NiaARM?tab=readme-ov-file#visualization
