# Advanced Topics

## GPU Usage

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
association_rules = rule_extraction.generate_rules(trained_autoencoder)
...
```

## Debugging

To be able to debug Aerial, this section explains how each of the parameters of Aerial can impact the number and the quality of the rules learned.

### What to do when Aerial does not learn any rules?

Following are some recommendations when Aerial can not find rules, assuming that the data preparation is done correctly (e.g., the data is discretized).

- **Longer training.** Increasing the number of epochs can make Aerial capture associations better. However, training for too long may lead to overfitting, which means non-informative rules with low association strength.
- **Adding more parameters.** Increasing the number of layers and/or dimension of the layers can again allow Aerial to discover associations that was not possible with lower number of parameters. This may require training longer as well.
- **Reducing antecedent similarity threshold.** Antecedent similarity threshold in Aerial is synonymous to minimum support threshold in exhaustive ARM methods. Reducing antecedent similarity threshold will result in more rules with potentially lower support.
- **Reducing consequent similarity threshold.** Consequent similarity threshold of Aerial is synoynmous to minimum confidence threshold in exhaustive ARM methods. Reducing this threshold will result in more rules with potentially lower confidence.

### What to do when Aerial takes too much time and learns too many rules?

Similar to any other ARM algorithm, when performing knowledge discovery by learning rules, it could be the case that the input parameters of the algorithm results in a huge search space and that the underlying hardware does not allow terminating in a reasonable time.

To overcome this, we suggest starting with smaller search spaces and gradually increasing. In the scope of Aerial, this can be done as follows:

1. Start with `max_antecedents=2`, observe the execution time and usefulness of the rules you learned. Then gradually increase this number if necessary for the task you want to achieve.
2. Start with `ant_similarity=0.5`, or even higher if necessary. A high antecedent similarity means you start discovering the most prominent patterns in the data first, that are usually easier to discover. This parameter is synonymous with the minimum support threshold of exhaustive ARM methods such as Apriori or FP-Growth (but not the same).
3. Do not set low `cons_similarity`. The consequent similarity is synonymous to a combination of minimum confidence and zhang's metric thresholds. There is no reason to set this parameter low, e.g., lower than 0.5. Similar to `ant_similarity`, start with a high number such as `0.9` and then gradually decrease if necessary.
4. Train less or use less parameters. If Aerial does not terminate for an unreasonable duration, it could also mean that the model over-fitted the data and is finding many non-informative rules which increase the execution time. To prevent that, start with smaller number of epochs and parameters. For datasets where the number of rows `n` is much bigger than the number columns `d`, such that `n >> d`, usually training for 2 epochs with 2 layers of decreasing dimensions per encoder and decoder is enough.
5. Another alternative is to apply ideas from the ARM rule explosion literature. One of the ideas is to learn rules for items of interest rather than all items (columns). This can be done with Aerial as it is exemplified in [Specifying Item Constraints](user_guide.md#2-specifying-item-constraints) section.
6. If the dataset is big and you needed to create a deeper neural network with many parameters, use GPU rather than a CPU. Please see the [GPU Usage](#gpu-usage) section for details.

Note that it is also always possible that there are no prominent patterns in the data to discover.

### What to do if Aerial produces error messages?

Please create an issue in this repository with the error message and/or send an email to e.karabulut@uva.nl.

## Logging Configuration

Aerial source code prints extra debug statements notifying the beginning and ending of major functions such as the training process or rule extraction. The log levels can be changed as follows:

```python
import logging
import aerial

# setting the log levels to DEBUG level
aerial.setup_logging(logging.DEBUG)
...
```

## Visualization

Rules learned by PyAerial can be visualized using [NiaARM](https://github.com/firefly-cpp/NiaARM) library. In the following, `visualizable_rule_list()` function converts PyAerial's rule format to NiaARM `RuleList()` format. And then visualizes the rules on a scatter plot using the visualization module of NiaARM

```python
...
from niaarm.visualize import scatter_plot
from niaarm import RuleList, Feature, Rule

def visualizable_rule_list(aerial_rules: dict, dataset: pd.DataFrame):
    rule_list = RuleList()
    for rule in aerial_rules:
        antecedents = [Feature(k, "cat", categories=[v]) for k, v in (i.split("__", 1) for i in rule["antecedents"])]
        ck, cv = rule["consequent"].split("__", 1)
        rule_list.append(Rule(antecedents, [Feature(ck, "cat", categories=[cv])], transactions=dataset))
    return rule_list

# learn rules with PyAerial as before
breast_cancer = fetch_ucirepo(id=14).data.features
trained_autoencoder = model.train(breast_cancer)
association_rules = rule_extraction.generate_rules(trained_autoencoder, ant_similarity=0.1)

# get rules in NiaARM RuleList format
visualizable_rules = visualizable_rule_list(association_rules, breast_cancer)
figure = scatter_plot(rules=visualizable_rules, metrics=('support', 'confidence', 'lift'), interactive=False)
figure.show()
```

Visualization of the PyAerial rules as a scatter plot showing their quality metrics:

![visualization.png](../../visualization.png)

Please see NiaARM for more visualization options: https://github.com/firefly-cpp/NiaARM?tab=readme-ov-file#visualization
