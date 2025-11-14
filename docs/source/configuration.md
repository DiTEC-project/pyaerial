# Configuration and Troubleshooting

## GPU Usage

The `device` parameter in `train()` can be used to run Aerial on GPU. Note that Aerial only uses a shallow Autoencoder
and therefore can also run on CPU without a major performance hindrance.

Furthermore, Aerial will also use the device specified in `train()` function for rule extraction, e.g., when performing
forward runs on the trained Autoencoder with the test vectors.

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

# a categorical tabular dataset
breast_cancer = fetch_ucirepo(id=14).data.features

# run Aerial on GPU
trained_autoencoder = model.train(breast_cancer, device="cuda")

# during the rule extraction stage, Aerial will continue to use the device specified above
result = rule_extraction.generate_rules(trained_autoencoder)
print(f"Mined {result['statistics']['rule_count']} rules on GPU")
```

## Logging Configuration

Aerial source code prints extra debug statements notifying the beginning and ending of major functions such as the
training process or rule extraction. The log levels can be changed as follows:

```python
import logging
import aerial

# setting the log levels to DEBUG level
aerial.setup_logging(logging.DEBUG)
...
```

## Training Parameters

The `train()` function allows you to customize various training parameters:

- `autoencoder`: You can implement your own Autoencoder and use it for ARM as part of Aerial, as long as the last layer matches the original version (see our paper or the source code)
- `noise_factor` (default=0.5): amount of random noise (`+-`) added to each neuron of the denoising Autoencoder before the training process
- `lr` (default=5e-3): learning rate
- `epochs` (default=1): number of training epochs
- `batch_size` (default=2): number of batches to train
- `loss_function` (default=torch.nn.BCELoss()): loss function
- `num_workers` (default=1): number of workers for parallel execution

**Example:**

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features

# Customize training parameters
trained_autoencoder = model.train(
    breast_cancer,
    epochs=5,
    lr=1e-3,
    batch_size=4
)

result = rule_extraction.generate_rules(trained_autoencoder)
print(f"Found {result['statistics']['rule_count']} rules")
```

**Note:** Longer training may lead to overfitting, which results in rules with low association strength (Zhang's metric). See [Advanced: Training and Architecture Tuning](#advanced-training-and-architecture-tuning) for more details.

## Debugging

The following is a step by step debugging guide for Aerial.

### What to do when Aerial does not learn any rules?

Following are some recommendations when Aerial can not find rules, assuming that the data preparation is done
correctly (e.g., the data is discretized).

- **Longer training.** Increasing the number of epochs can make Aerial capture associations better. However, training
  for too long may lead to overfitting, which means non-informative rules with low association strength.
- **Adding more parameters.** Increasing the number of layers and/or dimension of the layers can again allow Aerial to
  discover associations that was not possible with lower number of parameters. This may require training longer as well.
- **Reducing antecedent similarity threshold.** Antecedent similarity threshold in Aerial is synonymous to minimum
  support threshold in exhaustive ARM methods. Reducing antecedent similarity threshold will result in more rules with
  potentially lower support.
- **Reducing consequent similarity threshold.** Consequent similarity threshold of Aerial is synonymous to minimum
  confidence threshold in exhaustive ARM methods. Reducing this threshold will result in more rules with potentially
  lower confidence.

### What to do when Aerial takes too much time and learns too many rules?

Similar to any other ARM algorithm, when performing knowledge discovery by learning rules, it could be the case that the
input parameters of the algorithm results in a huge search space and that the underlying hardware does not allow
terminating in a reasonable time.

To overcome this, we suggest starting with smaller search spaces and gradually increasing. In the scope of Aerial, this
can be done as follows:

1. Start with `max_antecedents=2`, observe the execution time and usefulness of the rules you learned. Then gradually
   increase this number if necessary for the task you want to achieve.
2. Start with `ant_similarity=0.5`, or even higher if necessary. A high antecedent similarity means you start
   discovering the most prominent patterns in the data first, that are usually easier to discover. This parameter is
   synonymous with the minimum support threshold of exhaustive ARM methods such as Apriori or FP-Growth (but not the
   same).
3. Do not set low `cons_similarity`. The consequent similarity is synonymous to a combination of minimum confidence and
   zhang's metric thresholds. There is no reason to set this parameter low, e.g., lower than 0.5. Similar to
   `ant_similarity`, start with a high number such as `0.9` and then gradually decrease if necessary.
4. Train less or use less parameters. If Aerial does not terminate for an unreasonable duration, it could also mean that
   the model over-fitted the data and is finding many non-informative rules which increase the execution time. To
   prevent that, start with smaller number of epochs and parameters. For datasets where the number of rows `n` is much
   bigger than the number columns `d`, such that `n >> d`, usually training for 2 epochs with 2 layers of decreasing
   dimensions per encoder and decoder is enough.
5. Another alternative is to apply ideas from the ARM rule explosion literature. One of the ideas is to learn rules for
   items of interest rather than all items (columns). This can be done with Aerial as it is exemplified
   in [Specifying Item Constraints](user_guide.md#2-specifying-item-constraints) section.
6. If the dataset is big and you needed to create a deeper neural network with many parameters, use GPU rather than a
   CPU. Please see the [GPU Usage](#gpu-usage) section for details.

Note that it is also always possible that there are no prominent patterns in the data to discover.

### What to do if Aerial produces error messages?

Please create an issue in this repository with the error message and/or send an email to e.karabulut@uva.nl.

## Advanced: Training and Architecture Tuning

This section is for advanced users who want fine-grained control over Aerial's behavior. For most use cases, the default
settings and the [Parameter Tuning Guide](parameter_guide.md) are sufficient.

### Understanding Overfitting in Knowledge Discovery

Overfitting in knowledge discovery is fundamentally different from overfitting in traditional machine learning:

**Traditional Machine Learning:**

- High training accuracy but low test accuracy
- Model memorizes training data instead of learning generalizable patterns
- Solution: Early stopping, regularization, more training data

**Knowledge Discovery (Association Rule Mining):**

- **More rules with lower average quality**
- Model captures spurious correlations instead of meaningful associations
- Rules have high support and confidence but **low association strength (Zhang's metric)**
- Solution: Shorter training, stronger compression, higher quality thresholds (in addition to early stopping,
  regularization, more training data). Note that PyAerial already implements early stopping.

**Key Insight:** In knowledge discovery, overfitting doesn't mean poor generalization to new data—it means discovering
non-informative patterns that lack genuine associations.

**Signs of Overfitting in Aerial:**

- Many rules with low Zhang's metric (association strength near 0)
- High support and confidence but weak correlations (association strength)
- or higher number of rules with low support and confidence

### Impact of Training Duration (Epochs)

Training duration has a non-intuitive effect on rule quality in knowledge discovery.

**Shorter Training (1-3 epochs):**

- ✅ Fewer, higher-quality rules
- ✅ Captures strong, meaningful associations
- ✅ Higher average Zhang's metric (association strength)
- ✅ Faster execution
- ⚠️ May miss some patterns if data is complex

**Longer Training (5+ epochs):**

- ⚠️ More rules but lower average quality
- ⚠️ Captures spurious correlations and noise
- ⚠️ Lower average Zhang's metric
- ❌ Overfitting to data peculiarities

**Recommendation:**

- **Default (1-2 epochs)** works well for most datasets
- For datasets where `n >> d` (many rows, few columns): 2 epochs is usually sufficient
- Only increase epochs if you're getting no rules and suspect underfitting
- If rules have low Zhang's metric: **reduce epochs**, don't increase

**Example:**

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features

# Shorter training for higher quality rules
trained_autoencoder = model.train(breast_cancer, epochs=2)
result = rule_extraction.generate_rules(trained_autoencoder)

print(f"Rule count: {result['statistics']['rule_count']}")
print(f"Avg Zhang's metric: {result['statistics']['average_zhangs_metric']}")
```

### Impact of Architecture (layer_dims and Compression)

The autoencoder's architecture controls how aggressively it compresses information, which directly affects rule quality.

**Compression Ratio:** How much the autoencoder reduces dimensionality in the bottleneck layer. This is 
controlled by setting the last layer's dimension in `layer_dims`.

`layer_dims = [4, 2]` means 2 hidden layers of dimensions 4 and 2.

**More Aggressive Compression** (smaller `layer_dims`, e.g., `[4, 2]`):

- ✅ **Fewer, higher-quality rules**
- ✅ Forces model to preserve only essential feature relationships
- ✅ Filters out weak or spurious associations
- ✅ Higher average rule quality metrics
- ⚠️ May miss some nuanced patterns

**Less Aggressive Compression** (larger `layer_dims`, e.g., `[50, 25]`):

- ⚠️ More rules but lower average quality
- ⚠️ Preserves weaker associations and noise
- ⚠️ May capture spurious correlations
- ✅ Can discover more nuanced patterns

**Number of Layers:**

- Deeper networks (more layers) allow for more gradual compression
- Shallower networks (fewer layers) force more aggressive compression
- For most tabular datasets: 2-3 hidden layers per encoder/decoder is sufficient

**Recommendation:**

- **Let Aerial decide automatically** (don't specify `layer_dims`) for most use cases
- If you get too many low-quality rules: Use **smaller layer_dims** for stronger compression
- If you get no rules: Use **larger layer_dims** to preserve more associations
- Rule of thumb: Bottleneck dimension should be much smaller than input dimension

**Example:**

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features

# Aggressive compression for higher quality rules
# layer_dims=[4, 2] means: encoder has layers of size 4 then 2, decoder mirrors this
trained_autoencoder = model.train(breast_cancer, layer_dims=[4, 2], epochs=2)

result = rule_extraction.generate_rules(trained_autoencoder)
print(f"Rule count: {result['statistics']['rule_count']}")
print(f"Avg Zhang's metric: {result['statistics']['average_zhangs_metric']}")
```

### Balancing Training and Architecture

**Best Practices:**

1. **Start conservative**: Short training (2 epochs) + moderate compression (auto or `[8, 4]`)
2. **Evaluate quality**: Check Zhang's metric and rule count
3. **Adjust based on results**:
    - Too many low-quality rules → Reduce epochs OR increase compression
    - Too few rules → Reduce compression OR slightly increase epochs
    - Low Zhang's metric → Reduce epochs (likely overfitting)

**Anti-patterns:**

- ❌ Long training (10+ epochs) with weak compression → Maximum overfitting
- ❌ Increasing epochs when Zhang's metric is already low
- ❌ Using very large layer_dims on small datasets

For more details on these experiments, see
this [blog post on scalable knowledge discovery](https://erkankarabulut.github.io/blog/uva-dsc-seminar-scalable-knowledge-discovery/).

## Advanced: Boosting Rule Quality with Tabular Foundation Models

For domains with limited data—such as gene expression datasets with thousands of features but only dozens of samples—traditional rule mining algorithms struggle to discover meaningful patterns. Aerial addresses this challenge by enabling **transfer learning in knowledge discovery**, a paradigm shift from conventional algorithmic methods like Apriori, FP-Growth, and ECLAT that inherently lack this capability.

### The Challenge: High-Dimensional Small Tabular Data

In specialized domains (biomedical research, rare disease analysis, materials science), practitioners often face extreme dimensional imbalance where the number of features far exceeds available samples. Classical ARM algorithms fail in these settings because they cannot leverage knowledge from related domains. Aerial overcomes this limitation by incorporating tabular foundation models—neural networks pre-trained on diverse tabular datasets—that transfer learned representations to discover high-quality rules even from scarce data.

### Transfer Learning Strategies in Aerial

Aerial supports two fine-tuning strategies to adapt foundation models for rule mining:

#### 1. Weight Initialization (WI)

<img src="_static/assets/aerial-wi.png" alt="Weight Initialization Strategy" width="500"/>

The foundation model's pre-trained weights initialize Aerial's autoencoder, preserving learned feature relationships while adapting to the specific rule mining task. This strategy enables the model to leverage patterns from large-scale pre-training while specializing for the target domain.

#### 2. Projection-Guided Fine-Tuning via Double Loss (DL)

<img src="_static/assets/aerial-dl.png" alt="Double Loss Strategy" width="500"/>

This strategy uses a projection encoder to align Aerial's autoencoder reconstructions with embeddings from a tabular foundation model (e.g., TabPFN), jointly optimizing two complementary objectives:

- **Reconstruction loss (L_recon)**: Binary cross-entropy loss ensuring the autoencoder accurately reconstructs the original tabular data
- **Projection loss (L_proj)**: Cosine distance loss aligning the autoencoder's representations with the foundation model's meta-learned embedding space

The combined double loss function is: **L(θ) = L_recon + L_proj**

By optimizing both objectives simultaneously, this strategy encourages the autoencoder to not only reconstruct the input data but also produce representations that are semantically consistent with the foundation model's learned knowledge, leading to higher-quality rules with better generalization.

### Why This Matters

Traditional algorithmic methods operate without learned representations—they mine rules directly from raw data statistics. Aerial fundamentally changes this by:

- **Leveraging pre-trained models**: Enables rule discovery from small specialized datasets by transferring knowledge from foundation models
- **Enabling cross-domain transfer**: Knowledge learned from diverse tabular data transfers to new domains, even with minimal samples
- **Improving rule quality**: Foundation models capture semantic relationships that pure algorithmic methods miss in low-data regimes

### Implementation Note

**Important:** PyAerial does not yet provide out-of-the-box support for tabular foundation model integration. To use these transfer learning strategies, you will need to implement them yourself by:

1. Following the methodology described in [Karabulut et al. (2025)](https://arxiv.org/pdf/2509.20113)
2. Referring to the implementation in the [paper's companion repository](https://github.com/DiTEC-project/rule_learning_high_dimensional_small_tabular_data)
3. Adapting Aerial's autoencoder architecture to incorporate the weight initialization or double loss strategies

The paper provides comprehensive implementation details and the repository contains reference code for both fine-tuning approaches. Future versions of PyAerial may include built-in support for these advanced capabilities.
