# Parameter Tuning Guide

> **📝 Note on Parameter Names:** The parameters `min_rule_frequency` and `min_rule_strength` correspond to `ant_similarity` and `con_similarity` in the original [Aerial](https://proceedings.mlr.press/v284/karabulut25a.html) and [PyAerial](https://doi.org/10.1016/j.softx.2025.102341) papers.

> **🖥️ CPU Performance:** PyAerial runs fast on CPU. GPU acceleration is optional and only beneficial for very large datasets.

**On this page:** [Golden Rules](#golden-rules) · [Defaults](#default-parameter-values) · [Quick Reference](#quick-parameter-reference) · [Tuning for Specific Goals](#tuning-for-specific-goals) · [Training and Rule Quality](#training-parameters-and-rule-quality) · [Common Scenarios](#common-scenarios) · [When Things Don't Work](#when-things-dont-work)

## Golden Rules

1. Association rule mining is a **knowledge discovery** task.
2. Knowledge discovery is **unsupervised**.
3. There are **NO GROUND TRUTHS** or **best rules** in knowledge discovery, unless an explicit objective is given.

**Therefore**, we need to tell the algorithm what kind of patterns/rules we are looking for via the parameters.

## Default Parameter Values

Aerial uses these defaults when you don't specify parameters:

### Core Rule Extraction Parameters ⭐

| Parameter         | Default | What it Controls                                               |
|-------------------|---------|----------------------------------------------------------------|
| `min_rule_frequency`  | `0.5`   | How frequent patterns must be (analogous to minimum support)   |
| `min_rule_strength`   | `0.8`   | How reliable rules must be (confidence + association strength) |
| `max_antecedents` | `2`     | Maximum complexity (number of conditions per rule); `None` = unlimited |

### Training Parameters

| Parameter      | Default       | What it Controls                                                                 |
|----------------|---------------|----------------------------------------------------------------------------------|
| `epochs`       | `5`           | Training iterations (shorter training = fewer, higher-quality rules)             |
| `lr`           | `5e-3`        | Learning rate                                                                    |
| `batch_size`   | Auto          | Training batch size (based on dataset size)                                      |
| `layer_dims`   | Auto-selected | Autoencoder architecture (smaller = stronger compression = higher quality rules) |
| `show_progress`| `True`        | Show a progress bar during training                                              |
| `min_unmasked_features`| `1`   | Fewest features left unmasked per training batch                                 |
| `max_unmasked_features`| `10`  | Most features left unmasked per training batch                                   |

> **💡 Tip:** Train less for fewer, higher-quality rules. Only increase epochs if you are not getting enough rules for
> your use case — see [Training and Rule Quality](#training-parameters-and-rule-quality).

### Filtering Parameters

| Parameter               | Default                     | What it Controls                                                        |
|-------------------------|------------------------------|--------------------------------------------------------------------------|
| `filter_min_confidence` | Mirrors `min_rule_strength`  | Post-filter rules whose *exact* confidence is below this value           |
| `filter_min_support`    | `0.0001`                     | Post-filter rules whose *exact* support is below this value              |

> **Note:** `filter_min_confidence` defaults to whatever `min_rule_strength` you used for the call, since both are
> conditional-probability-like quantities. `filter_min_support` is **not** mirrored to `min_rule_frequency` — rule
> support (antecedent + consequent) is mathematically ≤ antecedent-only frequency, so the two aren't comparable at
> the same threshold; it instead defaults to a near-zero floor that only excludes degenerate rules.

### Other Parameters

| Parameter              | Default                                      | What it Controls                                     |
|------------------------|----------------------------------------------|------------------------------------------------------|
| `features_of_interest` | `None`                                       | Focus mining on specific features (item constraints) |
| `target_classes`       | `None`                                       | Restrict rules to predict specific class labels      |
| `quality_metrics`      | `['support', 'confidence', 'zhangs_metric']` | Which metrics to calculate                           |
| `num_workers`          | `1`                                          | Parallel processing (set to 4-8 for 1000+ rules)     |
| `device`               | Auto                                         | `'cuda'` for GPU or `'cpu'`                          |

**💡 Tip:** Start with defaults, then adjust the 3 core parameters based on your goals below.

## Quick Parameter Reference

| I want...                                | Set these parameters                          | Example                                                        |
|------------------------------------------|-----------------------------------------------|----------------------------------------------------------------|
| High-quality rules only (post-filtering) | `filter_min_confidence=0.7, filter_min_support=0.1` | `generate_rules(model, filter_min_confidence=0.7, filter_min_support=0.1)` |
| High support rules                       | `min_rule_frequency=0.7` (or higher)              | `generate_rules(model, min_rule_frequency=0.7)`                    |
| Low support rules                        | `min_rule_frequency=0.1` (or lower)               | `generate_rules(model, min_rule_frequency=0.1)`                    |
| High confidence rules                    | `min_rule_strength=0.8` (or higher)               | `generate_rules(model, min_rule_strength=0.8)`                     |
| Fewer rules                              | Increase `min_rule_frequency` and `min_rule_strength` | `generate_rules(model, min_rule_frequency=0.6, min_rule_strength=0.8)` |
| More rules                               | Decrease `min_rule_frequency` and `min_rule_strength` | `generate_rules(model, min_rule_frequency=0.2, min_rule_strength=0.5)` |
| Complex patterns                         | `max_antecedents=3` (or higher)               | `generate_rules(model, max_antecedents=3)`                     |
| Antecedents of unlimited length          | `max_antecedents=None`                        | `generate_rules(model, max_antecedents=None)`                  |
| Faster quality-metric calculation        | `num_workers=4` (or higher, for 1000+ rules)  | `generate_rules(model, num_workers=4)`                         |
| Rules predicting a class label           | `target_classes=[...]`                        | `generate_rules(model, target_classes=["Class"])`              |

**Note** that `filter_min_confidence` and `filter_min_support` are post-processing filters.

## Tuning for Specific Goals

### Support (via `min_rule_frequency`)

**Support** measures how frequently a pattern appears in your data. `min_rule_frequency` is analogous to the minimum
support threshold of traditional ARM methods.

| Direction | Set | What to expect |
|---|---|---|
| High support (common, generalizable patterns) | `min_rule_frequency=0.6-0.8` | Fewer rules, covering larger portions of the data |
| Low support (rare patterns) | `min_rule_frequency=0.05-0.2` | More rules, smaller coverage each, longer runtime |

**Performance:** rule extraction grows antecedent combinations from frequent ones only, so a lower `min_rule_frequency`
admits more combinations into the search and increases runtime. Start with moderate values (e.g., 0.3) and decrease
gradually.

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features
trained_autoencoder = model.train(breast_cancer)

# high support: common patterns
result = rule_extraction.generate_rules(trained_autoencoder, min_rule_frequency=0.7)

# low support: rare patterns
result = rule_extraction.generate_rules(trained_autoencoder, min_rule_frequency=0.1)
```

### Confidence (via `min_rule_strength`)

**Confidence** measures how often the rule's prediction is correct. `min_rule_strength` is analogous to a combination
of minimum confidence and Zhang's metric thresholds.

| Direction | Set | What to expect |
|---|---|---|
| High confidence (reliable rules) | `min_rule_strength=0.8-0.9` | Fewer, more reliable rules; >0.9 may yield very few |
| Low confidence (exploratory analysis) | `min_rule_strength=0.3-0.5` | More rules, weaker relationships, possible spurious correlations |

```python
# reliable rules
result = rule_extraction.generate_rules(trained_autoencoder, min_rule_strength=0.9)

# exploratory analysis
result = rule_extraction.generate_rules(trained_autoencoder, min_rule_strength=0.4)
```

### Association Strength (Zhang's Metric)

**Association strength** measures the correlation between antecedent and consequent, accounting for the prevalence of
both. This is unlike confidence, which is simply P(consequent|antecedent). Rules with high association strength are less
likely to be coincidental. `min_rule_strength` incorporates association strength, so increasing it (0.7+) yields
stronger associations.

**If Zhang's metric is unexpectedly low across your rules**, the Autoencoder likely over-fitted: reduce `epochs` or
use a smaller `layer_dims` — see [Training and Rule Quality](#training-parameters-and-rule-quality).

```python
result = rule_extraction.generate_rules(trained_autoencoder, min_rule_strength=0.7)
for rule in result['rules']:
    print(f"Zhang's metric: {rule['zhangs_metric']}")
```

### Controlling the Number of Rules

| Goal | `min_rule_frequency` | `min_rule_strength` | `max_antecedents` |
|---|---|---|---|
| Fewer rules | 0.6–0.8 | 0.7–0.9 | 1–2 |
| More rules | 0.1–0.3 | 0.3–0.6 | 3+ |

```python
# concise set of strong rules
result = rule_extraction.generate_rules(
    trained_autoencoder, min_rule_frequency=0.6, min_rule_strength=0.8, max_antecedents=2)

# comprehensive exploration
result = rule_extraction.generate_rules(
    trained_autoencoder, min_rule_frequency=0.2, min_rule_strength=0.5, max_antecedents=3)
```

## Training Parameters and Rule Quality

How long you train and how strongly the Autoencoder compresses directly affect rule quality.

### Overfitting in Knowledge Discovery

Overfitting here does not mean poor generalization to new data — it means discovering **more rules with lower average
quality**: the model captures spurious correlations instead of meaningful associations.

**Signs of overfitting:** many rules with low Zhang's metric (association strength near 0), or a large number of
low-support, low-confidence rules.

**Solution:** shorter training, stronger compression, higher quality thresholds.

### Training Duration (`epochs`)

| | Shorter training (1–5 epochs) | Longer training (10+ epochs) |
|---|---|---|
| Rules | ✅ Fewer, higher-quality | ⚠️ More, lower average quality |
| Associations | ✅ Strong, meaningful; higher Zhang's metric | ⚠️ Spurious correlations, noise; lower Zhang's metric |
| Caveat | May miss patterns in complex data | Overfits to data peculiarities |

**Recommendation:** the default works for most datasets. Only increase epochs if you get no rules and suspect
underfitting. If rules have low Zhang's metric, **reduce** epochs — don't increase.

### Architecture (`layer_dims` and Compression)

`layer_dims=[4, 2]` means two hidden layers of dimensions 4 and 2; the last dimension is the bottleneck, and the
decoder mirrors the encoder. Smaller dimensions mean stronger compression.

| | Stronger compression (e.g., `[4, 2]`) | Weaker compression (e.g., `[50, 25]`) |
|---|---|---|
| Rules | ✅ Fewer, higher-quality | ⚠️ More, lower average quality |
| Associations | ✅ Only essential relationships preserved | ⚠️ Weak associations and noise preserved |
| Caveat | May miss nuanced patterns | May capture spurious correlations |

**Recommendation:** let Aerial pick `layer_dims` automatically. Too many low-quality rules → smaller `layer_dims`;
no rules → larger `layer_dims`. For most tabular datasets, 1–2 hidden layers are sufficient.

### Balancing Training and Architecture

1. **Start conservative**: default epochs + auto architecture
2. **Evaluate**: check rule count and average Zhang's metric in `result['statistics']`
3. **Adjust**:
   - Too many low-quality rules → reduce epochs OR increase compression
   - Too few rules → reduce compression OR slightly increase epochs
   - Low Zhang's metric → reduce epochs (likely overfitting)

**Anti-patterns:** ❌ long training with weak compression (maximum overfitting) · ❌ increasing epochs when Zhang's
metric is already low · ❌ very large `layer_dims` on small datasets.

```python
# shorter training + stronger compression for higher-quality rules
trained_autoencoder = model.train(breast_cancer, epochs=2, layer_dims=[4])
result = rule_extraction.generate_rules(trained_autoencoder)
print(result['statistics']['rule_count'], result['statistics']['average_zhangs_metric'])
```

For the experiments behind these recommendations, see
this [blog post on scalable knowledge discovery](https://erkankarabulut.github.io/blog/uva-dsc-seminar-scalable-knowledge-discovery/).

## Common Scenarios

### Scenario 0: Don't Know What Parameters to Use

Start with defaults and filter results.

```python
from aerial import model, rule_extraction
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=14).data.features
trained_autoencoder = model.train(breast_cancer)

# extract rules with default thresholds, keep only high-quality ones
result = rule_extraction.generate_rules(trained_autoencoder, filter_min_confidence=0.6)
```

### Scenario 1: Finding Rare but Strong Patterns

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.05,  # low frequency for rare patterns
    min_rule_strength=0.8,    # high strength for reliable rules
    max_antecedents=2)
```

### Scenario 2: Quick Overview of Main Patterns

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.5,   # common patterns
    min_rule_strength=0.7,
    max_antecedents=2)        # limit complexity for interpretability
```

### Scenario 3: Comprehensive Exploration

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    min_rule_frequency=0.1,   # catch rare patterns
    min_rule_strength=0.5,
    max_antecedents=3)        # allow complex patterns
```

### Scenario 4: Classification Rules

```python
result = rule_extraction.generate_rules(
    trained_autoencoder,
    target_classes=["Class"],  # class label column on the consequent side
    min_rule_strength=0.7,
    min_rule_frequency=0.3)
```

### Scenario 5: Focused Mining with Item Constraints

Mine rules for specific features of interest instead of the entire feature space — useful when you have domain
knowledge about important features, or to reduce the search space and runtime.

```python
features_of_interest = [
    "age",                      # all values of 'age'
    {"menopause": "premeno"},   # only 'premeno' value of 'menopause'
    "tumor-size",
    {"node-caps": "yes"},
]

result = rule_extraction.generate_rules(
    trained_autoencoder, features_of_interest,
    min_rule_frequency=0.3, min_rule_strength=0.6)
```

**Note:** `features_of_interest` constrains the antecedent (left) side of rules. To constrain the consequent (right)
side, use `target_classes` (see Scenario 4).

## When Things Don't Work

### Aerial does not learn any rules

Assuming the data is prepared correctly (e.g., numerical columns are discretized):

- **Train longer.** More epochs can help Aerial capture associations — but see the overfitting warning
  [above](#training-parameters-and-rule-quality).
- **Add parameters.** Larger `layer_dims` can preserve associations that stronger compression would drop.
- **Reduce `min_rule_frequency`.** More rules with potentially lower support.
- **Reduce `min_rule_strength`.** More rules with potentially lower confidence.

Note that it is always possible there are no prominent patterns in the data to discover.

### Aerial takes too long or learns too many rules

Start with a small search space and grow it gradually:

1. **Start with `max_antecedents=2`**, observe execution time and rule usefulness, then increase if needed.
2. **Start with `min_rule_frequency=0.5` or higher** — discover the most prominent patterns first.
3. **Do not set `min_rule_strength` low** (below 0.5): there is rarely a reason to. Start high (e.g., 0.9) and
   decrease gradually.
4. **Train less or use fewer parameters.** Overfitting produces many non-informative rules that slow extraction —
   see [Training and Rule Quality](#training-parameters-and-rule-quality).
5. **Use item constraints** to mine rules only for features of interest
   (see [Specifying Item Constraints](user_guide.md#2-specifying-item-constraints)).
6. **Use a GPU** for big datasets with deep networks (see [GPU Usage](configuration.md#gpu-usage)).

### Aerial produces error messages

See [Getting Help](configuration.md#getting-help).

## Parameter Summary

| Parameter         | Analogous to (in traditional ARM)   | Effect when increased                 | Effect when decreased            |
|-------------------|-------------------------------------|---------------------------------------|----------------------------------|
| `min_rule_frequency`  | Minimum support                     | Fewer, higher support rules           | More, lower support rules        |
| `min_rule_strength` | Minimum confidence + Zhang's metric | Fewer, higher confidence rules        | More, lower confidence rules     |
| `max_antecedents` | Maximum itemset size                | More complex patterns, longer runtime | Simpler patterns, faster runtime |
| `epochs`          | —                                   | More rules, lower average quality     | Fewer, higher-quality rules      |
| `layer_dims`      | —                                   | More rules, lower average quality     | Fewer, higher-quality rules      |

For a deeper understanding of how each parameter affects rule quality, see this detailed blog
post: [Scalable Knowledge Discovery with PyAerial](https://erkankarabulut.github.io/blog/uva-dsc-seminar-scalable-knowledge-discovery/)

## Next Steps

- Check [User Guide](user_guide.md) for usage examples
- Review [API Reference](api_reference.md) for complete parameter documentation
- See [Configuration](configuration.md) for GPU usage, logging, and advanced topics