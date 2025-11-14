# How Aerial Works

## Introduction

Aerial is a **scalable neurosymbolic association rule mining (ARM) method** for tabular data.

It addresses the **rule explosion** and **execution time** issues in classical ARM by combining:

- **Autoencoder-based neural representation** of tabular data
- **Rule extraction** from learned neural embeddings

Learn more about the architecture, training, and rule extraction in our paper:
[Neurosymbolic Association Rule Mining from Tabular Data](https://proceedings.mlr.press/v284/karabulut25a.html)

## Pipeline Overview

The figure below shows the pipeline of operations for Aerial in 3 main stages.

![Aerial neurosymbolic association rule mining pipeline](_static/assets/pipeline.png)

### 1. Data Preparation

1. Tabular data is first one-hot encoded. This is done using `data_preparation.py:_one_hot_encoding_with_feature_tracking()`.
2. One-hot encoded values are then converted to vector format in the `model.py:train()`.
3. If the tabular data contains numerical columns, they are pre-discretized as exemplified in [Running Aerial for numerical values](user_guide.md#5-running-aerial-for-numerical-values).

### 2. Training Stage

1. An under-complete Autoencoder with either default automatically-picked number of layers and dimension (based on the dataset size and dimension) is constructed, or user-specified layers and dimension. (see [AutoEncoder](api_reference.md#autoencoder))
2. All the training parameters can be customized including number of epochs, batch size, learning rate etc. (see [train() function](api_reference.md#train))
3. An Autoencoder is then trained with a denoising mechanism to learn associations between input features. The full Autoencoder architecture is given in our [paper](https://proceedings.mlr.press/v284/karabulut25a.html).

### 3. Rule Extraction Stage

1. Association rules are then extracted from the trained Autoencoder using Aerial's rule extraction algorithm (see [rule_extraction.py:generate_rules()](api_reference.md#generate_rules)). Below figure shows an example rule extraction process.

2. **Example**. Assume `weather` and `beverage` are features with categories `{cold, warm}` and `{tea, coffee, soda}` respectively.

   The first step is to initialize a test vector of size 5 corresponding to 5 possible categories with equal probabilities per feature, `[0.5, 0.5, 0.33, 0.33, 0.33]`. Then we mark `weather(warm)` by assigning 1 to `warm` and 0 to `cold`, `[1, 0, 0.33, 0.33, 0.33]`, and call the resulting vector a *test vector*.

   Assume that after a forward run, `[0.7, 0.3, 0.04, 0.1, 0.86]` is received as the output probabilities. Since the probability of `p_weather(warm) = 0.7` is bigger than the given antecedent similarity threshold (`τ_a = 0.5`), and `p_beverage(soda) = 0.86` probability is higher than the consequent similarity threshold (`τ_c = 0.8`), we conclude with `weather(warm) → beverage(soda)`.

   ![Aerial rule extraction example](_static/assets/example.png)

3. Frequent itemsets (instead of rules) can also be extracted ([rule_extraction.py:generate_frequent_itemsets()](api_reference.md#generate_frequent_itemsets)).

4. Quality metrics (support, confidence, coverage, Zhang's metric, lift, conviction, Yule's Q, interestingness) are calculated automatically during rule extraction using optimized batch processing with optional parallelization support.

## Key Features

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
