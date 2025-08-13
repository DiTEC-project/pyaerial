# Ilustrative Experiments

This document includes instructions on how to run the experiments given in the paper
"Pyaerial: Scalable Association Rule Mining from Tabular Data"[5].

For a detailed evaluation of Aerial+ association rule mining method, please see
the papers [1] and [2].

### What is it?

The experiments consists of execution time and rule quality experiments on a g**ene
expression dataset** [3] that is pre-processed by [4]. We further pre-processed data
to put in a discrete form using z-score binning. The dataset can be found under the
`illustrative_experiments/data` folder.

### Baselines

Baseline association rule mining libraries written in Python, R and C/C++ include:

1. [MLxtend](https://rasbt.github.io/mlxtend/)
2. [PyECLAT](https://github.com/jeffrichardchemistry/pyECLAT)
3. [arulespy](https://github.com/mhahsler/arulespy)
4. [SPMF](https://www.philippe-fournier-viger.com/spmf/)
5. [NiaARM](https://github.com/firefly-cpp/NiaARM)
6. [ARM-AE](github.com/TheophileBERTELOOT/ARM-AE)

### Dataset

The dataset used in the experiments can be found under `illustrative_experiments/data` folder.

We use a gene expression levels dataset from the biomedical domain [3]. A pre-processed version of
it is taken from [4], and further pre-processed by applying z-score binning to prepare it for ARM.

### Installing Requirements

To be able to run the illustrative experiments, install the dependencies as follows.

**Python requirements**

Required Python libraries are given in `experiment_requirement.txt` of the same folder.
Run the following command to install Python requirements.

```
pip3 install -r experiment_requirements.txt
```

**R requirement to run `arules` library**

[arules](github.com/mhahsler/arules) is an association rule mining library written in R language. We use a
Python wrapper, namely [arulespy](https://github.com/mhahsler/arulespy) to be able to run arules.
arulespy still depends on R and arules R package. Run the following command in R console
to be able to install `arules`

**Running SPMF library**

[SPMF](https://www.philippe-fournier-viger.com/spmf/) is a Java library for ARM, which we access in Python
following the instructions in https://data-mining.philippe-fournier-viger.com/tutorial-how-to-call-spmf-from-python/.
To do so, we placed a `.jar` file under the `illustrative_experiments/algorithms/spmf_source` folder, and
accessed to that file within `illustrative_experiments/algorithms/spmf.py`.

```
install.packages("arules")
```

For more details and debugging, pleas see https://github.com/mhahsler/arules.

```
export PYTHONPATH="../pyaerial/illustrative_experiments/fim.so:$PYTHONPATH"
```

### Running the Experiments

After installing the requirements, simply run the following command to reproduce the experiments
**within the main folder of PyAerial, and not within illustrative_experiments folder**:

```python3 -m illustrative_experiments.experiment_benchmarking```

The output should look as follows:

![sample_output.png](sample_output.png)

The **hyperparameters** of each method can be changed within `illustrative_experiments/experiment_benchmarking.py` file.

### References

1. Karabulut, E., Groth, P. T., & Degeler, V. O. (2025) Neurosymbolic Association Rule Mining from Tabular Data. 19th
   International Conference on Neurosymbolic Learning and Reasoning (NeSy). Accepted/In Press. [pdf]. arXiv doi:
   10.48550/arXiv.2504.19354
2. Karabulut, E., Groth, P. T., & Degeler, V. O. (2025) Learning Semantic Association Rules from Internet of Things
   Data. Neurosymbolic Artificial Intelligence. Accepted/In Press. [pdf]. arXiv doi: 10.48550/arXiv.2412.03417
3. H. Gao, J. M. Korn, S. Ferretti, J. E. Monahan, Y. Wang, M. Singh,
   C. Zhang, C. Schnell, G. Yang, Y. Zhang, et al. High-throughput screen-
   ing using patient-derived tumor xenografts to predict clinical trial drug
   response. Nature medicine, 21(11):1318–1325, 2015.
4. C. Ruiz, H. Ren, K. Huang, and J. Leskovec. High dimensional, tabular
   deep learning with an auxiliary knowledge graph. Advances in Neural
   Information Processing Systems, 36:26348–26371, 2023.
5. Karabulut, Erkan, Paul Groth, and Victoria Degeler. "Pyaerial: Scalable Association Rule Mining from Tabular Data."
   Available at SSRN 5356320.