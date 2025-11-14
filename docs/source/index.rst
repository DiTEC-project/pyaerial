PyAerial Documentation
======================

Welcome to PyAerial's documentation!

PyAerial is a **Python implementation** of the Aerial scalable neurosymbolic association rule miner for tabular data. It utilizes an under-complete denoising Autoencoder to learn a compact representation of tabular data, and extracts a concise set of high-quality association rules with full data coverage.

Unlike traditional exhaustive methods (e.g., Apriori, FP-Growth), Aerial addresses the **rule explosion** problem by learning neural representations and extracting only the most relevant patterns, making it suitable for large-scale datasets.

Learn more about the architecture, training, and rule extraction in our paper:
`Neurosymbolic Association Rule Mining from Tabular Data <https://proceedings.mlr.press/v284/karabulut25a.html>`_

Performance
-----------

PyAerial significantly outperforms traditional ARM methods in **scalability** while maintaining high-quality results:

.. image:: _static/assets/benchmark.png
   :alt: PyAerial performance comparison
   :align: center
   :width: 700px

*Execution time comparison across datasets of varying sizes. PyAerial scales linearly while traditional methods (e.g., Mlxtend, SPMF) exhibit exponential growth.*

**Key advantages:**

- ⚡ **100-1000x faster** on large datasets compared to Apriori/FP-Growth Python implementations
- 📈 **Linear scaling** with dataset size (vs. exponential for traditional methods)
- 🎯 **No rule explosion** - extracts concise, high-quality rules with full data coverage
- 💾 **Memory efficient** - neural representation avoids storing exponential candidate sets

For comprehensive benchmarks and comparisons with Mlxtend (e.g., FPGrowth, Apriori etc.), and other ARM tools, see our benchmarking paper:
`PyAerial: Scalable association rule mining from tabular data <https://www.sciencedirect.com/science/article/pii/S2352711025003073>`_ (SoftwareX, 2025)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   parameter_guide
   configuration
   api_reference
   research
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
