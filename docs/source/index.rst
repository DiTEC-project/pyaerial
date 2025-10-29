PyAerial Documentation
======================

Welcome to PyAerial's documentation!

PyAerial is a **Python implementation** of the Aerial scalable neurosymbolic association rule miner for tabular data. It utilizes an under-complete denoising Autoencoder to learn a compact representation of tabular data, and extracts a concise set of high-quality association rules with full data coverage.

Unlike traditional exhaustive methods (e.g., Apriori, FP-Growth), Aerial addresses the **rule explosion** problem by learning neural representations and extracting only the most relevant patterns, making it suitable for large-scale datasets.

Learn more about the architecture, training, and rule extraction in our paper:
`Neurosymbolic Association Rule Mining from Tabular Data <https://arxiv.org/abs/2504.19354>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   api_reference
   advanced_topics
   research
   citation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
