"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

Construct an Autoencoder for association rule mining as described in the paper (Neurosymbolic association rule mining
from tabular data - https://proceedings.mlr.press/v284/karabulut25a.html)
"""

import os
import logging
from datetime import datetime

import torch
import pandas as pd
from torch import nn
import math
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader

from aerial.data_preparation import _one_hot_encoding_with_feature_tracking

logger = logging.getLogger("aerial")


class AutoEncoder(nn.Module):
    """
    This autoencoder is used to create a neural representation of tabular data for association rule mining
    """

    def __init__(self, input_dimension, feature_count, layer_dims: list = None):
        """
        The init function can either construct an under-complete Autoencoder based on the input dimension and feature
        count, automatically deciding the number of layers and layer dimensions.
        Or, if specified by the user, it can also use the layer counts and dimensions from the user.
        Note that fine-tuning layer count and dimensions based on your table dimension and size
        will result in better performance in general

        :param input_dimension: number of features after one-hot encoding (input dimension)
        :param feature_count: target feature count (initial column count of tabular data)
        :param layer_dims: (optional) list of int, specific dimensions for hidden layers
                           (excluding input/output dimensions)
        """
        super().__init__()

        self.input_dimension = input_dimension
        self.feature_count = feature_count
        self.input_vectors = None
        self.feature_value_indices = None
        self.feature_values = None

        # Determine the layer dimensions
        if layer_dims is None:
            # Compute default number of layers based on log base 32 (slower growth for fewer parameters)
            layer_count = max(1, math.ceil(math.log(input_dimension, 32)) - 1)

            # Calculate dimensions with consistent reduction ratio
            reduction_ratio = (feature_count / input_dimension) ** (1 / (layer_count))
            dimensions = [input_dimension]
            for i in range(1, layer_count):
                next_dim = max(feature_count, int(dimensions[-1] * reduction_ratio))
                dimensions.append(next_dim)
            # smaller dimensions lead to fewer higher quality rules, which is the desired outcome in most use cases
            dimensions.append(min(feature_count, 2))
        else:
            # Use provided layer dimensions, adding input and output dimensions
            dimensions = [input_dimension] + layer_dims

        self.dimensions = dimensions  # save for inspection

        # Build Encoder
        encoder_layers = []
        for i in range(len(dimensions) - 1):
            encoder_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i != len(dimensions) - 2:  # No activation after last encoder layer
                encoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder (mirror of encoder, excluding final layer's activation)
        decoder_layers = []
        reversed_dimensions = list(reversed(dimensions))
        for i in range(len(reversed_dimensions) - 1):
            decoder_layers.append(nn.Linear(reversed_dimensions[i], reversed_dimensions[i + 1]))
            if i != len(reversed_dimensions) - 2:
                decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        all weights are initialized with values sampled from uniform distributions with the Xavier initialization
        and the biases are set to 0, as described in the paper by Delong et al. (2023)
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def save(self, name):
        torch.save(self.encoder.state_dict(), name + "_encoder.pt")
        torch.save(self.decoder.state_dict(), name + '_decoder.pt')

    def load(self, name):
        if os.path.isfile(name + '_encoder.pt') and os.path.isfile(name + '_decoder.pt'):
            self.encoder.load_state_dict(torch.load(name + '_encoder.pt'))
            self.decoder.load_state_dict(torch.load(name + '_decoder.pt'))
            self.encoder.eval()
            self.decoder.eval()
            return True
        else:
            return False

    def forward(self, x, feature_value_indices):
        y = self.encoder(x)
        y = self.decoder(y)

        # Split the tensor into chunks based on the ranges
        chunks = [y[:, range.start:range.stop] for range in feature_value_indices]

        # Apply softmax to each chunk
        softmax_chunks = [F.softmax(chunk, dim=1) for chunk in chunks]

        # Concatenate the chunks back together
        y = torch.cat(softmax_chunks, dim=1)

        return y


def train(transactions: pd.DataFrame, autoencoder: AutoEncoder = None, noise_factor=0.5, lr=5e-3, epochs=2,
          batch_size=None, loss_function=torch.nn.BCELoss(), num_workers=1, layer_dims: list = None, device=None,
          patience: int = 10, delta: float = 1e-3, show_progress: bool = True):
    """
    Train an autoencoder for association rule mining.

    :param transactions: pandas DataFrame containing the tabular data
    :param autoencoder: optional pre-initialized AutoEncoder (if None, one is created automatically)
    :param noise_factor: noise factor for denoising autoencoder (default=0.5)
    :param lr: learning rate (default=5e-3)
    :param epochs: number of training epochs (default=2). Shorter training produces higher-quality rules.
    :param batch_size: batch size for training. If None (default), automatically determined based on
        dataset size: 2 for <100 rows, 8 for <500, 16 for <1000, 32 for <5000, 64 for larger datasets.
    :param loss_function: loss function to use (default=BCELoss)
    :param num_workers: number of parallel workers for data preparation (default=1)
    :param layer_dims: optional list of hidden layer dimensions
    :param device: device to train on ('cuda', 'cpu', or None for auto-detect)
    :param patience: early stopping patience - epochs to wait for improvement (default=10)
    :param delta: early stopping delta - minimum improvement threshold (default=1e-3)
    :param show_progress: if True (default), show a progress bar during training.
    :return: trained AutoEncoder, or None if training failed
    """
    # Auto batch_size based on dataset size
    n_samples = len(transactions)
    if batch_size is None:
        batch_size = next(
            (bs for threshold, bs in [(200, 2), (500, 4), (1000, 8), (5000, 32)] if n_samples < threshold), 64)
        logger.info(f"Auto batch_size: {batch_size} (based on {n_samples} samples)")
    input_vectors, feature_value_indices = _one_hot_encoding_with_feature_tracking(transactions, num_workers)

    if input_vectors is None:
        logger.error("Training stopped. Please fix the data issues first.")
        return None

    columns = input_vectors.columns.tolist()

    if not autoencoder:
        autoencoder = AutoEncoder(input_dimension=len(columns), feature_count=len(feature_value_indices),
                                  layer_dims=layer_dims)
    device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")
    autoencoder = autoencoder.to(device)
    autoencoder.train()
    autoencoder.input_vectors = input_vectors

    input_vectors = input_vectors.to_numpy(dtype=np.float32, copy=True)

    autoencoder.feature_value_indices = feature_value_indices
    autoencoder.feature_values = columns

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=2e-8)

    vectors_tensor = torch.from_numpy(input_vectors)

    dataset = TensorDataset(vectors_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    softmax_ranges = [range(cat['start'], cat['end']) for cat in feature_value_indices]

    best_loss = float("inf")
    patience_counter = 0
    total_batches = len(dataloader)

    # Format to match logging style: timestamp - aerial - INFO - Training: progress
    bar_fmt = "\033[94m{desc} - aerial - INFO - Training: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs | loss: {postfix}\033[0m"
    epoch_iter = tqdm(range(epochs), desc="", disable=not show_progress, bar_format=bar_fmt,
                      leave=True, position=0)

    for epoch in epoch_iter:
        epoch_iter.set_description_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3])

        epoch_loss = 0.0
        for batch, in dataloader:
            batch = batch.to(device, non_blocking=True)
            noisy_batch = (batch + torch.randn_like(batch) * noise_factor).clamp(0, 1)
            reconstructed_batch = autoencoder(noisy_batch, softmax_ranges)
            total_loss = loss_function(reconstructed_batch, batch)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        epoch_loss /= total_batches
        epoch_iter.set_postfix_str(f"{epoch_loss:.4f}")

        if epoch_loss < best_loss - delta:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iter.close()
                logger.debug(f"Early stopping triggered at epoch {epoch + 1}")
                break

    epoch_iter.close()
    logger.debug("Training completed.")
    return autoencoder
