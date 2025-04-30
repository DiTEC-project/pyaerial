"""
Copyright (c) [2025] [Erkan Karabulut - DiTEC Project]

Construct an Autoencoder for association rule mining as described in the paper (Neurosymbolic association rule mining
from tabular data - https://arxiv.org/abs/2504.19354)
"""

import os
import logging
import torch
import pandas as pd
from torch import nn
import math
import torch.nn.functional as F

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
            # Compute default number of layers based on log base 16
            layer_count = max(1, math.ceil(math.log(input_dimension, 16)) - 1)

            # Calculate dimensions with consistent reduction ratio
            reduction_ratio = (feature_count / input_dimension) ** (1 / (layer_count))
            dimensions = [input_dimension]
            for i in range(1, layer_count):
                next_dim = max(feature_count, int(dimensions[-1] * reduction_ratio))
                dimensions.append(next_dim)
            dimensions.append(feature_count)
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
        chunks = [y[:, start:end] for start, end in feature_value_indices]

        # Apply softmax to each chunk
        softmax_chunks = [F.softmax(chunk, dim=1) for chunk in chunks]

        # Concatenate the chunks back together
        y = torch.cat(softmax_chunks, dim=1)

        return y


def train(transactions: pd.DataFrame, autoencoder: AutoEncoder = None, noise_factor=0.5,
          lr=5e-3, epochs=1, batch_size=2, loss_function=torch.nn.BCELoss(), num_workers=1, layer_dims: list = None):
    """
    train an autoencoder for association rule mining
    """
    input_vectors, feature_value_indices = _one_hot_encoding_with_feature_tracking(transactions, num_workers)
    columns = input_vectors.columns.tolist()

    if not autoencoder:
        autoencoder = AutoEncoder(input_dimension=len(columns), feature_count=len(feature_value_indices),
                                  layer_dims=layer_dims)

    autoencoder.input_vectors = input_vectors

    input_vectors = input_vectors.to_numpy()

    autoencoder = torch.compile(autoencoder)
    autoencoder.feature_value_indices = feature_value_indices
    autoencoder.feature_values = columns

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=2e-8)

    vectors_tensor = torch.tensor(input_vectors, dtype=torch.float32)
    dataset = TensorDataset(vectors_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    softmax_ranges = [(cat['start'], cat['end']) for cat in feature_value_indices]

    for epoch in range(epochs):
        # print(f"Epoch {epoch + 1}/{epochs}")
        for batch_index, (batch,) in enumerate(dataloader):
            noisy_batch = (batch + torch.randn_like(batch) * noise_factor).clamp(0, 1)

            # Forward pass
            reconstructed_batch = autoencoder(noisy_batch, softmax_ranges)

            # Compute loss for the entire batch
            total_loss = sum(
                loss_function(
                    reconstructed_batch[:, start:end],
                    batch[:, start:end]
                )
                for (start, end) in softmax_ranges
            )

            # Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return autoencoder
