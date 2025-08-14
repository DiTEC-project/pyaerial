import unittest
import torch
import os

from torch import nn

from aerial.model import AutoEncoder


class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.input_dimension = 20
        self.feature_count = 5
        self.batch_size = 8

        self.model = AutoEncoder(input_dimension=self.input_dimension, feature_count=self.feature_count)
        self.input_data = torch.randn(self.batch_size, self.input_dimension)

        # feature_value_indices defines how softmax is applied
        # Here we fake 4 chunks roughly splitting 20 features
        self.feature_value_indices = [range(0, 5), range(5, 10), range(10, 15), range(15, 20)]

    def test_forward_pass_output_shape(self):
        output = self.model(self.input_data, self.feature_value_indices)
        self.assertEqual(output.shape, self.input_data.shape, "Output shape should match input shape.")

    def test_softmax_chunks_sum_to_one(self):
        output = self.model(self.input_data, self.feature_value_indices)

        for r in self.feature_value_indices:
            chunk = output[:, r.start:r.stop]  # use range's start and stop
            sums = chunk.sum(dim=1)
            # Because of softmax, each chunk's sum over its features should be close to 1
            self.assertTrue(
                torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
                f"Softmax chunk from {r.start} to {r.stop} does not sum to 1."
            )

    def test_encoder_decoder_shapes(self):
        encoded = self.model.encoder(self.input_data)
        self.assertEqual(encoded.shape[-1], self.feature_count, "Encoded feature size mismatch.")

        decoded = self.model.decoder(encoded)
        self.assertEqual(decoded.shape[-1], self.input_dimension, "Decoded feature size mismatch.")

    def test_save_and_load(self):
        save_path = "temp_model"
        self.model.save(save_path)

        new_model = AutoEncoder(input_dimension=self.input_dimension, feature_count=self.feature_count)
        loaded = new_model.load(save_path)

        self.assertTrue(loaded, "Model should load successfully.")

        # Cleanup temp files
        os.remove(save_path + "_encoder.pt")
        os.remove(save_path + "_decoder.pt")

    def test_invalid_load(self):
        new_model = AutoEncoder(input_dimension=self.input_dimension, feature_count=self.feature_count)
        loaded = new_model.load("non_existent_path")
        self.assertFalse(loaded, "Loading non-existent model should return False.")

    def test_forward_with_single_feature_chunk(self):
        single_chunk = [range(0, self.input_dimension)]
        output = self.model(self.input_data, single_chunk)
        sums = output.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_forward_with_minimal_input(self):
        model = AutoEncoder(input_dimension=2, feature_count=1)
        data = torch.randn(1, 2)
        output = model(data, [range(0, 2)])
        self.assertEqual(output.shape, (1, 2))

    def test_layer_dims_custom(self):
        layer_dims = [15, 10, 5]
        model = AutoEncoder(input_dimension=20, feature_count=5, layer_dims=layer_dims)
        self.assertEqual(model.dimensions, [20, 15, 10, 5])

    def test_weight_initialization(self):
        # All biases should be zero after init
        for m in self.model.encoder:
            if isinstance(m, nn.Linear):
                self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))
        for m in self.model.decoder:
            if isinstance(m, nn.Linear):
                self.assertTrue(torch.allclose(m.bias, torch.zeros_like(m.bias)))

    def test_symmetric_decoder_dimensions(self):
        enc_out_dim = [m.out_features for m in self.model.encoder if isinstance(m, nn.Linear)][-1]
        dec_in_dim = [m.in_features for m in self.model.decoder if isinstance(m, nn.Linear)][0]
        self.assertEqual(enc_out_dim, dec_in_dim)

    def test_forward_does_not_modify_input(self):
        input_clone = self.input_data.clone()
        _ = self.model(self.input_data, self.feature_value_indices)
        self.assertTrue(torch.allclose(self.input_data, input_clone))

    def test_forward_with_empty_feature_indices(self):
        with self.assertRaises(RuntimeError):
            _ = self.model(self.input_data, [])

    def test_deterministic_forward_with_fixed_seed(self):
        torch.manual_seed(0)
        model1 = AutoEncoder(self.input_dimension, self.feature_count)
        out1 = model1(self.input_data, self.feature_value_indices)
        torch.manual_seed(0)
        model2 = AutoEncoder(self.input_dimension, self.feature_count)
        out2 = model2(self.input_data, self.feature_value_indices)
        self.assertTrue(torch.allclose(out1, out2))

    def test_all_chunks_non_negative_and_sum_to_one(self):
        output = self.model(self.input_data, self.feature_value_indices)
        self.assertTrue(torch.all(output >= 0))
        for r in self.feature_value_indices:
            sums = output[:, r.start:r.stop].sum(dim=1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_small_batch_forward(self):
        small_batch = torch.randn(1, self.input_dimension)
        output = self.model(small_batch, self.feature_value_indices)
        self.assertEqual(output.shape, (1, self.input_dimension))

    def test_forward_on_large_batch(self):
        large_batch = torch.randn(1024, self.input_dimension)
        output = self.model(large_batch, self.feature_value_indices)
        self.assertEqual(output.shape, (1024, self.input_dimension))


if __name__ == "__main__":
    unittest.main()
