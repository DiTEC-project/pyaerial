import unittest
import torch
import os
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
        self.feature_value_indices = [(0, 5), (5, 10), (10, 15), (15, 20)]

    def test_forward_pass_output_shape(self):
        output = self.model(self.input_data, self.feature_value_indices)
        self.assertEqual(output.shape, self.input_data.shape, "Output shape should match input shape.")

    def test_softmax_chunks_sum_to_one(self):
        output = self.model(self.input_data, self.feature_value_indices)

        for start, end in self.feature_value_indices:
            chunk = output[:, start:end]
            sums = chunk.sum(dim=1)
            # Because of softmax, each chunk's sum over its features should be close to 1
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4),
                            f"Softmax chunk from {start} to {end} does not sum to 1.")

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


if __name__ == "__main__":
    unittest.main()
