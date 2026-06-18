from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectrogram_anomaly_ae.cnn_autoencoder import (
    CNNAutoencoder,
    count_parameters,
    load_autoencoder,
    reconstruct_images,
    save_autoencoder,
)


class CNNAutoencoderTests(unittest.TestCase):
    def test_reconstruction_preserves_nhwc_image_shape(self) -> None:
        model = CNNAutoencoder(latent_dim=16)
        images = np.random.default_rng(42).random((2, 100, 150, 3), dtype=np.float32)

        reconstructions = reconstruct_images(model, images, batch_size=1, device="cpu")

        self.assertEqual(reconstructions.shape, images.shape)
        self.assertEqual(count_parameters(model), 251_383)
        self.assertTrue(np.all((reconstructions >= 0.0) & (reconstructions <= 1.0)))

    def test_save_and_load_round_trip(self) -> None:
        model = CNNAutoencoder(latent_dim=16)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "ae.pt"
            save_autoencoder(model, checkpoint_path, history={"loss": [1.0]})
            loaded = load_autoencoder(checkpoint_path, device="cpu")

        images = np.random.default_rng(7).random((1, 100, 150, 3), dtype=np.float32)
        original_reconstruction = reconstruct_images(model, images, device="cpu")
        loaded_reconstruction = reconstruct_images(loaded, images, device="cpu")

        np.testing.assert_allclose(original_reconstruction, loaded_reconstruction)


if __name__ == "__main__":
    unittest.main()
