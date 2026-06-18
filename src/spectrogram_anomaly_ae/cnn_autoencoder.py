"""PyTorch CNN autoencoder used by the spectrogram anomaly notebooks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_INPUT_SHAPE = (100, 150, 3)
DEFAULT_IMAGE_SIZE = (150, 100)


class CNNAutoencoder(nn.Module):
    """Convolutional autoencoder for 100 x 150 RGB spectrogram images."""

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), ceil_mode=True),
            nn.Conv2d(8, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25 * 25 * 12, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 25 * 25 * 12),
            nn.ReLU(),
            nn.Unflatten(1, (12, 25, 25)),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 3), mode="nearest"),
            nn.Conv2d(12, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | torch.device | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def images_to_tensor(images: np.ndarray, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert NHWC float image arrays to NCHW tensors."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected images with shape (N, H, W, 3), got {images.shape}")
    tensor = torch.as_tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_images(tensor: torch.Tensor) -> np.ndarray:
    """Convert NCHW tensors back to NHWC NumPy arrays."""
    return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()


def reconstruct_images(
    model: nn.Module,
    images: np.ndarray,
    batch_size: int = 32,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Run batched inference and return reconstructions as NHWC NumPy arrays."""
    device = get_device(device)
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(images_to_tensor(images))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    reconstructions = []
    with torch.inference_mode():
        for (batch,) in loader:
            output = model(batch.to(device))
            reconstructions.append(tensor_to_images(output))
    return np.concatenate(reconstructions, axis=0)


def train_autoencoder(
    model: nn.Module,
    train_images: np.ndarray,
    validation_images: np.ndarray,
    *,
    epochs: int = 4000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str | torch.device | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> dict[str, list[float]]:
    """Train an autoencoder on NHWC NumPy image arrays."""
    device = get_device(device)
    if seed is not None:
        set_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(images_to_tensor(train_images))
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_tensor = images_to_tensor(validation_images, device)

    history = {"loss": [], "val_loss": []}
    epoch_iterable = range(epochs)
    if progress:
        from tqdm.auto import tqdm

        epoch_iterable = tqdm(epoch_iterable, desc="Training CNN-AE")

    for _ in epoch_iterable:
        model.train()
        batch_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.inference_mode():
            validation_loss = criterion(model(validation_tensor), validation_tensor).item()

        history["loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(float(validation_loss))

    return history


def save_autoencoder(
    model: CNNAutoencoder,
    path: str | Path,
    *,
    history: dict[str, list[float]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": model.latent_dim,
            "input_shape": DEFAULT_INPUT_SHAPE,
            "history": history or {},
        },
        path,
    )


def load_autoencoder(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> CNNAutoencoder:
    device = get_device(device)
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location=device)
    latent_dim = int(checkpoint.get("latent_dim", 16))
    model = CNNAutoencoder(latent_dim=latent_dim)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
