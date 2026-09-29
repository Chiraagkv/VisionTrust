"""
assurance/model_arch.py — Shared model architecture for the CV Assurance Platform.

A lightweight CNN that trains quickly on 32×32 synthetic images.
"""
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """
    Small 4-layer CNN for 32×32 RGB images.
    ~300 K parameters — trains in minutes on CPU.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 16×16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 8×8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 4×4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_model(path: str, num_classes: int = 10, device: str = "cpu") -> TinyCNN:
    """Load a TinyCNN from a .pt checkpoint."""
    model = TinyCNN(num_classes=num_classes)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

