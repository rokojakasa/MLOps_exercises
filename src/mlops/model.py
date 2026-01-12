import hydra
import torch
from omegaconf import DictConfig
from torch import nn


class Model(nn.Module):
    """My awesome model."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(
            config.conv1.in_channels, config.conv1.out_channels, config.conv1.kernel_size, config.conv1.stride
        )
        self.conv2 = nn.Conv2d(
            config.conv2.in_channels, config.conv2.out_channels, config.conv2.kernel_size, config.conv2.stride
        )
        self.conv3 = nn.Conv2d(
            config.conv3.in_channels, config.conv3.out_channels, config.conv3.kernel_size, config.conv3.stride
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.fc1.in_features, config.fc1.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, self.config.pool.kernel_size, self.config.pool.stride)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, self.config.pool.kernel_size, self.config.pool.stride)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, self.config.pool.kernel_size, self.config.pool.stride)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


@hydra.main(version_base=None, config_path="../../conf", config_name="model_config")
def main(config: DictConfig) -> None:
    """Test the model."""
    model = Model(config)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
