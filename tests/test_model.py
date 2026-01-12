from dataclasses import dataclass

import pytest
import torch
from mlops.model import Model


@dataclass
class ConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int


@dataclass
class FCConfig:
    in_features: int
    out_features: int


@dataclass
class PoolConfig:
    kernel_size: int
    stride: int


@dataclass
class ModelConfig:
    conv1: ConvConfig
    conv2: ConvConfig
    conv3: ConvConfig
    fc1: FCConfig
    dropout: float
    pool: PoolConfig


cfg = ModelConfig(
    conv1=ConvConfig(1, 32, 3, 1),
    conv2=ConvConfig(32, 64, 3, 1),
    conv3=ConvConfig(64, 128, 3, 1),
    fc1=FCConfig(128, 10),
    dropout=0.5,
    pool=PoolConfig(2, 2),
)


def test_model_forward():
    model = Model(cfg)
    x = torch.randn(1, 1, 28, 28)
    output = model(x)

    assert output.shape == (1, 10)


def test_error_on_wrong_shape():
    model = Model(cfg)
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    with pytest.raises(ValueError, match=r"Expected each sample to have shape \[1, 28, 28\]"):
        model(torch.randn(1, 1, 28, 29))


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = Model(cfg)
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)
