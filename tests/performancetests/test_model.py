import os
import time

import torch
from dotenv import load_dotenv
from mlops.model import Model
from omegaconf import OmegaConf

import wandb

# Load environment variables from .env file
# load_dotenv()


def load_model(artifact_name):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(name=artifact_name)
    artifact.download(root="artifact_dir")

    file_name = artifact.files()[0].name
    ckpt = torch.load(f"artifact_dir/{file_name}")
    config = OmegaConf.create(ckpt["config"])

    model = Model(config)
    model.load_state_dict(ckpt["state_dict"])

    return model


def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1
