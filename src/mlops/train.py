import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

import wandb
from mlops.data import corrupt_mnist
from mlops.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/model.pth"
STATS_PATH = "reports/figures"


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg) -> None:
    """Function to train the model."""
    print("Training day and night")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(project="corrupt_mnist", entity="rokojo-danmarks-tekniske-universitet-dtu", config=cfg_dict)

    # Load model config and create model
    model = Model(cfg).to(DEVICE)

    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.train.batch_size)

    loss_fn = hydra.utils.instantiate(cfg.train.loss_fn)
    optimizer = hydra.utils.instantiate(cfg.optimizer.optimizer, params=model.parameters())

    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(cfg.train.epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = [
                    wandb.Image(img[j].detach().cpu(), caption=f"Input image {j}") for j in range(min(5, len(img)))
                ]
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads)})

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot.numpy(),
                preds[:, class_id].numpy(),
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )

        # alternatively use wandb.log({"roc": wandb.Image(plt)}
        wandb.log({"roc": wandb.Image(plt.gcf())})
        plt.close()  # close the plot to avoid memory leaks and overlapping figures

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)
    run.link_artifact(artifact=artifact, target_path="Corrupt_mnist_models/models", aliases=["latest"])
    wandb.finish()


if __name__ == "__main__":
    train()
