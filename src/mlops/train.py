import torch
import matplotlib.pyplot as plt
import typer
from mlops.model import Model
from mlops.data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/model.pth"
STATS_PATH = "reports/figures"


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """
    Function to train the model.

        Parameters:
            lr (float): Learning rate of the optimizer
            batch_size (int): Size of each training batch
            epochs (int): Number of training epochs
    """
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = Model().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), MODEL_PATH)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"{STATS_PATH}/training_statistics.png")


if __name__ == "__main__":
    typer.run(train)
