import torch
from src.dataset import load_datasets
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_loss, plot_accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    train_batch, val_batch, test_batch = load_datasets()

    loss_train, loss_val, acc_train, acc_val = train_model(train_batch, val_batch, device)

    evaluate_model(test_batch, device)

    plot_loss(loss_train, loss_val)
    plot_accuracy(acc_train, acc_val)


if __name__ == "__main__":
    main()
