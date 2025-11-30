import torch
from model import FashionModel
from train import accuracy_fn


def evaluate_model(test_batch, device):
    model = FashionModel()
    state_dict = torch.load("model/fashion_model.pth")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss, test_accuracy = 0, 0

    with torch.inference_mode():
        for img, label in test_batch:
            img, label = img.to(device), label.to(device)
            y_pred = model(img)
            loss = loss_fn(y_pred, label)

            test_accuracy += accuracy_fn(y_pred, label).item()
            test_loss += loss.item()

    test_accuracy /= len(test_batch)
    test_loss /= len(test_batch)

    print(f"Test Accuracy: {test_accuracy:.2f} | Test Loss: {test_loss:.2f}")

    return test_loss, test_accuracy
