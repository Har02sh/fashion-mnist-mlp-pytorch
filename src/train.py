import torch
from model import FashionModel


def accuracy_fn(y_pred, y_target):
    y_pred = torch.argmax(y_pred, dim=1)
    return (y_pred == y_target).float().mean()


def validate_model(model, val_batch, device, loss_fn):
    model.eval()
    val_accuracy, val_loss = 0, 0
    with torch.inference_mode():
        for X, y in val_batch:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            val_accuracy += accuracy_fn(y_pred, y).item()
            val_loss += loss.item()

    val_accuracy /= len(val_batch)
    val_loss /= len(val_batch)
    return val_loss, val_accuracy


def train_model(train_batch, val_batch, device):
    model = FashionModel().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    EPOCH = 20
    best_val_loss = float('inf')
    patience, patience_counter = 3, 0

    loss_list_train, loss_list_val = [], []
    accuracy_list_train, accuracy_list_val = [], []

    for epoch in range(EPOCH):
        model.train()
        train_accuracy, train_loss = 0, 0

        for img_batch, label_batch in train_batch:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            y_pred = model(img_batch)
            loss = loss_fn(y_pred, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_accuracy += accuracy_fn(y_pred, label_batch).item()
            train_loss += loss.item()

        train_accuracy /= len(train_batch)
        train_loss /= len(train_batch)

        loss_list_train.append(train_loss)
        accuracy_list_train.append(train_accuracy)

        val_loss, val_accuracy = validate_model(model, val_batch, device, loss_fn)
        loss_list_val.append(val_loss)
        accuracy_list_val.append(val_accuracy)

        print(f"Epoch:{epoch} | Train Loss:{train_loss:.2f} | Train Acc:{train_accuracy:.2f} | Val Loss:{val_loss:.2f} | Val Acc:{val_accuracy:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model/fashion_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early Stopping Triggered.")
            break

    return loss_list_train, loss_list_val, accuracy_list_train, accuracy_list_val
