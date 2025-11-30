import matplotlib.pyplot as plt

def plot_loss(loss_list_train, loss_list_val):
    epoch = range(1, len(loss_list_train) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epoch, loss_list_train, label="train")
    plt.plot(epoch, loss_list_val, label="val")
    plt.title("Train vs Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(accuracy_list_train, accuracy_list_val):
    epoch = range(1, len(accuracy_list_train) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epoch, accuracy_list_train, label="train")
    plt.plot(epoch, accuracy_list_val, label="val")
    plt.title("Train vs Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
