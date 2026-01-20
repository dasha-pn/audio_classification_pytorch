import json
import matplotlib.pyplot as plt
from utils import ensure_dir


def main():
    ensure_dir("outputs/plots")

    with open("outputs/history.json") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # LOSS
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("outputs/plots/loss.png")
    plt.close()

    # ACCURACY
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("outputs/plots/accuracy.png")
    plt.close()


if __name__ == "__main__":
    main()
