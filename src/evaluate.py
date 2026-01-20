import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataset import AudioDataset
from model import AudioCNN
from utils import ensure_dir


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir("outputs/plots")

    dataset = AudioDataset("dataset")
    loader = DataLoader(dataset, batch_size=16)

    model = AudioCNN().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pt"))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(y.tolist())

    print(classification_report(y_true, y_pred, target_names=["speech", "music"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks([0, 1], ["speech", "music"])
    plt.yticks([0, 1], ["speech", "music"])
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
