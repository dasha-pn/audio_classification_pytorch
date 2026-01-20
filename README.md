# Audio Classification: Speech vs Music (PyTorch)

This project implements a clean, research-style pipeline for **audio classification** using PyTorch.  
The task is binary classification of audio signals into **speech** or **music**, based on log-mel spectrograms.

The codebase is structured with clear separation of responsibilities between training, evaluation, and visualization, making it suitable for experimentation and further research.

---

## Model

The model is a lightweight **Convolutional Neural Network (CNN)** designed for time-frequency audio representations:

- Input: log-mel spectrogram
- Architecture: stacked convolutional layers + pooling
- Output: class logits (`speech`, `music`)
- Loss: Cross-Entropy
- Optimizer: Adam

The architecture is intentionally simple and serves as a strong baseline for further experimentation (e.g. CRNN, transformers, or self-supervised features).

---

## Training

Training includes:
- Reproducible random seed
- Train/validation split
- Accuracy and loss tracking
- Best model checkpointing based on validation loss

Run training from the project root:

```bash
python src/train.py
```

Outputs:

- `outputs/checkpoints/best_model.pt`

- `outputs/history.json`

## Visualization

Training curves are generated **after training**, based on saved logs.  
This design keeps the training code clean and enables convenient post-hoc analysis.

Run the visualization script:

```bash
python src/visualize.py
```

Generated Plots

- `outputs/plots/loss.png` — training and validation loss curves

- `outputs/plots/accuracy.png` — training and validation accuracy curves

# Evaluation

Final evaluation is performed using the **best saved model checkpoint**.

## Run Evaluation

```bash
python src/evaluate.py
```

## Evaluation Metrics

- Precision

- Recall

- F1-score

- Confusion matrix

## Outputs

- Metrics printed to the console

- Confusion matrix plot saved to:

```bash
outputs/plots/confusion_matrix.png
```

