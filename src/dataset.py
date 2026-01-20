import os
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, root_dir, sr=22050, n_mels=64, max_len=3):
        self.samples = []
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len

        for label, cls in enumerate(["speech", "music"]):
            class_dir = os.path.join(root_dir, cls)
            for f in os.listdir(class_dir):
                if f.endswith(".wav"):
                    self.samples.append(
                        (os.path.join(class_dir, f), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, sr = librosa.load(path, sr=self.sr, mono=True)

        max_samples = self.sr * self.max_len
        if len(y) > max_samples:
            y = y[:max_samples]
        else:
            y = np.pad(y, (0, max_samples - len(y)))

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = torch.tensor(mel_db).unsqueeze(0)  #[1, n_mels, T]
        label = torch.tensor(label)

        return mel_db.float(), label
