from torch.utils.data import DataLoader, Dataset
import os
import torch
import numpy as np
import librosa
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
class WAVDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_rate=44100, win_length=512, hop_length=512, n_fft=2048):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mp3_path = os.path.join(self.root_dir, self.file_list[idx])

        mel_file = mp3_path.replace('.wav', '_mel.npy')
        if os.path.exists(mel_file):
            mel_spec = np.load(mel_file, allow_pickle=True)
        else:
            y, sr = librosa.load(mp3_path, sr=self.sample_rate)
            target_samples = int(30 * self.sample_rate)
            if len(y) >= target_samples:
                y = y[:target_samples]
            else:
                pad_length = target_samples - len(y)
                y = np.pad(y, (0, pad_length), 'constant')
            D = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length))
            mel_spec = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=128, hop_length=self.hop_length, win_length=self.win_length)
            np.save(mel_file, mel_spec)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, 0
def DataLoad(config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )
    dataset = WAVDataset(root_dir=config.dataset_dir, transform=transform)
    batch_size = config.train_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing WAV files")):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        pass
    return dataloader
