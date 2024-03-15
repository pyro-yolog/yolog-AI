from Models import VAEModel
from config import TrainConfig
import torch
from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig
from Models import *
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np
weights = "./VAEModel.pt"

model = VAEModel(TrainConfig)
model.load_state_dict(torch.load(weights))

new_spectrogram = model.decoder(torch.zeros((1, 16, 3, 3), dtype=torch.float)).detach()
print(new_spectrogram, new_spectrogram.shape)

new_spectrogram_resized = F.interpolate(new_spectrogram, size=(80, 80), mode='bilinear', align_corners=False)

config = SpeechT5HifiGanConfig(sampling_rate=44100)
model = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", config=config)
wav = model(new_spectrogram_resized.squeeze(1))


waveform_np = wav.detach().numpy()

plt.figure(figsize=(10, 4))
plt.plot(waveform_np.squeeze())
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()


torchaudio.save("output.wav", wav.cpu().detach(), 22050)
