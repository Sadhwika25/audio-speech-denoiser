"""
Synthetic Speech & Acoustic Noise Dataset Generator for Denoising Training.
Generates paired clean speech signals and realistic noisy mixtures at variable SNR.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..dsp.stft_processor import STFTProcessor, add_acoustic_noise


def generate_synthetic_speech(duration: float = 1.0, sample_rate: int = 16000) -> torch.Tensor:
    """
    Synthesizes speech-like acoustic signals with authentic pitch (F0)
    and vowel formant resonances (F1, F2, F3).
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # Fundamental pitch frequency (F0: ~120 Hz to 220 Hz)
    f0 = np.random.uniform(120, 220)
    
    # Formants (vowel resonances)
    f1 = np.random.uniform(350, 750)
    f2 = np.random.uniform(1200, 2200)
    f3 = np.random.uniform(2500, 3200)
    
    # Temporal envelope (syllable bursts and pauses)
    num_syllables = np.random.randint(2, 4)
    envelope = np.zeros(n_samples)
    for _ in range(num_syllables):
        center = np.random.uniform(0.15, 0.85)
        width = np.random.uniform(0.06, 0.14)
        envelope += np.exp(-((t - center) ** 2) / (2 * width ** 2))
    
    envelope = np.clip(envelope, 0, 1)

    # Glottal pulse train excitation
    pulse_train = np.sin(2 * np.pi * f0 * t)
    for harmonic in range(2, 8):
        pulse_train += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)

    # Formant filtering
    speech = (
        np.sin(2 * np.pi * f1 * t)
        + 0.6 * np.sin(2 * np.pi * f2 * t)
        + 0.3 * np.sin(2 * np.pi * f3 * t)
    ) * pulse_train * envelope

    # Normalize amplitude
    peak = np.max(np.abs(speech)) + 1e-8
    speech = 0.9 * (speech / peak)
    
    return torch.from_numpy(speech.astype(np.float32))


class SpeechDenoisingDataset(Dataset):
    """
    Dataset of (noisy_spectrogram, clean_spectrogram) pairs.
    Pre-cached in memory for high-speed training.
    """

    def __init__(
        self,
        num_samples: int = 500,
        sample_rate: int = 16000,
        duration: float = 1.0,
        snr_range: tuple = (0.0, 10.0),
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.snr_range = snr_range
        self.processor = STFTProcessor(sample_rate=sample_rate)

        # Pre-cache clean and noisy pairs
        self.data_pairs = []
        for _ in range(num_samples):
            clean_wf = generate_synthetic_speech(duration=duration, sample_rate=sample_rate)
            snr_db = np.random.uniform(*snr_range)
            noise_type = np.random.choice(["gaussian", "hum"])
            noisy_wf = add_acoustic_noise(clean_wf, target_snr_db=snr_db, noise_type=noise_type)

            with torch.no_grad():
                clean_mag, _ = self.processor.forward(clean_wf)
                noisy_mag, _ = self.processor.forward(noisy_wf)

            # Shape: (1, Freq, Time)
            self.data_pairs.append((noisy_mag.unsqueeze(0), clean_mag.unsqueeze(0)))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.data_pairs[idx]


def get_speech_dataloaders(
    train_samples: int = 400,
    val_samples: int = 100,
    batch_size: int = 16,
):
    train_dataset = SpeechDenoisingDataset(num_samples=train_samples, snr_range=(0.0, 10.0))
    val_dataset = SpeechDenoisingDataset(num_samples=val_samples, snr_range=(2.0, 8.0))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
