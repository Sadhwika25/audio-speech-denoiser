"""
DSP & STFT Audio Processing Engine.
Handles Forward STFT (Magnitude & Phase decomposition),
Inverse STFT (Waveform Synthesis), and SNR Metrics.
"""

import math
import torch
import numpy as np


class STFTProcessor:
    """
    Handles STFT spectral decomposition and Inverse STFT waveform reconstruction.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        sample_rate: int = 16000,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate

    def forward(self, waveform: torch.Tensor):
        """
        Decomposes 1D waveform into Magnitude and Phase spectrograms.
        Args:
            waveform: Tensor of shape (..., time_samples)
        Returns:
            magnitude: Tensor of shape (..., n_fft // 2 + 1, frames)
            phase: Tensor of shape (..., n_fft // 2 + 1, frames)
        """
        device = waveform.device
        window = torch.hann_window(self.win_length, device=device)
        
        stft_res = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        
        magnitude = torch.abs(stft_res)
        phase = torch.angle(stft_res)
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length: int = None) -> torch.Tensor:
        """
        Reconstructs time-domain 1D waveform using magnitude and phase via iSTFT.
        """
        device = magnitude.device
        window = torch.hann_window(self.win_length, device=device)
        
        complex_spec = torch.polar(magnitude, phase)
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=length,
        )
        return waveform


def calculate_snr(clean: torch.Tensor, degraded: torch.Tensor) -> float:
    """
    Calculates Signal-to-Noise Ratio (SNR) in decibels (dB):
    SNR = 10 * log10( P_signal / P_noise )
    """
    signal_power = torch.mean(clean ** 2) + 1e-12
    noise_power = torch.mean((clean - degraded) ** 2) + 1e-12
    snr_db = 10.0 * torch.log10(signal_power / noise_power)
    return float(snr_db.item())


def add_acoustic_noise(
    clean_audio: torch.Tensor,
    target_snr_db: float = 5.0,
    noise_type: str = "gaussian",
) -> torch.Tensor:
    """
    Mixes background noise into clean audio at a precise target SNR (dB).
    """
    signal_power = torch.mean(clean_audio ** 2) + 1e-12
    
    if noise_type == "gaussian":
        noise = torch.randn_like(clean_audio)
    elif noise_type == "hum":
        # Low frequency hum (120 Hz + 240 Hz harmonics)
        t = torch.linspace(0, len(clean_audio) / 16000.0, len(clean_audio))
        noise = torch.sin(2 * math.pi * 120 * t) + 0.5 * torch.sin(2 * math.pi * 240 * t)
        noise = noise.to(clean_audio.device)
    else:
        noise = torch.randn_like(clean_audio)

    noise_power = torch.mean(noise ** 2) + 1e-12
    desired_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
    scaled_noise = noise * torch.sqrt(desired_noise_power / noise_power)
    
    return clean_audio + scaled_noise
