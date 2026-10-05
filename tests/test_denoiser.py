"""
Unit Tests for Speech Denoising Autoencoder & DSP Processor.
"""

import pytest
import torch

from src.dsp.stft_processor import STFTProcessor, calculate_snr, add_acoustic_noise
from src.models import SpeechDenoisingAutoencoder
from src.data.speech_dataset import generate_synthetic_speech


def test_stft_istft_reconstruction():
    processor = STFTProcessor(sample_rate=16000)
    waveform = generate_synthetic_speech(duration=1.0, sample_rate=16000)

    # Forward STFT
    mag, phase = processor.forward(waveform)
    assert mag.size(0) == (512 // 2) + 1  # 257 bins
    assert phase.shape == mag.shape

    # Inverse STFT
    reconstructed = processor.inverse(mag, phase, length=len(waveform))
    assert reconstructed.shape == waveform.shape

    # Check reconstruction fidelity
    reconstruction_snr = calculate_snr(waveform, reconstructed)
    assert reconstruction_snr > 30.0  # Perfect/near-perfect STFT inversion


def test_snr_calculation_and_noise_addition():
    clean = torch.randn(16000)
    target_snr = 6.0
    noisy = add_acoustic_noise(clean, target_snr_db=target_snr)

    measured_snr = calculate_snr(clean, noisy)
    assert abs(measured_snr - target_snr) < 1.0  # Within 1 dB tolerance


def test_autoencoder_forward_and_mask_bounds():
    model = SpeechDenoisingAutoencoder()
    model.eval()

    dummy_mag = torch.rand(2, 1, 257, 101)
    with torch.no_grad():
        out_mag = model(dummy_mag)

    assert out_mag.shape == dummy_mag.shape
    # Denoised magnitude must be non-negative and suppressed (<= noisy)
    assert torch.all(out_mag >= 0.0)
    assert torch.all(out_mag <= dummy_mag + 1e-4)


def test_speech_generator():
    speech = generate_synthetic_speech(duration=1.0, sample_rate=16000)
    assert speech.shape == (16000,)
    assert not torch.isnan(speech).any()
    assert torch.max(torch.abs(speech)) > 0.1
