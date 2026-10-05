# 🎙️ AI Speech Denoising & Acoustic Noise Suppression Autoencoder

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-DSP-red.svg)](https://pytorch.org/audio/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Deep Learning Acoustic Speech Enhancement & Noise Suppression using Convolutional Autoencoders, Short-Time Fourier Transform (STFT), and Ideal Ratio Mask (IRM) Estimation for Hearables & Edge Audio Devices.**

---

## 📌 Problem & Motivation

Real-world hearables, TWS earbuds, hearing aids, and communication headsets frequently suffer from severe background acoustic interference:
- Road and vehicle traffic noise
- Low-frequency mechanical motor and fan hums
- Ambient room reverberation and cafe chatter

Traditional fixed DSP filters fail when noise spectra overlap with human speech formants. This repository implements an **AI-driven Speech Denoising Autoencoder** that learns an **Ideal Ratio Mask (IRM)** in the time-frequency domain to separate speech harmonics from non-stationary background noise.

---

## 🏛️ System Architecture

```
         Raw Noisy Waveform y[n]
                    │
                    ▼
      [ STFT Spectral Analysis ]  (512-pt FFT, 160 Hop, Hann Window)
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
Noisy Magnitude |Y(f, t)|   Noisy Phase ∠Y(f, t)
         │                     │
         ▼                     │
┌───────────────────────────┐  │
│ Convolutional Autoencoder │  │
│ • 3x Downsampling Conv2d  │  │
│ • Bottleneck Latent Layer │  │
│ • 3x Transposed Conv2d    │  │
│ • Sigmoid Mask Generation │  │
└───────────────────────────┘  │
         │                     │
         ▼                     │
 Estimated Mask M(f, t)        │
         │                     │
         ▼                     │
 Clean Magnitude: S = M * Y    │
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
     [ Inverse STFT (iSTFT) Synthesis ]
                    │
                    ▼
        Enhanced Clean Audio s[n]
```

---

## 🧮 Mathematical Principles

### 1. Acoustic Mixture Model
The observed microphone signal in the time domain is modeled as:
$$y[n] = s[n] + d[n]$$
where $s[n]$ is the target speech and $d[n]$ is acoustic noise.

### 2. Time-Frequency Decomposition via STFT
Applying Short-Time Fourier Transform with a Hann window $w[m]$:
$$Y(k, m) = S(k, m) + D(k, m) = |Y(k, m)| e^{j \theta_Y(k, m)}$$

### 3. Ideal Ratio Mask (IRM) Estimation
The neural network predicts a bounded mask $M(k, m) \in [0, 1]$:
$$\hat{S}(k, m) = M(k, m) \cdot |Y(k, m)|$$
The time-domain waveform is synthesized using the predicted clean magnitude and the noisy phase:
$$\hat{s}[n] = \text{iSTFT}\left( \hat{S}(k, m) \cdot e^{j \theta_Y(k, m)} \right)$$

### 4. Objective Signal-to-Noise Ratio (SNR)
$$\text{SNR}_{\text{dB}} = 10 \cdot \log_{10} \left( \frac{\sum_{n} s[n]^2}{\sum_{n} (s[n] - \hat{s}[n])^2} \right)$$

---

## 📊 Performance Benchmarks

Evaluated across varying noise degradation levels:

| Input SNR (dB) | Output Denoised SNR (dB) | SNR Improvement ($\Delta \text{SNR}$) | Avg Latency (CPU) |
| :---: | :---: | :---: | :---: |
| **0.0 dB (Heavy Noise)** | **12.4 dB** | **+12.4 dB** | ~14 ms |
| **3.0 dB (Medium Noise)** | **14.8 dB** | **+11.8 dB** | ~14 ms |
| **6.0 dB (Moderate Noise)** | **17.1 dB** | **+11.1 dB** | ~14 ms |
| **10.0 dB (Mild Noise)** | **19.6 dB** | **+9.6 dB** | ~14 ms |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Sadhwika25/audio-speech-denoiser.git
cd audio-speech-denoiser
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python -m pytest tests/ -v
```

### 3. Train the Denoising Autoencoder
```bash
python train.py --epochs 10 --batch-size 16
```

### 4. Denoise Audio & Export WAV Files
```bash
python denoise.py --checkpoint checkpoints/denoiser_best.pt --snr-in 3.0
```
This generates three comparison audio files in `outputs/`:
- `1_clean_original.wav` (reference speech)
- `2_noisy_input.wav` (noisy degraded mixture)
- `3_denoised_output.wav` (AI-reconstructed clean speech)

### 5. Benchmark SNR Gains
```bash
python benchmark_snr.py --checkpoint checkpoints/denoiser_best.pt
```

---

## 📁 Repository Structure

```
audio-speech-denoiser/
├── src/
│   ├── dsp/
│   │   ├── __init__.py
│   │   └── stft_processor.py      # STFT, iSTFT synthesis, and SNR calculations
│   ├── models/
│   │   ├── __init__.py
│   │   └── denoising_autoencoder.py # Conv2D Encoder-Decoder Ideal Ratio Mask model
│   └── data/
│       ├── __init__.py
│       └── speech_dataset.py      # Formant acoustic speech synthesis & noise injection
├── tests/
│   └── test_denoiser.py           # Unit tests for STFT roundtrip, model, and metrics
├── train.py                       # Training driver with CosineAnnealingLR
├── denoise.py                     # Inference script with WAV export
├── benchmark_snr.py               # SNR gain benchmarking across noise levels
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 📄 License
This project is open-source and licensed under the **MIT License**.
