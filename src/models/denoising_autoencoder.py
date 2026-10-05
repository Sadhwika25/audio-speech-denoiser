"""
Convolutional Denoising Autoencoder for Spectral Speech Enhancement.
Learns an Ideal Ratio Mask (IRM) or Clean Magnitude Estimator
to isolate human speech formants from acoustic background interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechDenoisingAutoencoder(nn.Module):
    """
    Convolutional Encoder-Decoder architecture for spectrogram speech enhancement.
    Input: (Batch, 1, Freq, Time) Noisy Magnitude Spectrogram
    Output: (Batch, 1, Freq, Time) Estimated Ideal Ratio Mask (IRM) in [0, 1]
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        
        # Encoder (Downsampling spectral details into latent representation)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck Latent Representation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder (Upsampling and reconstructing spectral speech mask)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=(2, 2), padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=(2, 2), padding=1, output_padding=(0, 0)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=3, stride=(2, 2), padding=1, output_padding=(1, 0)),
            nn.Sigmoid(),  # Ideal Ratio Mask bounded in [0, 1]
        )

    def forward(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_mag: (Batch, 1, Freq, Time)
        Returns:
            clean_mag: Estimated clean magnitude spectrogram
        """
        orig_shape = noisy_mag.shape
        
        # Encoder
        e1 = self.enc1(noisy_mag)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder
        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        mask = self.dec1(d2)

        # Ensure exact spatial alignment with input
        if mask.shape != orig_shape:
            mask = F.interpolate(mask, size=(orig_shape[-2], orig_shape[-1]), mode="bilinear", align_corners=False)

        # Multiply estimated mask with noisy magnitude to suppress noise
        clean_mag = mask * noisy_mag
        return clean_mag
