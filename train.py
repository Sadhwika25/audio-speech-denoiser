"""
Training Script for AI Speech Denoising Autoencoder.
Trains convolutional autoencoder to estimate Ideal Ratio Mask (IRM)
minimizing Spectral Reconstruction Loss.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.stdout.reconfigure(line_buffering=True)

from src.models import SpeechDenoisingAutoencoder
from src.data import get_speech_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Train Speech Denoising Autoencoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--val-samples", type=int, default=60)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}", flush=True)

    print(f"[*] Synthesizing acoustic training pairs ({args.train_samples} train / {args.val_samples} val)...", flush=True)
    train_loader, val_loader = get_speech_dataloaders(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
    )

    model = SpeechDenoisingAutoencoder().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Trainable Parameters: {total_params:,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    checkpoint_path = os.path.join(args.checkpoint_dir, "denoiser_best.pt")

    print("-" * 55, flush=True)
    print(f"{'Epoch':<10}{'Train Loss (MSE)':<20}{'Val Loss (MSE)':<20}", flush=True)
    print("-" * 55, flush=True)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for noisy_mag, clean_mag in train_loader:
            noisy_mag, clean_mag = noisy_mag.to(device), clean_mag.to(device)
            optimizer.zero_grad()
            pred_clean_mag = model(noisy_mag)
            loss = criterion(pred_clean_mag, clean_mag)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy_mag.size(0)
            n_train += noisy_mag.size(0)

        scheduler.step()
        train_loss /= n_train

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for noisy_mag, clean_mag in val_loader:
                noisy_mag, clean_mag = noisy_mag.to(device), clean_mag.to(device)
                pred_clean_mag = model(noisy_mag)
                loss = criterion(pred_clean_mag, clean_mag)
                val_loss += loss.item() * noisy_mag.size(0)
                n_val += noisy_mag.size(0)

        val_loss /= n_val

        print(f"{epoch:<10}{train_loss:<20.6f}{val_loss:<20.6f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

    elapsed = time.time() - start_time
    print("-" * 55, flush=True)
    print(f"[+] Training completed in {elapsed:.2f} seconds.", flush=True)
    print(f"[+] Best Validation Loss: {best_loss:.6f}", flush=True)
    print(f"[+] Saved checkpoint to: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
