"""
SNR Gain & Noise Suppression Benchmarking Driver.
Evaluates model across diverse acoustic noise conditions (0 dB to 15 dB).
"""

import argparse
import os
import torch
import numpy as np

from src.models import SpeechDenoisingAutoencoder
from src.dsp.stft_processor import STFTProcessor, calculate_snr, add_acoustic_noise
from src.data.speech_dataset import generate_synthetic_speech


def main():
    parser = argparse.ArgumentParser(description="Benchmark SNR Gains")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser_best.pt")
    parser.add_argument("--num-trials", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cpu")
    processor = STFTProcessor(sample_rate=16000)

    model = SpeechDenoisingAutoencoder()
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_snr_levels = [0.0, 3.0, 6.0, 10.0]
    
    print("\n" + "=" * 65)
    print("      ACOUSTIC NOISE SUPPRESSION BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Input SNR (dB)':<18}{'Output SNR (dB)':<20}{'SNR Gain (dB)':<18}{'Avg Latency (ms)':<15}")
    print("-" * 65)

    import time
    for target_snr in test_snr_levels:
        gains = []
        out_snrs = []
        latencies = []

        for _ in range(args.num_trials):
            clean_wf = generate_synthetic_speech(duration=1.0, sample_rate=16000)
            noisy_wf = add_acoustic_noise(clean_wf, target_snr_db=target_snr)

            t0 = time.perf_counter()
            with torch.no_grad():
                noisy_mag, noisy_phase = processor.forward(noisy_wf)
                pred_mag = model(noisy_mag.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                denoised_wf = processor.inverse(pred_mag, noisy_phase, length=len(clean_wf))
            latencies.append((time.perf_counter() - t0) * 1000.0)

            in_snr = calculate_snr(clean_wf, noisy_wf)
            out_snr = calculate_snr(clean_wf, denoised_wf)
            out_snrs.append(out_snr)
            gains.append(out_snr - in_snr)

        avg_out = np.mean(out_snrs)
        avg_gain = np.mean(gains)
        avg_lat = np.mean(latencies)

        print(f"{target_snr:<18.1f}{avg_out:<20.2f}{avg_gain:<+18.2f}{avg_lat:<15.2f}")

    print("=" * 65)
    print("[+] Benchmark completed successfully.")


if __name__ == "__main__":
    main()
