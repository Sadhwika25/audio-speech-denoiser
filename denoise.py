"""
Audio Denoising & Waveform Reconstruction Inference Driver.
Takes noisy speech audio, applies neural spectral masking, reconstructs
waveform via iSTFT, and exports before/after audio samples to disk.
"""

import os
import argparse
import torch
import soundfile as sf

from src.models import SpeechDenoisingAutoencoder
from src.dsp.stft_processor import STFTProcessor, calculate_snr, add_acoustic_noise
from src.data.speech_dataset import generate_synthetic_speech


def main():
    parser = argparse.ArgumentParser(description="Denoise Audio using Trained Autoencoder")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/denoiser_best.pt")
    parser.add_argument("--snr-in", type=float, default=3.0, help="Input SNR in dB")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")
    processor = STFTProcessor(sample_rate=16000)

    # Load model
    model = SpeechDenoisingAutoencoder()
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] Loaded trained model from {args.checkpoint}")
    else:
        print("[!] No checkpoint found; running with initialized weights.")

    model.eval()

    # Generate synthetic clean speech and add background noise
    print(f"[*] Synthesizing speech and injecting background noise at {args.snr_in:.1f} dB SNR...")
    clean_wf = generate_synthetic_speech(duration=1.5, sample_rate=16000)
    noisy_wf = add_acoustic_noise(clean_wf, target_snr_db=args.snr_in, noise_type="gaussian")

    # Step 1: Forward STFT to get Noisy Magnitude and Phase
    noisy_mag, noisy_phase = processor.forward(noisy_wf)

    # Step 2: Pass Noisy Magnitude through Neural Autoencoder
    with torch.no_grad():
        noisy_mag_batch = noisy_mag.unsqueeze(0).unsqueeze(0)  # (1, 1, Freq, Time)
        pred_clean_mag = model(noisy_mag_batch).squeeze(0).squeeze(0)

    # Step 3: Inverse STFT using predicted clean magnitude + original noisy phase
    denoised_wf = processor.inverse(pred_clean_mag, noisy_phase, length=len(clean_wf))

    # Step 4: Calculate Quality Metrics (SNR)
    snr_before = calculate_snr(clean_wf, noisy_wf)
    snr_after = calculate_snr(clean_wf, denoised_wf)
    snr_gain = snr_after - snr_before

    print("\n" + "=" * 55)
    print("         SPEECH DENOISING PERFORMANCE REPORT")
    print("=" * 55)
    print(f"  Input Noisy SNR:        {snr_before:>8.2f} dB")
    print(f"  Output Denoised SNR:     {snr_after:>8.2f} dB")
    print(f"  SNR Improvement:        {snr_gain:>+8.2f} dB")
    print("=" * 55)

    # Save audio files
    clean_path = os.path.join(args.output_dir, "1_clean_original.wav")
    noisy_path = os.path.join(args.output_dir, "2_noisy_input.wav")
    denoised_path = os.path.join(args.output_dir, "3_denoised_output.wav")

    sf.write(clean_path, clean_wf.numpy(), 16000)
    sf.write(noisy_path, noisy_wf.numpy(), 16000)
    sf.write(denoised_path, denoised_wf.numpy(), 16000)

    print(f"[+] Clean reference audio saved:   {clean_path}")
    print(f"[+] Noisy input audio saved:       {noisy_path}")
    print(f"[+] Denoised reconstructed audio:  {denoised_path}")


if __name__ == "__main__":
    main()
