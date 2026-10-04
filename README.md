# AI-Based Cognitive Radio Spectrum Intelligence

A simulated cognitive radio that **senses an 8-channel band, recognises what is in
each channel with a CNN, and picks a free channel to transmit on**, then tests the
same system **from a LEO satellite**, where orbital Doppler breaks a normally trained
classifier and Doppler-aware training fixes it.

```
I/Q samples → spectrum & spectrogram → channelizer → detection → CNN classifier → occupancy map → channel choice
                                                                              ↘ LEO extension: Doppler + path loss
```

## Results at a glance

| Experiment | Result |
|---|---|
| Classify 6 signal types (BPSK, QPSK, 16-QAM, OFDM, noise, interference) from raw I/Q | **99.5%** accuracy at SNR ≥ 0 dB, 94.4% over −10 to +20 dB (unseen test set) |
| Detect a signal in a channel, noise level known exactly | Energy detector and CNN equal: 90% detection at −8 dB SNR |
| Same, with the noise level uncertain by ±1 dB (realistic) | CNN still detects at −8 dB; energy detector needs −2 dB (**6 dB worse**) or raises 33% false alarms |
| Full 8-channel system, 10 runs × 60 slots | CNN names the signal in **98.8%** of busy channel-slots; sensing cuts collisions from 46–66% to **≈ 8%** |
| Same CNN on a satellite at 550 km, 2 GHz (Doppler up to ±46 kHz) | Drops to **69%** at SNR ≥ 0 dB |
| CNN trained with Doppler augmentation | Back to **99.5%** with Doppler, stays above 97% for the whole 8-minute pass |

<table>
<tr><td><img src="results/classifier_accuracy_vs_snr.png" width="420"></td><td><img src="results/classifier_confusion.png" width="360"></td></tr>
<tr><td><img src="results/spectrum_spectrogram.png" width="420"></td><td><img src="results/occupancy_map.png" width="420"></td></tr>
<tr><td><img src="results/detection_noise_uncertainty.png" width="420"></td><td><img src="results/channel_selection.png" width="420"></td></tr>
<tr><td><img src="results/leo_accuracy_vs_snr.png" width="420"></td><td><img src="results/leo_pass.png" width="420"></td></tr>
</table>

All numbers are in [`results/metrics.json`](results/metrics.json).

## End-to-end communication link (upgrade in progress)

The project is being extended from spectrum *analysis* into a full communication system:
information → bits → packets → modulation → I/Q → channel → receiver → information.

**Phase 1 (done): text over an AWGN link.** The receiver knows nothing in advance except the
preamble and pilots. It finds the frame, the symbol timing, the frequency offset, the phase and
the SNR from the received samples alone.

| Measured (100 frames per point, AWGN, random phase and arrival time) | BPSK | QPSK | 16-QAM |
|---|---|---|---|
| Loss vs textbook theory at BER 10⁻³ | 0.2 dB | 0.3 dB | 0.6 dB |
| SNR per sample for ≥ 90% of 64-byte packets to pass CRC | −1 dB | 3 dB | 10 dB |

- Frame: 63-symbol m-sequence preamble, 32-bit BPSK PHY header (modulation, length, CRC-16),
  payload, one pilot every 16 symbols. Packets: sequence number, length, CRC-16; losses are reported, never hidden.
- Receiver: matched filter → segmented preamble correlation (timing + frame sync, robust to frequency offset;
  threshold set from measured noise statistics) → multi-lag frequency estimate refined by the pilot phase slope →
  pilot-based gain/phase tracking → demapping → CRC. Optional coarse frequency search handles LEO-scale Doppler.

**Phase 2 (done): images.** Images are sent as raw 8-bit pixels (a compressed file breaks with one bit error).
Same channel, two receiver policies: *strict* drops packets that fail the CRC; *error-tolerant* keeps a damaged
packet if its header is self-consistent (CRC-valid packets always take priority).

| Measured (64×64 RGB, 193 packets, 2 trials per point) | BPSK | QPSK | 16-QAM |
|---|---|---|---|
| SNR for image PSNR ≥ 30 dB, error-tolerant | 0 dB | 2 dB | 10 dB |
| SNR for image PSNR ≥ 30 dB, strict | 2 dB | 4 dB | 12 dB |
| Mean PSNR gain of error-tolerant mode where packets were lost | +10.5 dB | +9.4 dB | +5.7 dB |
| SNR where the compressed PNG always decodes | 2 dB | 6 dB | 12 dB |

```bash
python scripts/demo_image_link.py --snr 2                                      # original vs strict vs tolerant vs PNG
python scripts/exp_image_link.py                                               # PSNR/SSIM vs SNR sweep (~3 min)
python scripts/demo_text_link.py --text "Hello satellite" --mod qpsk --snr 3   # every stage, printed + figure
python scripts/exp_link_ber.py                                                 # BER/SER/packet success vs SNR
python scripts/check_regression.py                                             # original results unchanged?
```
New experiment runs are saved under `results/experiments/<name>/<time>-seed<N>/` and are never overwritten.
The original validated numbers are frozen in `results/validated/metrics_v1.json`.

## How it works

1. **Signals** (`signals.py`): BPSK, QPSK and 16-QAM with root-raised-cosine pulse shaping
   (8 samples/symbol, β = 0.35); OFDM with 20 QPSK subcarriers, 128-point FFT and cyclic prefix;
   receiver noise; CW-tone or chirp interference. Random carrier phase and timing, AWGN at a chosen SNR.
2. **Band** (`spectrum.py`): eight channels side by side at 8× the channel rate. A channelizer
   (shift → root-raised-cosine low-pass → decimate) returns each channel in exactly the
   classifier's input format; its filter keeps channelised noise white.
   The noise floor is estimated from the median of the spectrum.
3. **Detection** (`detector.py`): energy detector with a Neyman–Pearson threshold for a
   1% false-alarm rate, compared with the CNN ("occupied" = top class is not noise).
4. **Classification** (`classifier.py`): 1-D CNN on the 2 × 1024 I/Q array
   (4 conv blocks, global average pooling, 110k parameters), trained on 15,000 examples
   from −10 to +20 dB, tested on a separate seed.
5. **Cognitive layer** (`occupancy.py`): licensed users switch on/off as Markov chains;
   the secondary user senses in slot *t* and transmits in slot *t + 1*, staying put while
   its channel is free and otherwise hopping to the free channel idle the longest.
6. **LEO** (`leo.py`): overhead pass at 550 km (7.59 km/s, ~8 min above 10°), Doppler
   f_D = −f_c·ṙ/c, free-space path loss 153–164 dB, link budget giving 0–11 dB SNR in 1 MHz.

## Project layout
```
spectrum_intel/   dsp, signals, dataset, spectrum, detector, classifier, occupancy, leo, plots
                  payload, coding, packet, modulation, transmitter, channel, receiver, link,
                  metrics, experiments                       (communication link)
scripts/          train_classifier, evaluate_detection, run_spectrum_demo, run_leo_experiment,
                  preview_signals, run_all, demo_text_link, exp_link_ber, check_regression,
                  demo_image_link, exp_image_link
models/           cnn_baseline.pt, cnn_doppler.pt  (trained weights, 455 KB each)
results/          figures + metrics.json (validated), validated/ (frozen copy), experiments/ (new runs)
tests/            114 automated checks
```

## Setup and run (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                    # 114 checks
python scripts/run_all.py           # every experiment with the saved models (~3 min)
python scripts/run_all.py --retrain # retrain both CNNs too (~20 min CPU, faster on Apple GPU)
```

## Limitations and next steps
Simulated signals only; channel effects limited to AWGN, phase, timing and Doppler;
fixed 8-channel grid; simplified orbit (directly overhead, no Earth rotation).
Next: real captures with an RTL-SDR or the RadioML 2018 dataset, multipath fading,
open-set "unknown signal" detection, spectrogram object detection for signals of any
width, and SGP4 orbits with real satellite elements.
