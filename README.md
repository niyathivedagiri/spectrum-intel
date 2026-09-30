# AI-Based Cognitive Radio Spectrum Intelligence

A simulated cognitive-radio pipeline that senses a band of spectrum, finds and
classifies the signals in it, and chooses a free channel — with an extension to
spectrum monitoring from a LEO satellite.

```
I/Q signals → FFT / spectrogram → detection → DL classifier → occupancy map → channel choice
```

## Components
- [x] Core DSP utilities — FFT spectrum, power, dB, frequency shift, AWGN
- [x] Signal generator — BPSK, QPSK, 16-QAM (RRC pulse shaping), OFDM, noise, interference
- [x] Labelled I/Q dataset — balanced classes, SNR −10 to +20 dB
- [ ] Spectrum analysis — PSD and spectrogram of a multi-channel band
- [ ] Energy detector — classical baseline
- [ ] CNN classifier — modulation recognition vs SNR
- [ ] Occupancy map and channel selection
- [ ] LEO extension — Doppler and path loss

## Project layout
```
spectrum_intel/
    dsp.py        core signal-processing tools
    signals.py    signal generator (6 classes)
    dataset.py    labelled dataset builder
scripts/
    preview_signals.py   draws the signal classes into outputs/
tests/            automated checks (python -m pytest)
```

## Signal classes

| Class | What it is | Occupied bandwidth (fraction of fs) |
|---|---|---|
| `bpsk` | 1 bit/symbol, root-raised-cosine pulses (β = 0.35, 8 samples/symbol) | 0.17 |
| `qpsk` | 2 bits/symbol, Gray coded | 0.17 |
| `qam16` | 4 bits/symbol, Gray coded | 0.17 |
| `ofdm` | 20 QPSK subcarriers, 128-point FFT, cyclic prefix 32 | 0.16 |
| `noise` | receiver noise only (empty channel) | whole band |
| `interference` | CW tone or linear chirp | one line / a sweep |

Each received example has a random carrier phase and AWGN at the chosen SNR,
and is scaled to unit power (like a receiver's AGC).

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest                                   # run all checks
python scripts/preview_signals.py                  # figures -> outputs/
python -m spectrum_intel.dataset --per-class 1000  # dataset -> data/generated/dataset.npz
```
