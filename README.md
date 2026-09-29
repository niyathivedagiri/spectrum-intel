# AI-Based Cognitive Radio Spectrum Intelligence

A simulated cognitive-radio pipeline that senses a band of spectrum, finds and
classifies the signals in it, and chooses a free channel — with an extension to
spectrum monitoring from a LEO satellite.

```
I/Q signals → FFT / spectrogram → detection → DL classifier → occupancy map → channel choice
```

## Components
- [x] Core DSP utilities — FFT spectrum, power, dB, frequency shift, AWGN
- [ ] Signal generator — BPSK, QPSK, 16-QAM, OFDM, noise, interference
- [ ] Spectrum analysis — PSD and spectrogram of a multi-channel band
- [ ] Energy detector — classical baseline
- [ ] CNN classifier — modulation recognition vs SNR
- [ ] Occupancy map and channel selection
- [ ] LEO extension — Doppler and path loss

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests
```bash
python -m pytest
```
