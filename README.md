# AI-Based Cognitive Radio Spectrum Intelligence

A software radio written in Python that covers how a wireless system works, from the raw signal up to
sending a photo through a satellite. Everything is simulated, so it runs on a laptop with no radio
hardware.

The project started as a spectrum monitor: listen to a band of 8 channels, work out which ones are busy
and what kind of signal is in each, and pick a free one to use. It was then extended into a complete
communication link that sends text and images packet by packet, copes with a bad channel, corrects its
own errors, and finally runs through a satellite passing overhead.

Every result comes from code in this repository and uses fixed random seeds, so it can be repeated. The
numbers below are what the code actually measured, and the places where things did not work as expected
are described too.


## Part 1: Sensing the spectrum

The radio records 8 channels side by side, splits them apart, and decides for each channel whether
anything is there and what it is: BPSK, QPSK, 16-QAM, OFDM, an interfering tone or chirp, or just noise.

The recognition is done by a small convolutional neural network (CNN) that looks directly at the raw
I/Q samples. On signals it never saw during training it is 99.5% accurate when the signal is at least as
strong as the noise, and 94.4% over the whole range from very weak (-10 dB) to strong (+20 dB).

The CNN was compared with the classic way of detecting a signal, an energy detector. When the noise level
is known exactly they are equally good. In practice the noise level is never known exactly, and with just
1 dB of uncertainty the energy detector needs a signal 6 dB stronger (or raises many false alarms), while
the CNN is not affected. With sensing switched on, collisions with the channels' owners dropped from
around half the time to about 8%.

The same CNN was also tested on a satellite at 550 km. The satellite's speed shifts every frequency
(Doppler, up to 46 kHz here) and the accuracy fell to 69%. Retraining it with frequency-shifted examples
brought it back to 99.5%.


## Part 2: Building the communication link

Here the radio stops only listening and starts transmitting. A message is cut into 64-byte packets, each
with a sequence number and a checksum (CRC). Each packet is turned into symbols, shaped into a waveform,
sent through a noisy channel, and picked up by a receiver that knows almost nothing in advance. The
receiver has to find where the packet starts, the exact timing, the frequency and phase error, and the
noise level, all from the received samples.

Text arrives perfectly once the signal is a little above the noise. The receiver loses only 0.2 to 0.6 dB
compared with textbook theory, which shows the synchronisation is working properly.

For images, a compressed file (PNG) breaks completely from a single wrong bit, so raw pixels are sent
instead. A "keep damaged packets" mode helps further: a picture with a few noisy rows is far better than a
picture with holes, and it improves image quality by about 6 to 10 dB where packets are lost.

With 64-byte packets the link delivers about 89 kbps with BPSK, 155 kbps with QPSK and 247 kbps with
16-QAM, but 16-QAM needs about 10 dB more signal than BPSK. A good part of every packet is overhead (29%
for BPSK, 51% for 16-QAM), because the preamble and headers stay the same size while the data part gets
shorter.

A live dashboard (a local web page, no internet needed) shows packets going through, the constellation,
spectrum and waveform, and lets the settings be changed while it runs.


## Part 3: A more realistic channel

Real channels do more than add noise, so the simulation adds fading (the signal strength going up and
down), echoes (the signal arriving twice, slightly delayed) and interference (a tone or a pulsing jammer).

Fading is costly. To get 99% of packets through, the link needs 5 dB with plain noise, but 19 dB when the
signal fades badly (Rayleigh fading). An equaliser trained on the preamble fixes echoes up to about 24
microseconds, but it adds its own noise on a clean channel, so it is not free.

Testing with fading exposed a real bug in the receiver: when the start of a packet arrived during a fade,
it sometimes locked onto the wrong place and lost the whole packet, even with a strong signal. This was
fixed, and a test now checks that exact case.


## Part 4: Choosing how fast to send

A strong signal can carry 16-QAM, a weak one only BPSK. The transmitter therefore chooses the modulation
for each packet from what the receiver reported about the previous one, as 4G does.

With a user moving towards and away from the base station, this delivered more data than any fixed choice
(142 kbps against 126 kbps for the best fixed one). Two lessons came out of it. First, aiming for a fixed
10% error rate is not the same as getting the most data through: sometimes a faster mode that loses a few
more packets still wins. Second, adapting only helps if the channel changes slowly. When it changes faster
than the feedback arrives, the information is already out of date, and adapting can do worse than simply
picking one robust mode.


## Part 5: Choosing which channel to use

The 8-channel band has owners who switch on and off. Several ways of choosing a channel were compared,
all against exactly the same owners and the same sensing mistakes, so the comparison is fair.

Learning how often each owner comes back gave the fewest collisions (6.3%). Also learning how good each
channel is for the radio's own link almost doubled the data sent (214 kbps instead of 112), but it did not
reduce collisions.

The interesting surprise came with poor sensing. A collision does not always destroy the radio's packet if
its signal is much stronger than the owner's. The rule that chases the most data "learned" this and started
colliding 38% of the time, which helps the radio but hurts the owner. A penalty for likely collisions was
then added. It cut collisions to about 17% while keeping about twice the data of the original rule. This
penalty was added after the problem was seen, and the code records that.


## Part 6: Correcting errors

The link can use the error-correcting code from Wi-Fi (a convolutional code with a Viterbi decoder) in three
strengths (rate 1/2, 2/3 and 3/4), plus an interleaver that shuffles bits so bursts of errors get spread out.

With rate 1/2 coding, 16-QAM works with about 10 dB less signal, and QPSK with about 5 dB less. Using "soft"
information (how sure the receiver is about each bit) is worth roughly another 1.5 to 3 dB over plain 0/1
decisions. Coding costs speed though: rate 1/2 roughly halves the data rate, so on a strong link no coding
is still best.

Two results were unexpected. First, once the data was protected, the weakest part became the small packet
header, which is still sent without coding, so weak packets now fail because of the header. Wi-Fi protects
its header for exactly this reason. Second, soft decoding did worse than simple decoding when a jammer was
pulsing. The receiver estimated the noise once per packet, so bits hit by a burst looked very confident but
were wrong. Measuring the noise locally around each symbol fixed it, and soft decoding is now the best in
every case tested.


## Part 7: Recognising signals it has never seen

A classifier trained on 6 kinds of signal will always name one of those 6, even for something completely
new. The goal here is to let it answer "unknown".

The usual approach, trusting the network's confidence, did not work at all: the network was just as sure
about an FM radio station as about QPSK and accepted 98% of unknown signals. Measuring how far a signal is
from everything seen in training (the Mahalanobis distance inside the network) worked much better. When the
signal was at least as strong as the noise, it rejected every FSK, AM and FM signal while keeping 95% of the
known ones.

It could not catch 8-PSK or 64-QAM, which it labelled as QPSK and 16-QAM. The network never needed to tell
them apart, so the information is simply not there, and nothing added afterwards can recover it. Earlier
layers of the network and classic signal statistics did not solve it either. Fixing this would need changes
to how the network is trained.


## Part 8: Through a satellite

Finally, the whole link runs through a satellite pass: a phone-strength transmitter on the ground sending
to a satellite at 550 km. During an 8-minute pass the signal gets stronger and weaker (about 0 to 11 dB) and
the frequency shifts by up to 46 kHz because of the satellite's speed. The receiver on its own only copes
with about 6 kHz.

With no correction, almost nothing got through (0.7 MB in a pass, only near the top). Searching for the
right frequency at the receiver worked (11.5 MB) but needed about 8 times more processing. Predicting the
shift from the satellite's known orbit, which is what real ground stations do, gave the same 11.5 MB for no
extra processing. After that, the remaining problem is the accuracy of the transmitter's crystal: a cheaper
crystal broke the link again, and a small extra search fixed it.

The link can also choose the modulation and the error-correcting code together for every packet (using the
error rates measured in Part 6). That gave 11.5 MB per pass, compared with 9.6 MB when only the modulation
could change. With a weaker transmitter the difference was much bigger, because below about 0 dB only coded
signals get through at all.


## How to run it

Python 3 on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                       # runs the 213 automated checks
python scripts/run_all.py              # reruns the spectrum and classifier experiments (~3 min)
python scripts/dashboard.py            # opens the live dashboard in the browser
```

Each part has its own experiment script in `scripts/`, for example `exp_fec.py` for error correction or
`exp_leo_link.py` for the satellite. Every run is saved in a new folder under `results/experiments/` and
never overwrites an older one. The original classifier results are frozen in `results/validated/`, and
`python scripts/check_regression.py` checks that later additions have not changed them.


## What is in the repository

- `spectrum_intel/` is the main code: signal generation, the band and channelizer, detectors, the CNN,
  the transmitter, channel and receiver, error correction, adaptation, channel selection, unknown-signal
  detection, the satellite model and the dashboard.
- `scripts/` contains the experiments and demos.
- `models/` holds the two trained networks (normal and Doppler-trained).
- `results/` holds the measured numbers and figures from every experiment.
- `tests/` holds the automated checks.


## Limitations and next steps

Everything here is simulated. The signals, the band and the satellite pass are generated, so the next real
step is testing on recorded radio signals, for example from an RTL-SDR or the public RadioML dataset. The
satellite orbit is a simple circle without Earth's rotation, weather or antenna losses, and real satellite
orbit data (SGP4) is not used. The packet header is still sent without error correction, packets that fail
are not resent, and the Viterbi decoder in Python is too slow for real-time 16-QAM. Another next step is
finding signals of any width directly in a spectrogram image, rather than only in 8 fixed channels.
