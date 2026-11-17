"""Referee whistles from the tactical video's soundtrack.

The tracking cannot see why play stopped. A foul or an offside is called with the ball still
tracked, often still moving, so nothing in the positions marks it -- the 66 s offside on the public
window is labelled 0% D by possession_hmm.py for exactly that reason. The referee's whistle marks
it, and the tactical video carries the stadium audio.

    ffmpeg 9.0.2 on PATH                      # decodes the mp4's AAC track; the submission was
                                              # generated with this version
    python starter/whistle.py                 # lists every whistle, caches them

    from starter.whistle import whistles
    for first, last in whistles():            # video frames, inclusive
        ...

How a whistle looks
-------------------
On a spectrogram the referee's whistle is two flat lines at once, near 3.80 kHz and 4.28 kHz,
each 30-38 dB above the median of the 3-5.5 kHz band, for 0.3-1 s. It starts and stops abruptly.
Crowd chants and voices also make narrow lines in that range, but they glide in pitch and rarely
land on both frequencies at once. Measured on the two public whistles (66.0 s and 187.7 s into
the window, 0.6 s and 0.5 s before the ground truth turns D):

    3797-3828 Hz and 4250-4336 Hz, prominence 27-38 dB, 0.5-0.6 s long

So a whistle frame needs both lines, each in a band around those frequencies, prominent above the
local band median. That ratio is what keeps the detector indifferent to how loud the crowd is.

What it finds, and what it misses
---------------------------------
Public window: exactly the two whistles above, nothing else, at every threshold from 18 to 25 dB.
The other two public stoppages (247.8 s, 273.7 s) are ball-out stoppages with no whistle in the
audio, and no public restart is whistled. Over the match it finds 60 whistles with the defaults,
including the first-half and second-half kick-offs, two of the four kick-offs after goals, and
the triple blow at full time. It misses a softer kick-off whistle at 1988.5 s (visible on the
spectrogram, below threshold) and the kick-off at 2483 s, where the audio is close to silent.
There is no full-match ground truth, so precision beyond the public window is unknown.
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import warnings

if sys.version_info < (3, 12):
    sys.exit("Python 3.12 or later is required (numpy 2.5.5 does not install on {}.{}).".format(
        *sys.version_info[:2]))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths  # noqa: E402

# Params
SAMPLE_RATE = 16000      # Hz. Whistle lines are under 5 kHz; 16 kHz keeps them below Nyquist.
N_FFT = 2048             # 128 ms window, 7.8 Hz bins: fine enough to separate the two lines.
HOP = 160                # 10 ms between spectra, four per video frame.
BAND = (3000, 5500)      # Hz. The prominence of each line is measured against this band's median.
LINE_LOW = (3650, 3950)  # Hz. The lower whistle line, measured at 3797-3828.
LINE_HIGH = (4150, 4450) # Hz. The upper line, measured at 4250-4336.
PROMINENCE = 20.0        # dB above the band median, both lines. Public whistles peak at 27-38;
                         # 18-25 all give the same public detections.
MIN_LENGTH = 0.15        # s. Shorter blips are chants crossing both bands by chance.
MAX_GAP = 0.05           # s. Gaps this short inside a whistle are bridged (warbling pea whistles).
CACHE_DIR = os.path.join(ROOT, "evaluation", "reports")
FFMPEG_VERSION = "9.0.2"  # the decoder the submission was generated with; another one may shift
                          # the samples slightly and so the detected whistle frames.


def video_path(game=GAME, data_dir=None):
    """The tactical mp4. Falls back to any *_tactical*.mp4 in the folder, since downloads get
    renamed (BAR-ATH_tactical-001.mp4)."""
    path = asset_paths(game, data_dir)["video"]
    if os.path.exists(path):
        return path
    found = sorted(glob.glob(os.path.join(os.path.dirname(path), game + "_tactical*.mp4")))
    if not found:
        raise FileNotFoundError(path)
    return found[0]


def check_ffmpeg_version():
    """Warn, without failing, when ffmpeg is not the version the submission was generated with."""
    try:
        first_line = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True,
                                    text=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return                      # a missing ffmpeg fails with its own error in read_audio
    words = first_line.split()
    found = words[2] if len(words) > 2 and words[:2] == ["ffmpeg", "version"] else first_line
    if found != FFMPEG_VERSION:
        warnings.warn("ffmpeg {} found, the submission was generated with {}; the detected whistles "
                      "(and so the labels) may differ.".format(found, FFMPEG_VERSION))


def read_audio(path, sample_rate=SAMPLE_RATE):
    """Mono int16 samples of the video's soundtrack, decoded by ffmpeg straight into memory."""
    check_ffmpeg_version()
    raw =subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
                          "-ar", str(sample_rate), "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def line_prominence(audio, sample_rate=SAMPLE_RATE, chunk=6000):
    """(low, high, times): dB of each whistle line's strongest bin above the band median, per
    10 ms hop, and the hop centres in seconds. Chunked so the match never sits in memory as a
    spectrogram."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    band = (freqs > BAND[0]) & (freqs < BAND[1])
    fb = freqs[band]
    low_bins = (fb > LINE_LOW[0]) & (fb < LINE_LOW[1])
    high_bins = (fb > LINE_HIGH[0]) & (fb < LINE_HIGH[1])
    window = np.hanning(N_FFT).astype(np.float32)

    n_hops = (len(audio) - N_FFT) // HOP + 1
    low, high = np.empty(n_hops, np.float32), np.empty(n_hops, np.float32)
    for start in range(0, n_hops, chunk):
        stop = min(n_hops, start + chunk)
        seg = audio[start * HOP:(stop - 1) * HOP + N_FFT].astype(np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(seg, N_FFT)[::HOP]
        power = np.abs(np.fft.rfft(frames * window, axis=1)[:, band]) ** 2
        db = 10 * np.log10(power + 1e-6)
        median = np.median(db, axis=1)
        low[start:stop] = db[:, low_bins].max(1) - median
        high[start:stop] = db[:, high_bins].max(1) - median
    times = (np.arange(n_hops) * HOP + N_FFT / 2) / sample_rate
    return low, high, times


def detect(low, high, times):
    """(start_s, end_s) of every stretch where both lines stand out, after bridging short gaps."""
    on = np.flatnonzero(np.minimum(low, high) > PROMINENCE)
    if not len(on):
        return []
    breaks = np.flatnonzero(np.diff(on) > MAX_GAP * SAMPLE_RATE / HOP + 1)
    starts, ends = on[np.r_[0, breaks + 1]], on[np.r_[breaks, len(on) - 1]]
    hop_s = HOP / SAMPLE_RATE
    return [(times[a], times[b] + hop_s) for a, b in zip(starts, ends)
            if times[b] + hop_s - times[a] >= MIN_LENGTH]


def whistles(game=GAME, data_dir=None, refresh=False):
    """[(first_frame, last_frame)] of every whistle, in video frames. Cached in evaluation/reports/
    after the first run (decoding and scanning 2 h of audio takes about half a minute)."""
    cache = os.path.join(CACHE_DIR, "{}_whistles.csv".format(game))
    if os.path.exists(cache) and not refresh:
        with open(cache) as handle:
            return [(int(r["first_frame"]), int(r["last_frame"])) for r in csv.DictReader(handle)]
    fps = GAMES[game]["fps"]
    found = [(int(round(s * fps)), int(round(e * fps)))
             for s, e in detect(*line_prominence(read_audio(video_path(game, data_dir))))]
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["first_frame", "last_frame"])
        writer.writerows(found)
    return found


def self_check(sample_rate=SAMPLE_RATE):
    """A synthetic two-tone whistle in noise, plus a single gliding tone that must not count."""
    rng = np.random.default_rng(0)
    t = np.arange(4 * sample_rate) / sample_rate
    audio = rng.normal(0, 300, len(t))
    blow = (t > 1.0) & (t < 1.5)
    audio += blow * 4000 * (np.sin(2 * np.pi * 3800 * t) + np.sin(2 * np.pi * 4290 * t))
    glide = (t > 2.5) & (t < 3.0)                     # a voice: one line, sweeping 3.6->4.4 kHz
    audio += glide * 4000 * np.sin(2 * np.pi * (3600 + 800 * (t - 2.5)) * t)
    found = detect(*line_prominence(audio.astype(np.int16), sample_rate))
    assert len(found) == 1 and abs(found[0][0] - 1.0) < 0.1 and abs(found[0][1] - 1.5) < 0.1, found


if __name__ == '__main__':
    self_check()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="rescan instead of using the cache")
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    found = whistles(refresh=args.refresh)
    print("{} whistles".format(len(found)))
    for first, last in found:
        s = first / fps
        print("  video {:d}:{:04.1f}  frames {}-{}  {:.2f} s".format(
            int(s // 60), s % 60, first, last, (last - first) / fps))
