"""Velocity vectors from the tracking: the direction the speed column leaves out.

The ATD gives every track slot a position and a speed. Speed is a magnitude; it cannot say whether
a player is running at the ball or away from it, or whether the ball just changed direction. This
module differences consecutive positions to get (vx, vy) in metres per second, for every slot and
for the ball.

    pip install -r requirements.txt          # numpy only
    python starter/velocity.py               # computes velocities and checks them against speed

    from starter.velocity import read_positions, velocity
    x, y, speed = read_positions()           # (n_frames, n_slots + 1), metres; last column = ball
    vx, vy = velocity(x, y)                  # m/s, NaN where there is no honest answer

Rows are video frames, so row i lines up with row i of a submission -- the same frame_id - 1 shift
load_atd.py describes.

How it is computed
------------------
The video runs at 25 fps, so consecutive frames are 1/25 s = 40 ms apart:

    vx[i] = (x[i] - x[i-1]) * fps            # metres per frame -> metres per second

Positions are scaled to metres on both axes first. The coordinates are normalised 0-1 and the pitch
is not square, so differencing the raw fractions would understate every movement along the short
axis.

Three cases give NaN instead of a number:

  * the slot is missing on frame i or on frame i-1. A slot that vanishes and reappears has no
    velocity across the gap -- bridging it would spread one movement over however long the tracker
    was blind.
  * the implied speed is beyond what the object can do: 12 m/s for a player (the fastest
    footballers peak around 10-10.5), 40 m/s for the ball (above a hard shot). These are the tracker
    handing a slot to a different person between two frames, or a false ball detection, not
    movement. MAX_PLAYER_SPEED and MAX_BALL_SPEED below explain how each limit was chosen.
  * frame 0, which has no previous frame.

Checked against the speed column
--------------------------------
On graded frames the player velocities agree with the provided speed: median |v| / speed is 1.0005,
correlation 0.97, and 98% of frames are within 0.9 m/s. The speed column looks like the same
frame-to-frame difference, minus the jumps -- it never exceeds 9.4 m/s, while the unfiltered
difference reaches 103. Dropping the 4320 jumps (0.11% of frames) is what lifts the correlation
from 0.93 to 0.97.

The ball's speed column is not usable: it is present on 20971 frames against 113488 with a ball
position, it never exceeds 7.3 m/s, and it correlates 0.06 with the ball's own movement. For the
ball, the differenced position is the only velocity there is.

Smoothing
---------
`smooth` is the half-width of a centred moving average over the velocities, in frames. 0 is the
raw difference; 2 averages +/- 80 ms, which takes the frame-to-frame jitter out of the player tracks
without blurring a change of direction. Gaps stay gaps: a frame whose raw velocity is NaN is NaN
after smoothing too, and missing neighbours are left out of the average rather than counted as 0.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DEFAULT_SOURCE, half_video_blocks, require, source_frames  # noqa: E402

# Params
# Raw ATD line: "<frame_id>:<player>;<player>;...:<ball>", each group "x,y,speed" with NaN for a
# missing observation.
FIELD_SEP = ":"
GROUP_SEP = ";"

# Fastest plausible movement, in m/s. Anything faster between two frames is a tracking jump, not
# movement. Both limits are judgement calls backed by physics and by the data, not tuned values.
#
# Players, 12 m/s (43 km/h). The fastest footballers peak at about 10-10.5 m/s and Usain Bolt's peak
# was about 12.4, so nothing on this pitch actually moves at 12. On top of that, one 40 ms step is
# noisy: 98% of steps are within 0.9 m/s of the speed column, so a real 10.5 m/s sprint can read
# as 11.4 for a frame. 12 leaves that margin. The unfiltered graded steps agree: the count falls
# steadily up to 12 (8-9 m/s: 4344, 9-10: 744, 10-11: 350, 11-12: 274) and then stops falling
# (12-14: 390, 14-20: 697, 20-50: 435, over 50: 83, max 103) -- above 12 is a separate population
# of slot swaps. The provided speed column itself never exceeds 9.4. Steps between 10 and 12 (624)
# are a mix of real sprints and noise, and are kept: losing a real sprint costs more than keeping
# one noisy frame.
#
# Ball, 40 m/s (144 km/h). Hard shots are 30-35 m/s and the ball almost never leaves a boot faster
# than 40. In the data the count per m/s falls from 20-25 m/s (340 per m/s) to 30-35 (26) and
# 35-40 (17), then levels out (40-50: 10, 50-75: 4, plus 13 steps over 75) -- the same change from
# a falling tail to a flat floor of false detections. The ball edge is softer than the player
# one: positions are 2D on the pitch plane, so a ball in the air is placed where its image projects
# onto the grass, and that projection can briefly overstate its speed. Some steps just under 40 are
# still noise. Raise it only with a reason.
MAX_PLAYER_SPEED = 12.0
MAX_BALL_SPEED = 40.0

# Default smoothing half-width, in frames: +/- 2 frames = +/- 80 ms at 25 fps.
SMOOTH = 2


def read_positions(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """(x, y, speed), each (n_frames, n_slots + 1), indexed by video frame. The last column is the
    ball. x and y are in metres, speed in m/s; NaN where the tracker had nothing.

    Reads the raw ATD with the standard library and numpy, so it runs without kloppy. The 196 MB
    tactical file takes a few seconds.
    """
    raw_path, = require(["atd_raw"], game, data_dir, source)
    length, width = GAMES[game]["pitch"]
    n_frames = source_frames(source, game)

    rows = {}
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            groups = parts[1].split(GROUP_SEP) + [parts[2].strip()]
            rows[int(parts[0]) - 1] = [[float(value) for value in group.split(",")]
                                       for group in groups]

    n_columns = len(next(iter(rows.values())))
    data = np.full((n_frames, n_columns, 3), np.nan, dtype=np.float32)
    for frame, values in rows.items():
        data[frame] = values
    return data[..., 0] * length, data[..., 1] * width, data[..., 2]


def velocity(x, y, fps=GAMES[GAME]["fps"], smooth=SMOOTH,
             max_player_speed=MAX_PLAYER_SPEED, max_ball_speed=MAX_BALL_SPEED):
    """(vx, vy) in m/s, same shape as x and y, from positions in metres one video frame apart.

    The last column is taken to be the ball, as read_positions returns it, and gets the ball's
    speed limit. Pass a single column (shape (n, 1)) and it is treated as the ball too -- slice the
    players off first if that is not what you want.
    """
    vx = np.full(x.shape, np.nan, dtype=np.float32)
    vy = np.full(y.shape, np.nan, dtype=np.float32)
    vx[1:] = (x[1:] - x[:-1]) * fps
    vy[1:] = (y[1:] - y[:-1]) * fps

    limit = np.full(x.shape[1], max_player_speed, dtype=np.float32)
    limit[-1] = max_ball_speed
    with np.errstate(invalid="ignore"):
        jump = np.hypot(vx, vy) > limit
    vx[jump] = np.nan
    vy[jump] = np.nan

    if smooth > 0:
        vx, vy = _moving_average(vx, smooth), _moving_average(vy, smooth)
    return vx, vy


def _moving_average(values, half_width):
    """Centred mean over +/- half_width rows, ignoring NaN neighbours and keeping NaN rows NaN.

    Cumulative sums, so a wide window costs the same as a narrow one.
    """
    finite = np.isfinite(values)
    total = np.concatenate((np.zeros((1, values.shape[1])),
                            np.cumsum(np.where(finite, values, 0.0), axis=0)))
    count = np.concatenate((np.zeros((1, values.shape[1])), np.cumsum(finite, axis=0)))

    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total[high] - total[low]) / (count[high] - count[low])
    return np.where(finite, mean, np.nan).astype(np.float32)


def graded_mask(n_frames, game=GAME):
    """True on the video frames inside the two halves -- the ones the evaluation scores."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def compare_with_speed(vx, vy, speed, mask):
    """Agreement between |v| and the provided speed column, over the masked frames."""
    magnitude = np.hypot(vx, vy)[mask].ravel()
    reference = speed[mask].ravel()
    both = np.isfinite(magnitude) & np.isfinite(reference)
    magnitude, reference = magnitude[both], reference[both]
    moving = reference > 1.0  # a ratio near zero speed is all noise
    return {
        "n": int(both.sum()),
        "median_ratio": float(np.median(magnitude[moving] / reference[moving])),
        "corr": float(np.corrcoef(magnitude, reference)[0, 1]),
        "within": float(np.mean(np.abs(magnitude - reference) <= 0.9)),
    }


def self_check(fps=GAMES[GAME]["fps"]):
    """Synthetic tracks with known answers. Needs no data; runs before every real computation."""
    t = np.arange(10, dtype=np.float32)
    x = np.stack([t * 0.2, t * 0.1], axis=1)  # player at 0.2 m/frame = 5 m/s, ball at 2.5 m/s
    y = np.zeros_like(x)
    vx, vy = velocity(x, y, fps, smooth=0)
    assert np.isnan(vx[0]).all()
    assert np.allclose(vx[1:, 0], 0.2 * fps) and np.allclose(vx[1:, 1], 0.1 * fps)

    x[5, 0] = np.nan                          # a gap: no velocity into or out of it
    x[8, 0] += 1.0                            # a 1 m jump in one frame = 25 m/s, a slot swap
    vx, _ = velocity(x, y, fps, smooth=0)
    assert np.isnan(vx[[5, 6, 8, 9], 0]).all() and np.isfinite(vx[7, 0])

    vx, _ = velocity(x, y, fps, smooth=2)     # smoothing keeps gaps and averages the rest
    assert np.isnan(vx[[5, 6, 8, 9], 0]).all() and np.allclose(vx[7, 0], 0.2 * fps)


if __name__ == '__main__':
    self_check()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="tactical or broadcast")
    parser.add_argument("--smooth", type=int, nargs="+", default=[0, SMOOTH],
                        help="half-widths in frames to compare (0 = raw)")
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    print("reading {} tracking ...".format(args.source))
    x, y, speed = read_positions(source=args.source)
    mask = graded_mask(len(x)) if args.source == "tactical" else np.ones(len(x), dtype=bool)
    print("  {} frames x {} slots + ball, {} fps -> {:.0f} ms between frames\n".format(
        len(x), x.shape[1] - 1, fps, 1000.0 / fps))

    for half_width in args.smooth:
        vx, vy = velocity(x, y, fps, smooth=half_width)
        players = compare_with_speed(vx[:, :-1], vy[:, :-1], speed[:, :-1], mask)
        ball = np.hypot(vx[:, -1], vy[:, -1])[mask]
        dropped = int(np.sum(np.isfinite(x[1:, :-1]) & np.isfinite(x[:-1, :-1])
                             & ~np.isfinite(vx[1:, :-1])))
        print("smooth +/-{} frames ({:.0f} ms)".format(half_width, 1000.0 * half_width / fps))
        print("  players vs speed column: median ratio {median_ratio:.4f}  corr {corr:.4f}  "
              "within 0.9 m/s {within:.1%}  (n={n})".format(**players))
        print("  player jumps dropped: {}".format(dropped))
        print("  ball |v| on {} graded frames, percentiles 50/90/99: {}\n".format(
            int(np.isfinite(ball).sum()),
            np.round(np.nanpercentile(ball, [50, 90, 99]), 1).tolist()))
