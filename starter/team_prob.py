"""Per-frame P(home has the ball) from tracking alone: a logistic regression on team-symmetric
features, fitted on the open IDSSE matches (never on BAR-ATH). Used by possession_hmm.py in place
of its nearest-player evidence (knob W_PROB).

Features and design follow research/algo_investigation/feats_shared.py and research/bayes/
bayes_poss.py: 14 home-minus-away contrasts (nearest distance, players near the ball, time since
and until each team's last and next touch, touch rates over 1, 3 and 10 s) times 4 context inputs
(ball seen, time since and until it was seen, ball speed), no intercept. Fitted on both team orders,
so swapping the teams flips the log-odds exactly. The fit is `tune_idsse.py fit-team`; the
weights are in team_prob_weights.json.
"""

import json
import os
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_prob_weights.json")
FPS = 25            # the touch windows below are in frames at 25 fps; both IDSSE and BAR-ATH are
MAX_INTERP = 12     # frames: ball gaps up to this are interpolated
TOUCH_R = 1.5       # m: a player this close, and nearer than any opponent, is touching the ball
CAP = 250           # frames: cap on time since / until


def _interp_short(xy):
    obs = np.isfinite(xy).all(1)
    n, out, idx = len(xy), xy.copy(), np.arange(len(xy))
    o = np.flatnonzero(obs)
    if len(o) < 2:
        return out
    prev = np.maximum.accumulate(np.where(obs, idx, -1))
    nxt = np.minimum.accumulate(np.where(obs, idx, n)[::-1])[::-1]
    fill = ~obs & (prev >= 0) & (nxt < n) & (nxt - prev - 1 <= MAX_INTERP)
    for d in range(2):
        out[fill, d] = np.interp(idx[fill], o, xy[o, d])
    return out


def _speed(xy):
    s = np.full(len(xy), np.nan)
    step = xy[2:] - xy[:-2]
    s[1:-1] = np.hypot(step[:, 0], step[:, 1]) * FPS / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(sliding_window_view(np.pad(s, 2, constant_values=np.nan), 5), axis=1)


def _since(mask):
    idx = np.arange(len(mask))
    return np.minimum(idx - np.maximum.accumulate(np.where(mask, idx, -10 ** 9)), CAP).astype(np.float32)


def _until(mask):
    return _since(mask[::-1])[::-1]


def _rate(x, w, past):
    c, i = np.r_[0.0, np.cumsum(x)], np.arange(len(x))
    lo, hi = (np.maximum(i - w + 1, 0), i + 1) if past else (i, np.minimum(i + w, len(x)))
    return (c[hi] - c[lo]) / (hi - lo)


def _nearest(ball, P):
    d = np.hypot(P[..., 0] - ball[:, None, 0], P[..., 1] - ball[:, None, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmin(d, axis=1), np.sum(d < 3.0, axis=1), np.sum(d < 8.0, axis=1)


def design(x, y, team, home=0):
    """(n_frames, 70) design matrix from positions (ball in the last column) and the team of each
    slot, per slot or per frame and slot."""
    ball = _interp_short(np.stack([x[:, -1], y[:, -1]], 1).astype(np.float64))
    P = np.stack([x[:, :-1], y[:, :-1]], 2).astype(np.float64)
    known = np.isfinite(ball).all(1)
    is_home = np.broadcast_to(team == home, P.shape[:2])[..., None]
    dh, nh3, nh8 = _nearest(ball, np.where(is_home, P, np.nan))
    da, na3, na8 = _nearest(ball, np.where(is_home, np.nan, P))
    th = known & (dh <= TOUCH_R) & ~(da < dh)
    ta = known & (da <= TOUCH_R) & ~(dh < da)
    sh, sa, uh, ua = _since(th), _since(ta), _until(th), _until(ta)
    lg = lambda v: np.log(np.nan_to_num(v, nan=30.0) + 0.5)
    z = np.column_stack([
        lg(da) - lg(dh), np.clip(np.nan_to_num(da - dh), -20, 20) / 5,
        nh3 - na3, (nh8 - na8) / 3,
        np.log1p(sa) - np.log1p(sh), np.log1p(ua) - np.log1p(uh),
        *[_rate(th, w, True) - _rate(ta, w, True) for w in (25, 75, 250)],
        *[_rate(th, w, False) - _rate(ta, w, False) for w in (25, 75, 250)],
        np.sign(sa - sh), np.sign(ua - uh),
    ])
    s = np.column_stack([known, np.minimum(_since(known), 250) / 250, np.minimum(_until(known), 250) / 250,
                         np.log1p(np.nan_to_num(_speed(ball))) / 3])
    z, s = np.nan_to_num(z), np.nan_to_num(s)
    return np.hstack([z] + [z * s[:, [j]] for j in range(s.shape[1])])


def fit(X, y, l2=1.0, iters=50):
    """L2 logistic regression, no intercept, on both team orders (Newton steps). y = 1 for home."""
    X, y = np.vstack([X, -X]), np.r_[y, 1 - y].astype(np.float64)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) + l2 * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


_W = None   # the weights in use; tune_idsse.py swaps in per-fold weights for its held-out check


def p_home(x, y, team, w=None):
    global _W
    if w is None:
        if _W is None:
            _W = np.array(json.load(open(WEIGHTS))["w"])
        w = _W
    return 1 / (1 + np.exp(-np.clip(design(x, y, team) @ w, -30, 30)))


if __name__ == "__main__":
    # self-check: the fit recovers a known direction, and swapping the teams flips P exactly
    rng = np.random.default_rng(0)
    n, slots = 3000, 5
    x, y = rng.uniform(0, 100, (n, slots)), rng.uniform(0, 60, (n, slots))
    x[:, -1], y[:, -1] = x[:, 0] + rng.normal(0, 1, n), y[:, 0] + rng.normal(0, 1, n)   # ball at slot 0
    team = np.array([0, 0, 1, 1])
    X = design(x, y, team)
    w = fit(X, np.ones(n))
    p, q = p_home(x, y, team, w), p_home(x, y, 1 - team, w)
    assert p.mean() > 0.8, p.mean()
    assert np.allclose(p + q, 1), "not team-symmetric"
    print("ok")
