"""Choose possession_hmm.py knobs on other games: the 7 IDSSE Bundesliga matches, not BAR-ATH.

RULES.md allows external data about other games and does not allow tuning on the public ground
truth, so this is where knob values can be chosen and held out. Data: CC-BY 4.0, Bassek et al. 2025,
https://doi.org/10.1038/s41597-025-04505-y, mirrored at huggingface.co/datasets/pysport/idsse-data.
Put the XML files in data/idsse/.

    python starter/tune_idsse.py calibrate            # IDSSE vs ATD statistics
    python starter/tune_idsse.py search --n 300       # random joint search, appends results
    python starter/tune_idsse.py report               # leave-one-match-out result
    python starter/tune_idsse.py fit-team             # team_prob.py weights, all 7 matches
    python starter/tune_idsse.py team-check --seed 1  # held out: W_PROB against nearest-player

Labels: BallStatus 0 = D, else BallPossession 1 = H, 2 = A, between the first and last live frame of
each half. Each setting is scored twice, raw and with short spells merged, because IDSSE spells are
shorter than the public window's and it is not known whether DFL and the ground truth mean the same
by "owning" the ball. A setting must beat the default on both.

Degrading: IDSSE is clean optical tracking and the ATD is not, so positions go through `degrade`,
calibrated on the ATD alone (no labels): ball gaps resampled from the ATD's own gap and visible run
lengths, player dropouts, and slow position error sized by `calibrate`.

Not tuned, because IDSSE has no audio and no Smart Tagging: whistle, set-piece and phase knobs. Those
signals are stood in for by DFL events at fixed knob values: restarts thinned to the ATD's set-piece
count, fouls and offsides as whistles. Those events are human-produced data about other games, which
the rules allow; here they only simulate the automatic signals.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "evaluation")]
import possession_hmm as hmm  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from velocity import read_positions  # noqa: E402

# Params
IDSSE_DIR = os.path.join(ROOT, "data", "idsse")
CACHE_DIR = os.path.join(IDSSE_DIR, "cache")
RESULTS = os.path.join(IDSSE_DIR, "search.jsonl")
PITCH = (105.0, 68.0)    # every IDSSE match file reports this size
FPS = 25
MERGE_S = 0.5            # the second label version: team spells shorter than this are merged away.
                         # 2 s at first, which would also merge BAR-ATH's public 0.76 s and 1.32 s
                         # possessions; 0.5 s is below the shortest one.
SET_PIECE_KEEP = 0.55    # BAR-ATH has 59 tagged set pieces; IDSSE matches have about 100 restarts.
PLAYER_MISSING = 0.07    # the ATD tracks a mean of 20.5 of 22 players.
PLAYER_GAP_S = 3.0       # ponytail: mean player dropout, a guess; ATD slots swap too much to measure
                         # per person. It also ignores false team labels, which the ATD does have.
ERROR_TAU_S = 1.0        # position error is slow: the ATD ball is smooth frame to frame.
BALL_ERROR = 0.4         # m, per axis. An upper end. Net speed over 1 s barely changes the share of
                         # seen ball frames under 3 m/s on the ATD (31.1% -> 29.1%) or at 0.4 (58.7%
                         # -> 57.8%), but jumps at 0.8 (33.2% -> 52.3%), so the ATD does not have
                         # 0.8 m of slow error. Its low still share is not from error; see the log.
PLAYER_ERROR = 0.05      # m, per axis. Clean IDSSE player speeds already match the ATD (mean player
                         # speed p10/50/90: 0.66/1.55/2.95 vs 0.71/1.54/2.86). 0.3 gave p10 1.34.

# The knobs searched and their values. Whistle, tag and hold knobs are left out on purpose.
SPACE = {
    "R_CTRL": [1.5, 2.0, 2.5, 3.0],
    "V_PASS": [6.0, 8.0, 10.0, 12.0],
    "GAP_SCALE": [0.5, 1.0, 1.5, 2.0, 3.0],
    "W_CTRL": [0.5, 1.0, 2.0],
    "W_CARRY": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "CARRY_MAX": [1.0, 3.0, 5.0],
    "C_SWITCH": [8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    "C_SWITCH_TOUCH": [1.0, 3.0, 6.0],
    "C_TO_D": [10.0, 20.0, 30.0],
    "C_FROM_D": [5.0, 10.0, 15.0],
    "S0": [-3.0, -2.5, -2.0, -1.5],
    "W_SLOW": [0.5, 1.0, 2.0],
    "V_PLAY": [1.2, 1.5, 1.8],
    "W_STILL": [1.0, 2.0, 3.0],
    "V_STILL": [1.5, 3.0],
    "W_OUT": [2.0, 4.0, 6.0],
    "W_RESTART": [2.0, 4.0, 6.0],
    "TOUCH_TURN": [15.0, 25.0, 35.0],
    "R_TOUCH": [2.0, 3.0],
    "CA_MARGIN": [0.0, 0.5, 1.0, 1.5],
    "R_CLOSE": [0.75, 1.0, 1.5],
    "TOUCH_NEXT": [1.0, 3.0],
}

FRAMESET = re.compile(r'GameSection="([^"]+)".*TeamId="([^"]+)"')
PLAYER = re.compile(r'N="(\d+)".* X="([^"]+)" Y="([^"]+)"')
BALL = re.compile(r'N="(\d+)" T="([^"]+)" X="([^"]+)" Y="([^"]+)".*BallPossession="(\d)" BallStatus="(\d)"')
EVENT = re.compile(r'<Event [^>]*EventTime="([^"]+)"[^>]*>\s*<(FreeKick|CornerKick|ThrowIn|GoalKick|'
                   r'KickOff|Penalty|Foul|Offside)\b')
RESTARTS = {"FreeKick", "CornerKick", "ThrowIn", "GoalKick", "KickOff", "Penalty"}
WHISTLES = {"Foul", "Offside"}


def matches():
    return sorted(re.search(r"MAT-(\w+)\.xml", f)[1]
                  for f in glob.glob(os.path.join(IDSSE_DIR, "DFL_04_03_*.xml")))


def source_file(kind, match):
    return glob.glob(os.path.join(IDSSE_DIR, "DFL_{}_*{}.xml".format(kind, match)))[0]


def parse_match(match):
    """Per half: x, y (n_frames, n_players + 1; ball last, metres from the corner, like the ATD),
    labels, the home/away team of each player, and restart and whistle frames."""
    cache = os.path.join(CACHE_DIR, match + ".npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {s: {k: z[s + "/" + k] for k in ("x", "y", "label", "team", "restarts", "whistles")}
                for s in ("firstHalf", "secondHalf")}

    info = open(source_file("02_01", match), encoding="utf8").read()
    home = re.search(r'<Team [^>]*TeamId="([^"]+)"[^>]*Role="home"', info)[1]
    tracks, balls, section, team, person = {}, {"firstHalf": [], "secondHalf": []}, None, None, None
    with open(source_file("04_03", match), encoding="utf8") as handle:
        for line in handle:
            if "<FrameSet" in line:
                section, team = FRAMESET.search(line).groups()
                person = re.search(r'PersonId="([^"]+)"', line)[1]
                current = None
                if team == "referee":
                    continue
                if team != "BALL":
                    current = tracks.setdefault(section, {}).setdefault((team != home, person), [])
            elif "<Frame " in line:
                if team == "BALL":
                    m = BALL.search(line)
                    balls[section].append((int(m[1]), datetime.fromisoformat(m[2]).timestamp(),
                                           float(m[3]), float(m[4]), int(m[5]), int(m[6])))
                elif current is not None:
                    m = PLAYER.search(line)
                    current.append((int(m[1]), float(m[2]), float(m[3])))

    events = EVENT.findall(open(source_file("03_02", match), encoding="utf8").read())
    out = {}
    for section, rows in balls.items():
        rows = np.array(rows)
        live = np.flatnonzero(rows[:, 5] == 1)
        rows = rows[live[0]:live[-1] + 1]
        first, n = int(rows[0, 0]), len(rows)
        keys = sorted(tracks[section])
        x, y = np.full((2, n, len(keys) + 1), np.nan, dtype=np.float32)
        for slot, key in enumerate(keys):
            track = np.array(tracks[section][key])
            at = track[:, 0].astype(int) - first
            keep = (at >= 0) & (at < n)
            x[at[keep], slot], y[at[keep], slot] = track[keep, 1], track[keep, 2]
        x[:, -1], y[:, -1] = rows[:, 2], rows[:, 3]
        x += PITCH[0] / 2
        y += PITCH[1] / 2
        label = np.where(rows[:, 5] == 1, np.where(rows[:, 4] == 1, hmm.H, hmm.A), hmm.D).astype(np.int8)
        times = np.array([datetime.fromisoformat(t).timestamp() for t, _ in events])
        kinds = np.array([k for _, k in events])
        inside = (times >= rows[0, 1]) & (times <= rows[-1, 1])
        frames = np.searchsorted(rows[:, 1], times[inside])
        out[section] = {"x": x, "y": y, "label": label,
                        "team": np.array([int(away) for away, _ in keys], dtype=np.int8),
                        "restarts": frames[np.isin(kinds[inside], list(RESTARTS))],
                        "whistles": frames[np.isin(kinds[inside], list(WHISTLES))]}
    os.makedirs(CACHE_DIR, exist_ok=True)
    partial = "{}.{}.tmp.npz".format(cache[:-4], os.getpid())
    np.savez(partial, **{s + "/" + k: v for s, half in out.items() for k, v in half.items()})
    os.replace(partial, cache)
    return out


def parse_only(match):
    parse_match(match)


def on_off_runs(flag, blocks):
    """(off lengths, on lengths) of a per-frame flag, in frames, within each block."""
    off, on = [], []
    for first, last in blocks:
        edges = np.flatnonzero(np.diff(flag[first:last + 1].astype(np.int8))) + 1
        bounds = np.r_[0, edges, last + 1 - first]
        for a, b in zip(bounds[:-1], bounds[1:]):
            (on if flag[first + a] else off).append(b - a)
    return np.array(off), np.array(on)


def atd_ball_runs():
    """The ATD ball's (gap, visible) run lengths over the graded frames. No labels involved."""
    x, _, _ = read_positions(GAME)
    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    return on_off_runs(np.isfinite(x[:, -1]), blocks)


def renewal_mask(n, gaps, visible, rng):
    """True where seen: alternating runs drawn from the two length samples."""
    seen = np.zeros(n, dtype=bool)
    i, on = 0, bool(rng.random() < visible.sum() / (visible.sum() + gaps.sum()))
    while i < n:
        length = int(rng.choice(visible if on else gaps))
        seen[i:i + length] = on
        i, on = i + length, not on
    return seen


def slow_error(shape, sigma, rng):
    """AR(1) position error with standard deviation sigma and time constant ERROR_TAU_S."""
    rho = np.exp(-1.0 / (ERROR_TAU_S * FPS))
    out = np.empty(shape, dtype=np.float32)
    out[0] = rng.normal(0, sigma, shape[1:])
    step = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), shape)
    for i in range(1, shape[0]):
        out[i] = rho * out[i - 1] + step[i]
    return out


def degrade(half, gaps, visible, rng, ball_error=BALL_ERROR, player_error=PLAYER_ERROR):
    """Copy of half's x, y with the ATD's ball gaps, player dropouts and position error."""
    x, y = half["x"].copy(), half["y"].copy()
    n, slots = x.shape
    x += slow_error((n, slots), 1.0, rng) * np.r_[np.full(slots - 1, player_error), ball_error]
    y += slow_error((n, slots), 1.0, rng) * np.r_[np.full(slots - 1, player_error), ball_error]
    seen = renewal_mask(n, gaps, visible, rng)
    x[~seen, -1] = y[~seen, -1] = np.nan
    off = np.array([PLAYER_GAP_S * FPS])
    on = np.array([PLAYER_GAP_S * FPS * (1 - PLAYER_MISSING) / PLAYER_MISSING])
    for slot in range(slots - 1):
        lost = ~renewal_mask(n, off, on, rng)
        x[lost, slot] = y[lost, slot] = np.nan
    return x, y


def signals(half, rng):
    """Event stand-ins for the automatic signals: (soon, phase, set_starts, whistled)."""
    n = len(half["x"])
    starts = half["restarts"][rng.random(len(half["restarts"])) < SET_PIECE_KEEP]
    pre = int(round(hmm.PARAMS["T_PRE"] * FPS))
    soon = np.zeros(n, dtype=bool)
    for s in starts:
        soon[max(0, s - pre):s] = True
    return soon, np.zeros(n, dtype=np.float32), [int(s) for s in starts], [int(w) for w in half["whistles"]]


def merge_short(label, min_s=MERGE_S):
    """Labels with team spells under min_s removed when the same team holds on both sides."""
    out = label.copy()
    cuts = np.flatnonzero(label[1:] != label[:-1]) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(label)]
    for i in range(1, len(starts) - 1):
        if (label[starts[i]] != hmm.D and (ends[i] - starts[i]) / FPS < min_s
                and label[starts[i - 1]] == label[starts[i + 1]] != hmm.D):
            out[starts[i]:ends[i]] = label[starts[i - 1]]
    return out


def score(truths, preds):
    """Spell F1 and macro F1 over halves; the halves are kept apart by an X frame."""
    join = lambda parts: np.concatenate([np.r_[np.asarray(a, dtype=np.int8), [pm.X]] for a in parts])
    gt, pred = join(truths), join(preds)
    pred[gt == pm.X] = pm.X
    runs = pm.valid_runs(gt)
    return {"spell": pm.spell_scores(gt, pred, runs)["mean_f1"],
            "macro": pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))),
            "n_pred": len(pm.possession_spells(pred, runs)),
            "n_gt": len(pm.possession_spells(gt, runs)),
            "dead": float(np.mean(pred[gt != pm.X] == pm.D))}


def attack_of(x, team):
    """+1 where H attacks +x: home players average on the low-x side of the halfway line."""
    home, away = np.nanmean(x[:, :-1][:, team == hmm.H]), np.nanmean(x[:, :-1][:, team == hmm.A])
    return np.full(len(x), 1 if home < away else -1, dtype=np.int8)


def prepared(match, runs, seed=0, clean=False):
    """One match, ready to decode. Degrading and event thinning are seeded by match and seed, so
    every setting sees the same data."""
    rng = np.random.default_rng([seed, int(match, 36)])
    out = []
    for half in parse_match(match).values():
        x, y = (half["x"], half["y"]) if clean else degrade(half, *runs, rng)
        out.append(dict(x=x, y=y, team=half["team"], label=half["label"], signals=signals(half, rng)))
    return out


def run_setting(p, halves):
    preds = []
    for half in halves:
        x, team = half["x"], half["team"]
        soon, phase, starts, whistled = half["signals"]
        labels, _ = hmm.decode(x, half["y"], team, soon, phase, starts, whistled, attack_of(x, team),
                               [(0, len(x) - 1)], p, PITCH, FPS)
        preds.append(labels)
    raw = [h["label"] for h in halves]
    return {"raw": score(raw, preds), "merged": score([merge_short(t) for t in raw], preds)}


def calibrate():
    """Nearest-player distance and ball speed on the ATD, clean IDSSE and degraded IDSSE."""
    def stats(x, y):
        vx, vy = hmm.velocity(x, y, FPS)
        distance = np.nanmin(np.hypot(x[:, :-1] - x[:, -1:], y[:, :-1] - y[:, -1:]), axis=1)
        speed = np.hypot(vx[:, -1], vy[:, -1])
        return (np.round(np.nanpercentile(distance, [25, 50, 75]), 2).tolist(),
                np.round(np.nanpercentile(speed, [50, 90, 99]), 1).tolist(),
                round(float(np.isfinite(x[:, -1]).mean()), 3),
                round(float(np.isfinite(x[:, :-1]).sum(axis=1).mean()), 1))

    x, y, _ = read_positions(GAME)
    keep = np.zeros(len(x), dtype=bool)
    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    for a, b in blocks:
        keep[a:b + 1] = True
    print("{:<24} near-player p25/50/75, ball speed p50/90/99, ball seen, players".format(""))
    print("{:<24} {}".format("ATD", stats(x[keep], y[keep])))
    runs = on_off_runs(np.isfinite(x[:, -1]), blocks)
    half = parse_match(matches()[0])["firstHalf"]
    print("{:<24} {}".format("IDSSE clean", stats(half["x"], half["y"])))
    for ball, player in ((0.4, 0.2), (0.8, 0.3), (1.2, 0.3), (1.6, 0.5)):
        rng = np.random.default_rng(0)
        print("{:<24} {}".format("error {} {}".format(ball, player),
                                 stats(*degrade(half, *runs, rng, ball, player))))


_WORKER = {}


def worker(task):
    index, values, match, runs, seed = task
    if match not in _WORKER:
        _WORKER[match] = prepared(match, runs, seed)
    result = run_setting(dict(hmm.PARAMS, **values), _WORKER[match])
    return dict(index=index, match=match, seed=seed, values=values, **result)


def search(n, seed):
    rng = np.random.default_rng(seed)
    settings = [{}]                # the default first
    for _ in range(n):
        settings.append({k: float(rng.choice(v)) for k, v in SPACE.items()})
    done = set()
    if os.path.exists(RESULTS):
        done = {(r["index"], r["match"]) for r in map(json.loads, open(RESULTS))}
    runs = atd_ball_runs()
    tasks = [(i, s, m, runs, 0) for i, s in enumerate(settings) for m in matches() if (i, m) not in done]
    print("{} tasks".format(len(tasks)), flush=True)
    with mp.Pool(max(1, os.cpu_count() - 1)) as pool, open(RESULTS, "a") as out:
        pool.map(parse_only, matches(), chunksize=1)     # fill the cache before workers read it
        for k, row in enumerate(pool.imap_unordered(worker, tasks)):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if k % 70 == 0:
                print(k, flush=True)


def load_results():
    table, values = {}, {}
    for row in map(json.loads, open(RESULTS)):
        table.setdefault(row["index"], {})[row["match"]] = row
        values[row["index"]] = row["values"]
    names = matches()
    keep = sorted(i for i, t in table.items() if len(t) == len(names))
    grid = {kind: np.array([[table[i][m][kind]["spell"] for m in names] for i in keep])
            for kind in ("raw", "merged")}
    macro = np.array([[table[i][m]["raw"]["macro"] for m in names] for i in keep])
    return names, keep, [values[i] for i in keep], grid, macro, table


def choose(raw, merged):
    """Index of the best setting on the given rows' columns: highest mean of the two label versions
    among settings beating the default (row 0) on both; 0 if none do."""
    r, m = raw.mean(axis=1), merged.mean(axis=1)
    fair = (r > r[0]) & (m > m[0])
    if not fair.any():
        return 0
    return int(np.argmax(np.where(fair, (r + m) / 2, -np.inf)))


def report():
    names, keep, values, grid, macro, table = load_results()
    raw, merged = grid["raw"], grid["merged"]
    print("{} settings over {} matches; the default is row 0\n".format(len(keep), len(names)))
    default = table[keep[0]]
    print("default: mean spellF1 raw {:.3f} merged {:.3f} macroF1 {:.3f}; possessions per match "
          "predicted {:.0f} vs DFL {:.0f}; D {:.1%}\n".format(
              raw[0].mean(), merged[0].mean(), macro[0].mean(),
              np.mean([default[m]["raw"]["n_pred"] for m in names]),
              np.mean([default[m]["raw"]["n_gt"] for m in names]),
              np.mean([default[m]["raw"]["dead"] for m in names])))

    print("leave one match out: choose on six, score on the seventh")
    gains = []
    for k, m in enumerate(names):
        train = [j for j in range(len(names)) if j != k]
        best = choose(raw[:, train], merged[:, train])
        gains.append((raw[best, k] - raw[0, k], merged[best, k] - merged[0, k]))
        print("  {}: setting {:>4}  raw {:.3f} -> {:.3f}  merged {:.3f} -> {:.3f}".format(
            m, keep[best], raw[0, k], raw[best, k], merged[0, k], merged[best, k]))
    gains = np.array(gains)
    print("  mean held-out gain: raw {:+.3f}, merged {:+.3f}; matches improved raw {}/{}, merged {}/{}".format(
        *gains.mean(axis=0), (gains[:, 0] > 0).sum(), len(names), (gains[:, 1] > 0).sum(), len(names)))

    best = choose(raw, merged)
    print("\nchosen on all seven: setting {}  raw {:.3f} merged {:.3f} macroF1 {:.3f}".format(
        keep[best], raw[best].mean(), merged[best].mean(), macro[best].mean()))
    print("  " + " ".join("{}={:g}".format(k, v) for k, v in sorted(values[best].items())
                          if v != hmm.PARAMS[k]))

    print("\nmean spellF1 by knob value over all random settings (raw / merged):")
    for knob, options in SPACE.items():
        cells = []
        for option in options:
            rows = [j for j in range(1, len(keep)) if values[j].get(knob) == option]
            if rows:
                cells.append("{:g}: {:.3f}/{:.3f}".format(option, raw[rows].mean(), merged[rows].mean()))
        print("  {:<15} default {:<6g} {}".format(knob, hmm.PARAMS[knob], "  ".join(cells)))


TEAM_FRAMES = 6000         # live frames sampled per half for the team-probability fit


def team_rows(match, runs, seed):
    """(halves, X, y): degraded halves of one match and a sample of their live frames' design rows."""
    import team_prob
    rng = np.random.default_rng([seed, int(match, 36), 1])
    halves, Xs, ys = prepared(match, runs, seed), [], []
    for h in halves:
        live = np.flatnonzero(h["label"] != hmm.D)
        keep = rng.choice(live, min(len(live), TEAM_FRAMES), replace=False)
        Xs.append(team_prob.design(h["x"], h["y"], h["team"])[keep])
        ys.append(h["label"][keep] == hmm.H)
    return halves, np.vstack(Xs), np.concatenate(ys)


def team_data(seed):
    runs = atd_ball_runs()
    with mp.Pool(min(7, os.cpu_count())) as pool:
        return dict(zip(matches(), pool.starmap(team_rows, [(m, runs, seed) for m in matches()])))


def fit_team(seed):
    import team_prob
    data = team_data(seed)
    w = team_prob.fit(np.vstack([d[1] for d in data.values()]), np.concatenate([d[2] for d in data.values()]))
    json.dump({"w": w.tolist(), "matches": matches(), "degrade_seed": seed, "frames_per_half": TEAM_FRAMES,
               "source": "IDSSE, CC-BY 4.0, Bassek et al. 2025; fitted by tune_idsse.py fit-team"},
              open(team_prob.WEIGHTS, "w"), indent=0)
    print("wrote", team_prob.WEIGHTS)


def team_check(seed):
    """Leave one match out, team_prob weights fitted on the six training matches: the HMM with
    nearest-player evidence against W_PROB, on the same degraded data."""
    import team_prob
    data = team_data(seed)
    settings = {"nearest-player": dict(hmm.PARAMS, W_PROB=0.0, W_CTRL=1.0),
                "W_PROB": dict(hmm.PARAMS, W_PROB=1.0, W_CTRL=0.0)}
    rows = []
    for test in matches():
        train = [m for m in matches() if m != test]
        team_prob._W = team_prob.fit(np.vstack([data[m][1] for m in train]), np.concatenate([data[m][2] for m in train]))
        r = {k: run_setting(v, data[test][0]) for k, v in settings.items()}
        rows.append([[r[k][v]["spell"] for v in ("raw", "merged")] for k in settings])
        print(test, " | ".join("{} raw {:.3f} merged {:.3f} macro {:.3f}".format(
            k, r[k]["raw"]["spell"], r[k]["merged"]["spell"], r[k]["raw"]["macro"]) for k in settings), flush=True)
    a = np.array(rows)
    for i, k in enumerate(settings):
        g = a[:, i] - a[:, 0]
        print("{:<15} raw {:.3f} merged {:.3f} | against nearest-player raw {:+.3f} ({}/7), merged {:+.3f} ({}/7)".format(
            k, *a[:, i].mean(0), g[:, 0].mean(), (g[:, 0] > 0).sum(), g[:, 1].mean(), (g[:, 1] > 0).sum()))
    team_prob._W = None


def self_check():
    """Known answers for the parts with logic. Needs no data."""
    a, b = hmm.H, hmm.A
    label = np.array([a] * 100 + [b] * 10 + [a] * 115 + [hmm.D] * 10 + [b] * 10 + [a] * 115, dtype=np.int8)
    merged = merge_short(label)
    assert (merged[:225] == a).all(), "a 0.4 s spell inside one team's possession stays"
    assert (merged[235:245] == b).all(), "a short spell after a stoppage is a real possession"

    rng = np.random.default_rng(0)
    seen = renewal_mask(200000, np.array([100]), np.array([300]), rng)
    assert abs(seen.mean() - 0.75) < 0.03
    off, on = on_off_runs(seen, [(0, len(seen) - 1)])
    assert np.median(off[1:]) == 100 and np.median(on[1:]) == 300

    error = slow_error((50000, 1), 2.0, rng)
    assert abs(error.std() - 2.0) < 0.3

    perfect = score([label], [label])
    assert perfect["spell"] == 1.0 and perfect["macro"] == 1.0

    raw = np.array([[0.5, 0.5], [0.6, 0.6], [0.9, 0.9]])
    merged = np.array([[0.5, 0.5], [0.6, 0.6], [0.4, 0.4]])
    assert choose(raw, merged) == 1, "a setting must beat the default on both label versions"
    assert choose(np.array([[0.5], [0.4]]), np.array([[0.5], [0.9]])) == 0

    line = '<Frame N="10" T="2022-10-15T12:19:58.160+00:00" X="1.50" Y="-4.64" Z="3.76" D="1" S="2" A="0" M="62" BallPossession="2" BallStatus="1"/>'
    m = BALL.search(line)
    assert m[1] == "10" and m[3] == "1.50" and m[5] == "2" and m[6] == "1"
    assert PLAYER.search(line).groups() == ("10", "1.50", "-4.64")


if __name__ == "__main__":
    self_check()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["calibrate", "search", "report", "fit-team", "team-check", "check"])
    parser.add_argument("--n", type=int, default=300, help="random settings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate()
    elif args.command == "search":
        search(args.n, args.seed)
    elif args.command == "report":
        report()
    elif args.command == "fit-team":
        fit_team(args.seed)
    elif args.command == "team-check":
        team_check(args.seed)
    else:
        print("self-check ok")
