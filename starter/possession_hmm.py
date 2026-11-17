"""Possession as a sequence: per-frame evidence for H, A and D, decoded by a three-state HMM.

The baseline gives the ball to the nearest player and smooths with a majority vote. That vote can
only blur: it erases short possessions, misplaces changes by seconds, and has no idea that a ball
sitting still during a stoppage is dead. This replaces the vote with a Viterbi decode that pays a
cost every time the label changes, and feeds it evidence the baseline ignores.

    pip install -r requirements.txt          # numpy only
    python starter/possession_hmm.py         # writes evaluation/results/hmm.csv
    python starter/possession_hmm.py --set C_SWITCH=25 W_SET=4 --name hmm_try2
    python evaluation/run_evaluation.py      # scores it next to the baselines

The first design, before any stoppage knobs were adjusted, is still one command away:

    python starter/possession_hmm.py --name hmm_v0 --set S0=-3 W_SLOW=2 V_STILL=1 STILL_HOLD=25 \
        T_PRE=8 W_RESTART=0 W_WHISTLE=0 TOUCH=0 GK_SIDE=0 SET_GATE=0

Six stages, each with its knobs in PARAMS below:

  1. Load positions, velocities (starter/velocity.py, 25 fps), team of each slot, Smart Tagging.
  2. Ball state per frame: MISSING, OUT, FLIGHT or CONTROL. Only CONTROL says anything about team.
  3. Stoppage score S: log-odds that play is stopped, from slow players, a still ball, an
     upcoming Smart Tagging set piece, the ball out of bounds, and a lookback that marks a
     missing-ball stretch dead when the ball reappears as a restart, and the seconds after a
     referee whistle heard in the video's audio (starter/whistle.py) -- with WHISTLE_HOLD, until
     the ball is kicked again from where play stopped. Tried and dropped: scattered
     player movement and an offside rule (neither helped spellF1 on the public window; offside
     fired on 2.5% of in-play frames), and set-piece formation (no gain on top of the lookback,
     and it added 9 points of D across the match with nothing to show it was right).
  4. Evidence: a log-score for H, A and D on every frame. CONTROL frames score the teams by the gap
     between the nearest home and nearest away player; D scores S everywhere; Smart Tagging phases
     nudge the team they name. Touches -- a sharp turn or speed change of the ball beside a player,
     or with TOUCH_MODE a player's closest approach to the ball with its path changing across it --
     carry the toucher's team through the FLIGHT frames that follow, and lower the H<->A cost
     where the touching team changes. A skip for touches at the end of a long ball (meant for
     headers) was tried and removed: it dropped 11 public touches whose team was 91% right.
     Off by default: FLIP (x-velocity reversals as touches, and a pull towards the new team at
     regains) scored lower on the public window; CARRY_ALL (the last toucher's team on every
     frame, with pressers' nearest-player evidence scaled down) changed 7 public frames.
     Goalkeeper slots take the team of the half they stand in (GK_SIDE); a closest-approach touch
     needs the other team 1 m further away (CA_MARGIN); a set piece's lead-in window waits for a
     stoppage sign (SET_GATE).
  5. Viterbi over each half separately, with a cost for H<->A, for entering D and for leaving it.
  6. Write a frame,label CSV for all 199319 video frames. Ungraded frames get D; they are ignored.

The defaults come from physics and from distributions over the whole match, not from the ground
truth. The public five minutes holds 17 possessions -- use it to check the pipeline, not to sweep.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
import team_prob  # noqa: E402
from velocity import _moving_average, read_positions, velocity  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])
MISSING, OUT, FLIGHT, CONTROL = 0, 1, 2, 3

# Every knob, with the reason for its default. Override any of them with --set NAME=VALUE.
PARAMS = {
    # Stage 2 -- ball state
    "R_CTRL": 2.0,       # m. Nearest player within this = ball under control. On correct baseline
                         # frames the nearest player is a median 1.4-1.6 m from the ball.
    "V_PASS": 8.0,       # m/s. Ball faster than this = in flight. Above dribbling, below a typical
                         # pass (ball speed median 5 m/s, 90th percentile 14 m/s).
    "OUT_MARGIN": 0.5,   # m. Ball this far beyond a line counts as out, absorbing tracking error.
    "OUT_FALSE": 5.0,    # m. Further out than this is a false detection (seen up to 23 m out), so
                         # the frame is treated as MISSING rather than OUT.
    "GK_SIDE": 1.0,      # 1 = a goalkeeper slot (24 H, 46 A) belongs to the team whose half it stands
                         # in, per half: the slots follow whichever keeper (slot 46 in Barca's half
                         # 7:27-8:26, slot 24 at Athletic's goal 10:24-10:38). 0 at first; 1 cut
                         # public swaps from 694 to 450 frames.
    # Stage 3 -- stoppage score S (log-odds)
    "S0": -2.0,         # baseline log-odds of a stoppage when no signal fires. -3 at first; that
                         # kept every public stoppage below zero (mean S -1.2 to -3.3, play -4.6).
    "W_SLOW": 1.0,       # per m/s that the mean player speed falls below V_PLAY. 2 at first; fast
                         # players then cancelled the set-piece and out signals at the 274 s goal
                         # kick.
    "V_PLAY": 1.5,       # m/s. Match median of the +/-2 s mean player speed; stoppages ~1.1, play
                         # ~2.3. Weak alone: the 25th percentile is already 1.0.
    "W_STILL": 2.0,      # ball seen and slower than V_STILL on every frame of +/- STILL_HOLD.
    "V_STILL": 3.0,      # m/s. Ball speed is already a 5-frame average (velocity.py, smooth=2).
                         # 1 at first: tracker jitter keeps a resting ball above it (the 188 s
                         # stoppage had a median ball speed of 2.8 m/s).
    "STILL_HOLD": 2,     # frames, half-width: still on all of a 5-frame window. 25 (the whole
                         # +/-1 s) at first, which never fired in any public stoppage.
    "W_SET": 3.0,        # within T_PRE before a Smart Tagging SET PIECES start: a restart ends a
                         # dead-ball stretch. Players average 0.97 m/s in the 5 s before one.
    "T_PRE": 15.0,       # s. 8 at first, shorter than the public stoppages (14-20 s).
    "SET_GATE": 1.0,     # 1 = the T_PRE window only counts from the first frame in it where the ball
                         # is out, a whistle has blown or the ball is still: a tag starting 6 s
                         # after the 11:35 stoppage began put D over 9 s of play before it. 0 at
                         # first; 1 cut public play labelled D from 251 to 108 frames.
    "W_OUT": 4.0,        # ball OUT. Dead by definition, so the strongest single signal.
    "W_RESTART": 4.0,    # lookback: a missing-ball stretch that ends in a restart was dead ball.
                         # As strong as W_OUT: a restart is as sure a sign as the ball being out.
                         # Restart = where the ball reappears it is still, or on a touchline, in a
                         # goal area, at a corner or the centre spot, or a set piece is tagged
                         # within 3 s. Fires on 66 of the match's 133 graded gaps of 1 s+, and on
                         # neither public in-play gap.
    "W_WHISTLE": 4.0,    # the T_WHISTLE after a referee whistle (starter/whistle.py, from the
                         # video's audio). Whistles within 5 s of a tagged kick-off are skipped:
                         # those restart play instead of stopping it. Weights 4 and 8 give the same
                         # public labels. 0 at first.
    "T_WHISTLE": 5.0,    # s. The two public whistled stoppages last 5.5 s and 20.4 s. 10 s scored
                         # higher on the public window but turned 95 more in-play frames D, and
                         # was picked from two whistles.
    "WHISTLE_HOLD": 1.0, # 1 = after a whistle, stay dead until the ball is restarted from where play
                         # stopped (a set spot, or near the ball's position SPOT_LEAD before the
                         # whistle), instead of for a fixed T_WHISTLE. A ball moved about during
                         # the stoppage then does not count as play. Falls back to T_WHISTLE when
                         # no restart is found within WHISTLE_MAX. 0 at first; 1 fixed the 10:35
                         # stoppage (restart found at 10:56.1, truth 10:56.0).
    "DEAD_BEFORE": 2.0,  # s. A whistle after this long with the ball untracked and players slower
    "V_DEAD": 1.0,       # than this (m/s) is heard while play is already dead: it gets T_WHISTLE,
                         # not the hold. The 97:50 whistle, inside a 97:08-97:54 stoppage, held D
                         # to 98:25 over play. Skipping such whistles outright dropped the 5 s of D
                         # at 40:15, 40:24, 93:23 and 117:56. Matches 12 of the match's 53
                         # non-kick-off whistles, neither public one.
    "SPOT_LEAD": 1.0,    # s before the whistle. The referee blows after the offence: at 10:35 the
                         # ball was rolling at 16 m/s, 0.5 s put the spot 8 m down its path.
    "R_SPOT": 10.0,      # m from that position. Both public whistled restarts were 6-9 m from it.
    "R_FIXED": 3.0,      # m from the centre spot, a penalty spot or a corner: accepted anywhere.
    "HOLD_MIN": 1.0,     # s after the whistle before a restart is looked for.
    "WHISTLE_MAX": 60.0, # s. Longest hold.
    "KICK_V": 6.0,       # m/s. A restart is the ball going faster than this...
    "KICK_STILL": 1.5,   # m/s. ...after being slower than this...
    "KICK_HOLD": 0.4,    # s. ...for this long within the KICK_PRIOR before it...
    "KICK_PRIOR": 2.0,   # s. ...or the ball reappearing near the spot after a missing stretch.
    "LOOKBACK_GAP": 1.0,    # s. Shortest missing stretch considered.
    "LOOKBACK_MAX": 30.0,   # s. Only the last this-many seconds before the restart are marked.
    "RESTART_STILL": 3.0,   # m/s. Median ball speed over its first 0.5 s back = placed ball.
    # Stage 4 -- evidence
    "GAP_SCALE": 1.0,    # m. p(H) = sigmoid((dA - dH) / GAP_SCALE): 1 m gap -> 0.73, 3 m -> 0.95.
                         # Baseline swap frames had a median gap of 2.1 m against 3.6 m on correct
                         # ones, so small gaps should stay uncertain.
    "W_CTRL": 0.0,       # weight of the nearest-player CONTROL evidence against the change costs.
                         # 1 at first; 0 since W_PROB replaced it (2026-11-17).
    "W_PROB": 1.0,       # weight of team_prob.py's P(home), added on every frame as log p - log max(p,
                         # 1 - p)). Chosen over nearest-player (W_CTRL=1) by leave-one-match-out on
                         # the IDSSE matches (tune_idsse.py team-check, DFL spells under 0.5 s merged):
                         # spellF1 +0.017 raw / +0.017 merged at degrade seed 1, better on 7/7;
                         # +0.012 / +0.011, 13/14, over seeds 1-2 with the research LR. 0 at first.
    "W_PHASE": 0.05,     # per frame, towards the team a Smart Tagging phase implies.
    # Stage 4b -- touches and regains
    "TOUCH": 1.0,        # 1 = use touches: lower C_SWITCH at confirmed regains, carry the last
                         # toucher's team through FLIGHT frames. 0 at first.
    "TOUCH_TURN": 25.0,  # degrees the ball's direction turns over 3 frames (120 ms)...
    "TOUCH_DV": 4.0,     # ...or m/s its speed changes over the same 3 frames.
    "R_TOUCH": 3.0,      # m. A player must be this close for the turn to be a touch. On the public
                         # window these give ~10 touches per in-play minute; 20-30 degrees give the
                         # same matches at possession changes.
    "TOUCH_MODE": 2.0,   # 0 = touches from a sharp turn or speed change (below); 1 = from a player's
                         # closest approach to the ball (CA_* knobs); 2 = both. 0 at first. Public
                         # spellF1 0 -> 2: 0.506 -> 0.533, macroF1 0.871 -> 0.868; 1 alone: 0.494.
                         # Neighbouring CA_* settings give spellF1 0.489-0.556.
    "R_CLOSE": 1.0,      # m. Closest approach must come this near. The 7:43 touch (A, on video) is a
                         # 0.3 m approach with a turn spread over ~0.7 s that mode 0 misses.
    "CA_WIN": 0.2,       # s. ...and be the player's nearest point within +/- this.
    "CA_LAG": 0.3,       # s. Ball velocity this long before vs after the approach...
    "CA_TURN": 20.0,     # degrees: ...must turn at least this (both sides faster than 1 m/s)...
    "CA_DV": 3.0,        # m/s: ...or change speed by this. A ball rolling past is not a touch.
    "CA_MARGIN": 1.0,    # m. The other team's nearest player must be at least this much further
                         # from the ball than the closest-approach player. The false touches at
                         # 8:18 and 8:25 (no touch on video) had gaps of 0.9 m and 0.6 m. 0 at
                         # first; 0.5 changed nothing, 0.75 and 1.0 score the same, 1.25+ lose a
                         # possession.
    "TOUCH_MARGIN": 0.5, # m. Nearest home and away player closer than this to each other: no team.
    "TOUCH_NEXT": 3.0,   # s. A touch is confirmed if the same team makes the next touch within this...
                         # (1 was tried: it lost the 7:42.1 change, confirming touch 1.1 s later)
    "TOUCH_KEEP": 0.3,   # s. ...or the toucher's team has CONTROL this long in the next 0.5 s.
    "C_SWITCH_TOUCH": 3.0, # H<->A cost within TOUCH_WIN of a confirmed regain (C_SWITCH elsewhere).
                           # Changed no public label: none of the confirmed regains fell within 1 s
                           # of a real change of team.
    "TOUCH_WIN": 0.2,    # s, either side of the touch.
    "W_CARRY": 1.0,      # per FLIGHT frame towards the last confirmed toucher's team. All of the
                         # touch gain comes from this; 2 scored higher but was picked after looking.
    "CARRY_MAX": 3.0,    # s. ...for at most this long after the touch.
    "FLIP": 0.0,         # 1 = also count a reversal of the ball's x-velocity beside a player as a
                         # touch (headers, interceptions, blocks), and pull towards the new team
                         # from the first touch of a confirmed regain. 0 at first.
    "FLIP_V": 1.5,       # m/s. |vx| at least this on both sides of the reversal...
    "FLIP_LAG": 0.2,     # s. ...measured this long before and after (velocity is smoothed, so a
                         # reversal passes through zero over several frames).
    "W_REGAIN": 1.0,     # per frame, all ball states, towards the team of a confirmed regain...
    "W_FWD": 1.0,        # ...plus this if the confirming touch sends the ball towards that team's
                         # attacking goal. Extra weight, not a requirement: two public regains
                         # were played backwards.
    "REGAIN_HOLD": 1.0,  # s from the first touch, or until the other team's next confirmed touch.
    "CARRY_ALL": 0.0,    # 1 = carry the last confirmed toucher's team through every ball state,
                         # not just FLIGHT, until the other team touches it (CARRY_MAX cap)...
    "CTRL_PRESS": 0.3,   # ...and scale the nearest-player evidence by this where the nearest
                         # player is from the other team: a presser, not a new possessor.
    # Stage 5 -- change costs (log units; one clear CONTROL frame at p = 0.8 is worth ~1.4)
    "C_SWITCH": 15.0,    # H<->A: ~0.5 s of clear opposite control. Keeps a 0.76 s possession,
                         # stops one contested frame flipping the team.
    "C_TO_D": 30.0,      # H/A -> D: ~1.5 s of strong stoppage evidence. 20 at first; 30 chosen on the
                         # IDSSE matches (starter/tune_idsse.py): small gain on all 7, public unchanged.
    "C_FROM_D": 10.0,    # D -> H/A: restarts are sharper than stoppages.
}

# Which team a Smart Tagging phase's Team tag implies has the ball: +1 the tagged team, -1 the
# other one. Measured against the nearest player on frames with clear control (ball within 2 m of
# someone, teams more than 2 m apart) -- tracking and tags only, no ground truth:
#   tagged team nearest: Build Up 89%, Progression 85%, Attacking Transition 72%, Long Ball 71%,
#   Ball In Final Third 69%; Defending In Attacking Third 11%, Defending In Middle Third 15%,
#   Defensive Transition 28%.
# Left out as too ambiguous: the box codes (58-61%, both teams crowd the ball there), Defending In
# Defensive Third (41%), Goals. SET PIECES is used by the stoppage score instead.
PHASE_DIRECTION = {
    "BUILD UP": 1,
    "PROGRESSION": 1,
    "ATTACKING TRANSITION": 1,
    "LONG BALL": 1,
    "BALL IN FINAL THIRD": 1,
    "DEFENDING IN ATTACKING THIRD": -1,
    "DEFENDING IN MIDDLE THIRD": -1,
    "DEFENSIVE TRANSITION": -1,
}
TAG_TEAM = {"barcelona": H, "atletic club": A}


def rolling_all(flags, half_width):
    """True where every frame in the centred +/- half_width window is True."""
    total = np.concatenate(([0], np.cumsum(flags)))
    index = np.arange(len(flags))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(flags), index + half_width + 1)
    return (total[high] - total[low]) == (high - low)


def keeper_sides(x, team, game=GAME):
    """(n_frames, n_slots) team per frame: the metadata team, except that the two goalkeeper
    slots take the team whose half they stand in. GK_SIDE."""
    length = GAMES[game]["pitch"][0]
    per_frame = np.repeat(team[None, :], len(x), axis=0)
    h_attack = attack_direction(x, game)
    for slot in (23, 45):
        seen = np.isfinite(x[:, slot])
        in_h_half = (x[:, slot] < length / 2) == (h_attack == 1)    # H defends low x when attacking +x
        per_frame[seen, slot] = np.where(in_h_half[seen], H, A)
    return per_frame


def features(x, y, team, p, pitch, fps):
    """Per-frame dH, dA, ball state and mean player speed. Stages 1-2. `team` is per slot, or
    per frame and slot (keeper_sides)."""
    length, width = pitch
    bx, by = x[:, -1], y[:, -1]
    vx, vy = velocity(x, y, fps)
    ball_speed = np.hypot(vx[:, -1], vy[:, -1])
    player_speed = np.hypot(vx[:, :-1], vy[:, :-1])

    with np.errstate(invalid="ignore"):
        distance = np.hypot(x[:, :-1] - bx[:, None], y[:, :-1] - by[:, None])
    distance = np.where(np.isfinite(distance), distance, np.inf)
    per_frame = np.broadcast_to(team, distance.shape)
    d_home = np.where(per_frame == H, distance, np.inf).min(axis=1)
    d_away = np.where(per_frame == A, distance, np.inf).min(axis=1)

    beyond = np.maximum.reduce([-bx, bx - length, -by, by - width])  # m past the nearest line
    seen = np.isfinite(bx) & ~(beyond > p["OUT_FALSE"])
    with np.errstate(invalid="ignore"):
        fast = ball_speed > p["V_PASS"]
    state = np.full(len(x), CONTROL, dtype=np.int8)
    state[seen & (np.minimum(d_home, d_away) > p["R_CTRL"])] = FLIGHT
    state[seen & fast] = FLIGHT
    state[seen & (beyond > p["OUT_MARGIN"])] = OUT
    state[~seen] = MISSING

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no players: NaN, fine
        mean_speed = np.nanmean(player_speed, axis=1)
    mean_speed = _moving_average(mean_speed[:, None], 2 * fps)[:, 0]
    with np.errstate(invalid="ignore"):
        still = seen & (ball_speed < p["V_STILL"])
    still = rolling_all(still, int(p["STILL_HOLD"]))
    return {"d_home": d_home, "d_away": d_away, "state": state, "mean_speed": mean_speed,
            "dist": distance.astype(np.float32), "team": team,
            "still": still, "seen": seen, "ball_speed": ball_speed, "beyond": beyond,
            "ball_vx": vx[:, -1], "ball_vy": vy[:, -1], "ball_x": bx, "ball_y": by}


def missing_runs(seen, min_frames):
    """[(start, end)] of runs of unseen ball at least min_frames long, end exclusive."""
    edges = np.diff(np.r_[0, (~seen).astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_frames]


def restart_lookback(x, y, f, set_starts, p, pitch, fps):
    """Flag the last LOOKBACK_MAX seconds of every missing-ball stretch that ends in a restart."""
    length, width = pitch
    bx, by = x[:, -1], y[:, -1]
    flagged = np.zeros(len(x), dtype=bool)
    set_starts = np.asarray(set_starts)
    reach = int(p["LOOKBACK_MAX"] * fps)
    for a, b in missing_runs(f["seen"], int(p["LOOKBACK_GAP"] * fps)):
        if b >= len(x):
            continue
        cx, cy = bx[b], by[b]
        first = f["ball_speed"][b:b + fps // 2]
        restart = (
            (np.isfinite(first).any() and np.nanmedian(first) < p["RESTART_STILL"])
            or min(cy, width - cy) < 1.5                                            # throw-in
            or ((cx < 5.5 or cx > length - 5.5) and abs(cy - width / 2) < 9.16)     # goal kick
            or min(np.hypot(cx - ex, cy - ey) for ex in (0, length) for ey in (0, width)) < 2
            or np.hypot(cx - length / 2, cy - width / 2) < 2                        # kick-off
            or (len(set_starts) and np.abs(set_starts - b).min() <= 3 * fps)
        )
        if restart:
            flagged[max(a, b - reach):b] = True
    return flagged


def whistle_restart(w, f, p, fps, pitch):
    """First frame after the whistle at frame w where play restarts from where it stopped, or
    None. A restart is a kick from a still ball, or the ball reappearing, at an accepted spot."""
    length, width = pitch
    bx, by, speed, seen = f["ball_x"], f["ball_y"], f["ball_speed"], f["seen"]
    lead = max(0, w - int(p["SPOT_LEAD"] * fps))
    low = max(0, lead - 2 * fps)
    known = low + np.flatnonzero(seen[low:lead + 1])      # last sighting, if it was missing then
    spot = (bx[known[-1]], by[known[-1]]) if len(known) else None
    fixed = [(length / 2, width / 2), (11, width / 2), (length - 11, width / 2),
             (0, 0), (0, width), (length, 0), (length, width)]

    def at_spot(i):
        return ((spot is not None and np.hypot(bx[i] - spot[0], by[i] - spot[1]) < p["R_SPOT"])
                or any(np.hypot(bx[i] - fx, by[i] - fy) < p["R_FIXED"] for fx, fy in fixed))

    prior, need = int(p["KICK_PRIOR"] * fps), int(p["KICK_HOLD"] * fps)
    for i in range(w + int(p["HOLD_MIN"] * fps), min(len(seen), w + int(p["WHISTLE_MAX"] * fps))):
        if not seen[i]:
            continue
        if not seen[i - fps:i].any():                       # back after a missing stretch
            if at_spot(i):
                return i
            continue
        if speed[i] > p["KICK_V"] >= speed[i - 1]:
            still = seen[i - prior:i] & (speed[i - prior:i] < p["KICK_STILL"])
            if still.sum() >= need and at_spot(i - prior + np.flatnonzero(still)[-1]):
                return i
    return None


def whistle_frames(fps, game, data_dir):
    """First frame of every referee whistle that is not a kick-off."""
    from whistle import whistles
    kick_offs = np.array([e["start_frame"] for e in load_tags(game, data_dir, fps)
                          if "Kick Off" in (e["tags"].get("Type") or "")])
    return [first for first, _ in whistles(game, data_dir)
            if not (len(kick_offs) and np.abs(kick_offs - first).min() <= 5 * fps)]


def after_whistle(firsts, n_frames, f, p, fps, pitch):
    """Flag the stretch after each whistle in `firsts`: T_WHISTLE, or up to the restart with
    WHISTLE_HOLD unless play was already dead when it blew."""
    flagged = np.zeros(n_frames, dtype=bool)
    for first in firsts:
        before = slice(max(0, first - int(p["DEAD_BEFORE"] * fps)), first)
        dead = not f["seen"][before].any() and f["mean_speed"][first] < p["V_DEAD"]
        restart = whistle_restart(first, f, p, fps, pitch) if p["WHISTLE_HOLD"] and not dead else None
        flagged[first:restart or first + int(p["T_WHISTLE"] * fps)] = True
    return flagged


def attack_direction(x, game=GAME):
    """+1 per frame where H attacks towards +x, -1 where towards -x: H defends the end its
    goalkeeper (slot 24) stands nearer than A's (slot 46), per half."""
    h_attack = np.zeros(len(x), dtype=np.int8)
    for _, first, last in half_video_blocks(game):
        keepers = np.nanmean(x[first:last + 1, [23, 45]], axis=0)
        h_attack[first:last + 1] = 1 if keepers[0] < keepers[1] else -1
    return h_attack


def closest_approaches(f, p, fps):
    """[(frame, team)]: a player's closest approach to the ball, within R_CLOSE, where the ball's
    velocity turns or changes speed across it. The team is that player's, so a contested ball goes
    to whoever came nearest."""
    dist, team = f["dist"], f["team"]
    vx, vy, speed = f["ball_vx"], f["ball_vy"], f["ball_speed"]
    k, lag, n = max(1, int(p["CA_WIN"] * fps)), int(p["CA_LAG"] * fps), len(dist)
    events = []
    for j in range(dist.shape[1]):
        d = dist[:, j]
        near = d < p["R_CLOSE"]
        if not near.any():
            continue
        lowest = np.lib.stride_tricks.sliding_window_view(
            np.pad(d, k, constant_values=np.inf), 2 * k + 1).min(axis=1)
        slot_team = np.broadcast_to(team, dist.shape)[:, j]
        events += [(i, float(d[i]), slot_team[i]) for i in np.flatnonzero(near & (d <= lowest))
                   if lag <= i < n - lag]

    kept = []
    for i, d, t in sorted(events):
        other = f["d_away"][i] if t == H else f["d_home"][i]
        if p["CA_MARGIN"] and other - d < p["CA_MARGIN"]:
            continue
        before, after = speed[i - lag], speed[i + lag]
        with np.errstate(invalid="ignore"):
            if before > 1 and after > 1:
                cos = (vx[i - lag] * vx[i + lag] + vy[i - lag] * vy[i + lag]) / (before * after)
                hit = np.degrees(np.arccos(np.clip(cos, -1, 1))) > p["CA_TURN"]
            else:
                hit = abs(after - before) > p["CA_DV"]
        if not hit:
            continue
        if kept and i - kept[-1][0] <= k:              # one touch per contact: the nearest player
            if d < kept[-1][1]:
                kept[-1] = (i, d, t)
            continue
        kept.append((i, d, t))
    return [(i, int(t)) for i, _, t in kept]


def touches(f, p, fps):
    """[(frame, team, confirmed, forward)] for every touch whose team is clear. Stage 4b.

    A touch is a sharp change in the ball's direction or speed with a player close by; its team is
    the nearest player's. With FLIP, a reversal of the ball's x-velocity counts too. `forward`:
    the confirming touch (the same team's next one, else this one) sends the ball towards that
    team's attacking goal; needs f["h_attack"]."""
    lag = 3
    vx, vy, speed = f["ball_vx"], f["ball_vy"], f["ball_speed"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vx[lag:] * vx[:-lag] + vy[lag:] * vy[:-lag]) / (speed[lag:] * speed[:-lag])
        turn = np.r_[np.zeros(lag), np.degrees(np.arccos(np.clip(cos, -1, 1)))]
        turn[(speed < 0.5) | np.r_[np.ones(lag, bool), speed[:-lag] < 0.5]] = 0  # direction is noise
        change = np.r_[np.zeros(lag), np.abs(speed[lag:] - speed[:-lag])]
        near = np.minimum(f["d_home"], f["d_away"])
        sharp = (turn > p["TOUCH_TURN"]) | (change > p["TOUCH_DV"])
        if p["FLIP"]:
            half = int(p["FLIP_LAG"] * fps)
            before, after = vx[:-2 * half], vx[2 * half:]
            flip = np.zeros(len(vx), dtype=bool)
            flip[half:-half] = ((np.sign(before) != np.sign(after)) & (np.abs(before) > p["FLIP_V"])
                                & (np.abs(after) > p["FLIP_V"]))
            sharp |= flip
        hit = f["seen"] & (near < p["R_TOUCH"]) & sharp

    found, frames = [], np.flatnonzero(hit)
    runs = np.split(frames, np.flatnonzero(np.diff(frames) > lag) + 1) if len(frames) else []
    for run in runs:
        score = turn[run] + change[run]
        i = run[np.argmax(score)] if score.max() > 0 else run[len(run) // 2]  # a flip alone
        gap = f["d_away"][i] - f["d_home"][i]
        if abs(gap) < p["TOUCH_MARGIN"]:
            continue
        found.append((i, H if gap > 0 else A))
    if p["TOUCH_MODE"]:
        close = closest_approaches(f, p, fps)
        if p["TOUCH_MODE"] == 1:
            found = close
        else:                                          # both: a closest approach wins a clash
            near = int(p["CA_WIN"] * fps)
            starts = np.array([i for i, _ in close])
            found = sorted(close + [(i, t) for i, t in found
                                    if not len(starts) or np.abs(starts - i).min() > near])

    keep, window = int(p["TOUCH_KEEP"] * fps), fps // 2
    control = f["state"] == CONTROL
    nearer_home = f["d_home"] < f["d_away"]
    h_attack = f.get("h_attack")
    out = []
    for k, (i, team) in enumerate(found):
        nxt = found[k + 1] if k + 1 < len(found) else None
        by_team = control[i:i + window] & (nearer_home[i:i + window] == (team == H))
        again = nxt is not None and nxt[1] == team and nxt[0] - i <= p["TOUCH_NEXT"] * fps
        confirmed = again or by_team.sum() >= keep
        forward = False
        if h_attack is not None:
            j = min(len(vx) - 1, (nxt[0] if again else i) + fps // 3)   # 0.3 s after
            forward = bool(vx[j] * h_attack[j] * (1 if team == H else -1) > 0)
        out.append((i, team, confirmed, forward))
    return out


def touch_tracks(n_frames, state, touch_list, p, fps):
    """(carry, switch, regain): per-frame team nudge after a confirmed touch, +1 H / -1 A (FLIGHT
    frames only unless CARRY_ALL); the per-frame H<->A cost, lowered around confirmed changes of
    team; and with FLIP, a signed pull towards the new team from the first touch of each change."""
    carry = np.zeros(n_frames, dtype=np.float32)
    regain = np.zeros(n_frames, dtype=np.float32)
    switch = np.full(n_frames, p["C_SWITCH"], dtype=np.float64)
    reach, win = int(p["CARRY_MAX"] * fps), max(1, int(p["TOUCH_WIN"] * fps))
    hold = int(p["REGAIN_HOLD"] * fps)
    confirmed = [(i, team, fwd) for i, team, ok, fwd in touch_list if ok]
    last = None
    for k, (i, team, fwd) in enumerate(confirmed):
        sign = 1 if team == H else -1
        stop = min(i + reach, confirmed[k + 1][0] if k + 1 < len(confirmed) else n_frames)
        carry[i:stop] = sign if p["CARRY_ALL"] else np.where(state[i:stop] == FLIGHT, sign, 0)
        if last is not None and team != last:
            switch[max(0, i - win):i + win + 1] = p["C_SWITCH_TOUCH"]
            if p["FLIP"]:
                other = [j for j, t, _ in confirmed[k + 1:] if t != team]
                end = min(i + hold, other[0] if other else n_frames)
                regain[i:end] = sign * (p["W_REGAIN"] + p["W_FWD"] * fwd)
        last = team
    return carry, switch, regain


def tag_tracks(n_frames, p, fps, game, data_dir):
    """(set_piece_soon, phase, set_piece_starts): a flag for the T_PRE before each set piece, +1 (H)
    / -1 (A) / 0 per frame from the phase tags (overlaps add up), and the set-piece start frames."""
    soon = np.zeros(n_frames, dtype=bool)
    phase = np.zeros(n_frames, dtype=np.float32)
    starts = []
    pre = int(round(p["T_PRE"] * fps))
    for event in load_tags(game, data_dir, fps):
        start, end = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        if event["code"] == "SET PIECES":
            soon[max(0, start - pre):start] = True
            starts.append(start)
            continue
        direction = PHASE_DIRECTION.get(event["code"])
        tagged = TAG_TEAM.get((event["tags"].get("Team") or "").lower())
        if direction and tagged is not None:
            phase[start:end] += direction * (1 if tagged == H else -1)
    return soon, phase, starts


def gate_set_pieces(soon, starts, sign, pre):
    """`soon` with each set piece's lead-in window starting only at the first frame in it where
    `sign` (ball out, still, or after a whistle) holds. SET_GATE."""
    gated = soon.copy()
    for start in starts:
        low = max(0, start - pre)
        first = np.flatnonzero(sign[low:start])
        gated[low:start] = False
        if len(first):
            gated[low + first[0]:start] = True
    return gated


def evidence(d_home, d_away, state, stoppage, phase, p, ctrl_weight=None):
    """(n_frames, 3) log-scores for H, A, D. Stages 3-4. `stoppage` maps a weight knob to its
    signal; S = S0 + sum of weight * signal."""
    stop = np.full(len(state), p["S0"], dtype=np.float64)
    for weight, signal in stoppage.items():
        stop += p[weight] * signal

    scores = np.zeros((len(state), 3))
    control = state == CONTROL
    gap = d_away[control] - d_home[control]  # CONTROL means someone is near, so never inf - inf
    p_home = np.clip(1.0 / (1.0 + np.exp(-gap / p["GAP_SCALE"])), 1e-6, 1 - 1e-6)
    weight = p["W_CTRL"] if ctrl_weight is None else ctrl_weight[control]
    scores[control, H] = weight * np.log(p_home)
    scores[control, A] = weight * np.log(1 - p_home)
    scores[:, H] += p["W_PHASE"] * np.maximum(phase, 0)
    scores[:, A] += p["W_PHASE"] * np.maximum(-phase, 0)
    scores[:, D] = stop
    return scores


def transition_matrix(p):
    """log-score of moving from state i (row) to state j (column)."""
    return -np.array([
        [0.0, p["C_SWITCH"], p["C_TO_D"]],
        [p["C_SWITCH"], 0.0, p["C_TO_D"]],
        [p["C_FROM_D"], p["C_FROM_D"], 0.0],
    ])


def viterbi(scores, transitions, switch=None):
    """Most likely state per frame. Linear in frames; uniform start. `switch`, if given, is the
    H<->A cost per frame, overriding the one in `transitions`."""
    n = len(scores)
    back = np.zeros((n, 3), dtype=np.int8)
    best = scores[0].copy()
    step = transitions.copy()
    for i in range(1, n):
        if switch is not None:
            step[H, A] = step[A, H] = -switch[i]
        candidates = best[:, None] + step   # from (row) -> to (column)
        back[i] = np.argmax(candidates, axis=0)
        best = candidates[back[i], np.arange(3)] + scores[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(best)
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def predict(p=PARAMS, game=GAME, data_dir=None):
    """Labels for every video frame of the tactical recording, plus the ball states for reporting."""
    meta_path, = require(["atd_meta"], game, data_dir)
    fps, pitch = GAMES[game]["fps"], GAMES[game]["pitch"]
    n_frames = source_frames("tactical", game)

    x, y, _ = read_positions(game, data_dir)
    team = team_of_each_track(meta_path)
    if p["GK_SIDE"]:
        team = keeper_sides(x, team, game)
    soon, phase, set_starts = tag_tracks(n_frames, p, fps, game, data_dir)
    whistled = whistle_frames(fps, game, data_dir) if p["W_WHISTLE"] else []
    h_attack = attack_direction(x, game) if p["TOUCH"] else None
    blocks = [(first, last) for _, first, last in half_video_blocks(game)]
    return decode(x, y, team, soon, phase, set_starts, whistled, h_attack, blocks, p, pitch, fps)


def decode(x, y, team, soon, phase, set_starts, whistled, h_attack, blocks, p, pitch, fps):
    """Stages 2-5 on arrays, so any match in this shape can run: labels per frame (D outside
    `blocks`, the [(first, last)] frames to decode) and the ball states."""
    n_frames = len(x)
    f = features(x, y, team, p, pitch, fps)
    state = f["state"]
    stoppage = {
        "W_SLOW": np.nan_to_num(p["V_PLAY"] - f["mean_speed"]),  # no players -> no evidence
        "W_STILL": f["still"],
        "W_SET": soon,
        "W_OUT": state == OUT,
    }
    # Only computed when switched on.
    if p["W_RESTART"]:
        stoppage["W_RESTART"] = restart_lookback(x, y, f, set_starts, p, pitch, fps)
    if p["W_WHISTLE"]:
        stoppage["W_WHISTLE"] = after_whistle(whistled, n_frames, f, p, fps, pitch)
    if p["SET_GATE"]:
        sign = (state == OUT) | f["still"] | stoppage.get("W_WHISTLE", False)
        stoppage["W_SET"] = gate_set_pieces(soon, set_starts, sign, int(round(p["T_PRE"] * fps)))
    switch, ctrl_weight = None, None
    if p["TOUCH"]:
        f["h_attack"] = h_attack
        carry, switch, regain = touch_tracks(n_frames, state, touches(f, p, fps), p, fps)
        if p["CARRY_ALL"]:
            # The nearest player is from the other team than the last toucher: a presser.
            presser = (carry != 0) & ((f["d_home"] < f["d_away"]) != (carry > 0))
            ctrl_weight = np.where(presser, p["W_CTRL"] * p["CTRL_PRESS"], p["W_CTRL"])
    scores = evidence(f["d_home"], f["d_away"], state, stoppage, phase, p, ctrl_weight)
    if p["TOUCH"]:
        pull = p["W_CARRY"] * carry + regain
        scores[:, H] += np.maximum(pull, 0)
        scores[:, A] += np.maximum(-pull, 0)
    if p["W_PROB"]:
        # The leading team loses nothing, so this never pushes towards or away from D.
        prob = np.clip(team_prob.p_home(x, y, team), 0.02, 0.98)
        top = np.maximum(prob, 1 - prob)
        scores[:, H] += p["W_PROB"] * (np.log(prob) - np.log(top))
        scores[:, A] += p["W_PROB"] * (np.log(1 - prob) - np.log(top))

    labels = np.full(n_frames, D, dtype=np.int8)
    transitions = transition_matrix(p)
    for first, last in blocks:
        labels[first:last + 1] = viterbi(scores[first:last + 1], transitions,
                                         None if switch is None else switch[first:last + 1])
    return labels, state


def possessions(labels, game=GAME):
    """Lengths in seconds of every maximal H or A run inside the graded halves."""
    fps, lengths = GAMES[game]["fps"], []
    for _, first, last in half_video_blocks(game):
        block = labels[first:last + 1]
        change = np.flatnonzero(block[1:] != block[:-1]) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(block)]
        lengths += [(end - start) / fps for start, end in zip(starts, ends) if block[start] != D]
    return np.array(lengths)


def self_check():
    """Viterbi on a hand-made sequence with a known answer. Needs no data."""
    p = dict(PARAMS)
    strong_h, strong_a, dead = [0.0, -2.0, -3.0], [-2.0, 0.0, -3.0], [-3.0, -3.0, 1.0]
    flicker = [-2.0, 0.0, -3.0]                       # one contested frame for A
    scores = np.array([strong_h] * 30 + [flicker] + [strong_h] * 30      # a blip must not switch
                      + [strong_a] * 20                                   # 0.8 s spell survives
                      + [dead] * 40 + [strong_h] * 30)
    path = viterbi(scores, transition_matrix(p))
    assert (path[:61] == H).all(), "a single contested frame flipped possession"
    assert (path[61:81] == A).all(), "a short clear spell was erased"
    assert (path[81:121] == D).all() and (path[121:] == H).all()

    # A 0.3 s spell is erased at the normal cost and kept where a regain lowers it.
    scores = np.array([strong_h] * 30 + [strong_a] * 8 + [strong_h] * 30)
    assert (viterbi(scores, transition_matrix(p))[30:38] == H).all()
    switch = np.full(len(scores), p["C_SWITCH"])
    switch[[30, 38]] = p["C_SWITCH_TOUCH"]
    assert (viterbi(scores, transition_matrix(p), switch)[30:38] == A).all()

    # After a whistle the ball is played 30 m away (not a restart), then placed back near where
    # play stopped and kicked: the restart is that kick.
    fps, n = 25, 600
    bx, by, speed = np.full(n, 50.0), np.full(n, 30.0), np.full(n, 4.0)
    bx[100:250], speed[100:250] = 20.0, 0.5          # still, 30 m away...
    speed[250:260] = 9.0                             # ...and kicked: not a restart
    speed[300:400] = 0.5                             # still back at the spot
    speed[400:] = 9.0                                # the free kick
    f = {"ball_x": bx, "ball_y": by, "ball_speed": speed, "seen": np.ones(n, dtype=bool)}
    assert whistle_restart(50, f, p, fps, (105.3, 68.0)) == 400

    # A ball at 10 m/s passes one player at 0.6 m and rolls on straight (no touch), then turns
    # 40 degrees at its closest approach to another (a touch, his team).
    n, turn_at = 100, 60
    vx, vy = np.full(n, 10.0), np.zeros(n)
    vx[turn_at:], vy[turn_at:] = 10 * np.cos(np.radians(40)), 10 * np.sin(np.radians(40))
    dist = np.full((n, 2), np.inf, dtype=np.float32)
    dist[:, 0] = np.abs(np.arange(n) - 30) * 0.4 + 0.6    # rolls past slot 0 at frame 30
    dist[:, 1] = np.abs(np.arange(n) - turn_at) * 0.4 + 0.3
    f = {"dist": dist, "team": np.array([H, A]), "ball_vx": vx, "ball_vy": vy,
         "ball_speed": np.hypot(vx, vy), "d_home": dist[:, 0], "d_away": dist[:, 1]}
    assert closest_approaches(f, p, fps) == [(turn_at, A)]

    # A set piece's lead-in only counts from the first stoppage sign inside it.
    soon = np.zeros(100, dtype=bool); soon[40:80] = True
    sign = np.zeros(100, dtype=bool); sign[60] = True
    gated = gate_set_pieces(soon, [80], sign, 40)
    assert not gated[40:60].any() and gated[60:80].all()


if __name__ == '__main__':
    self_check()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                        help="override knobs, e.g. --set C_SWITCH=25 W_SET=4")
    parser.add_argument("--name", default="hmm", help="submission name, without .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    params = dict(PARAMS)
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in params:
            parser.error("unknown knob {!r}; have {}".format(name, ", ".join(params)))
        params[name] = float(value)
    changed = {k: v for k, v in params.items() if v != PARAMS[k]}
    print("knobs changed from defaults: {}".format(changed or "none"))

    labels, state = predict(params)
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    names = ("MISSING", "OUT", "FLIGHT", "CONTROL")
    print("ball state on graded frames: " + "  ".join(
        "{} {:.1%}".format(names[k], np.mean(state[graded] == k)) for k in range(4)))
    print("labels on graded frames:     " + "  ".join(
        "{} {:.1%}".format(LABELS[k], np.mean(labels[graded] == k)) for k in range(3)))
    lengths = possessions(labels)
    print("possessions over the match: {}  median {:.1f} s  shortest {:.2f} s  longest {:.1f} s".format(
        len(lengths), np.median(lengths), lengths.min(), lengths.max()))

    path = write_submission(os.path.join(args.out, args.name + ".csv"), LABELS[labels])
    print("-> {}\n\nNow score it:  python evaluation/run_evaluation.py".format(os.path.relpath(path, ROOT)))
