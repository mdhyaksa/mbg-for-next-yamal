# Possession from tracking: an HMM with a learned team probability

Technical document for the More Than A Hack 2027 possession challenge (BAR-ATH, tactical ATD).

## 1. Approach

Every video frame gets one of H (Barcelona), A (Athletic) or D (dead ball). Each frame gets a
log-score for each of the three states, and a three-state Viterbi decoder picks the most likely
label sequence, paying a cost every time the label changes. The change costs encode that
possessions last seconds rather than frames. The per-frame scores hold what the tracking, the
Smart Tagging and the video's audio say about that frame.

The nearest-player baseline with a majority vote erases short possessions, misplaces changes by
seconds and has no notion of a dead ball. A decoder with change costs addresses all of these with a
small set of named knobs, runs in about 10 s on numpy alone once the ATD is cached, and lets each source of
evidence be switched off and measured on its own. Other decoders were built and dropped (section 6).

The pipeline is `starter/possession_hmm.py`:

1. Load ATD positions (ball in the last column), the team of each track slot, velocities at 25 fps,
   Smart Tagging events and phases, and referee whistles from the video's audio.
2. Give each frame a ball state: MISSING, OUT (0.5 to 5 m beyond a line), FLIGHT (faster than
   8 m/s) or CONTROL (a player within 2 m).
3. Compute a stoppage score S, the log-odds that play is stopped. It rises with slow players (mean
   speed under 1.5 m/s), a still ball and the ball out of bounds. It also rises in the 15 s before a
   Smart Tagging set piece, counted only from the first frame in that window where the ball is out,
   still or a whistle has blown. A restart lookback marks a missing-ball stretch of 1 s or more as
   dead, for up to its last 30 s, when the ball reappears still, on a line, in a goal area, at a
   corner or the centre spot, or with a set piece tagged within 3 s. After a referee whistle
   (`starter/whistle.py`), play stays dead until the ball is kicked again from where play stopped,
   or for 5 s if no such restart is found. Kick-off whistles are skipped, and a whistle heard while
   play is already dead gets the 5 s only. The whistle detector looks for two flat lines near 3.80
   and 4.28 kHz, 30 to 38 dB above the band median, lasting 0.3 to 1 s.
4. Score the teams with a per-frame P(home has the ball) from a logistic regression on tracking
   features (`starter/team_prob.py`, section 3). It is added to H and A as `log p - log max(p, 1-p)`,
   so the leading team loses nothing and the term never moves a frame towards or away from D.
   Detected touches add to this. A touch is a turn of 25 degrees or a speed change of 4 m/s within
   3 m of a player, or a player's closest approach within 1 m with the ball's path changing across
   it. It is confirmed by the same team's next touch within 3 s, or by its control in the next
   0.5 s. The confirmed toucher's team is carried through the FLIGHT frames that follow, for up to
   3 s, and the H/A switch cost drops near a confirmed regain. The two goalkeeper slots take the team
   of the half they stand in, because the ATD's keeper slots follow whichever keeper is nearest.
   Smart Tagging phases nudge the team they name by 0.05 per frame.
5. Decode each half with Viterbi. Switching between H and A costs 15 (about 0.5 s of clear evidence
   for the other team), entering D costs 30 (about 1.5 s of strong stoppage evidence) and leaving D
   costs 10, because restarts are sharper than stoppages.
6. Write a `frame,label` CSV for all 199,319 frames. Ungraded frames get D.

## 2. Assets used

| Asset | Used for | Status |
|---|---|---|
| Tactical ATD (positions, team of each slot) | everything | used |
| Smart Tagging | set-piece lead-in for D, restart lookback, skipping kick-off whistles, phase nudge | used |
| Tactical video audio | referee whistles | used |
| Broadcast ATD | a second source for ball gaps | tried and dropped: it cut missing-ball frames only from 40.7% to 37.4% |
| Homography | ball height from the camera, since the ATD has none | tried and dropped: too noisy, with nothing to check it against |

## 3. Training data and knob selection

The only labelled data is the IDSSE open dataset: 7 Bundesliga 2022/23 matches, CC-BY 4.0
(Bassek et al. 2025, https://doi.org/10.1038/s41597-025-04505-y, mirrored at
huggingface.co/datasets/pysport/idsse-data). Its per-frame `BallPossession` and `BallStatus` flags
come from the DFL's data provider. It covers 664.5 minutes and 3,061 possessions. `BallStatus` 0 is
read as D, and otherwise `BallPossession` 1 or 2 as H or A. Nothing from BAR-ATH was labelled, by
hand or otherwise.

IDSSE is clean optical tracking, so `starter/tune_idsse.py` degrades it before scoring, with every
setting calibrated on the ATD alone and no labels. Ball gaps are drawn from the ATD's own seen and
missing run lengths (median gap 10.7 s, ball seen on 59.3% of graded frames). 7% of player frames
drop out. Slow AR(1) position error of 0.4 m on the ball and 0.05 m on players keeps the speed
distributions close to the ATD's.

Each candidate is scored leave one match out, twice: on the raw DFL labels, and with team spells
under 0.5 s merged into the possession around them. DFL possessions are short (median 3.9 s in
5-minute windows, against 7.6 s on the public window) and DFL counts a single touch as a
possession, which the competition's ground truth does not. The merge threshold sits below the
shortest possession in the public window (0.76 s), so it never removes a spell the competition
would count. A change has to win on both versions.

Two defaults were changed this way. `C_TO_D` went from 20 to 30, a small gain on all 7 matches,
raw and merged, on clean and degraded tracking. The team probability (`W_PROB` 1) replaced the
nearest-player evidence (`W_CTRL` from 1 to 0). Its weights are fitted on all 7 matches with
`tune_idsse.py fit-team`: 6,000 live frames per half, both team orders, L2 penalty 1. Fitted on six
matches and scored on the seventh, it gains 0.017 spellF1 on raw labels and 0.017 on merged labels,
and is better on all 7 matches (`tune_idsse.py team-check --seed 1`). A second implementation of the same fit
(scikit-learn, not in this repository), over two degradation seeds, gives 0.012 and 0.011, better on 13 of 14 match results.

The alternatives were compared on the same degraded data, over both seeds (14 held-out match
results, spellF1):

| Method | raw | merged | macroF1 | Possessions per match |
|---|---|---|---|---|
| HMM, nearest-player evidence | 0.332 | 0.343 | 0.715 | 233 |
| HMM, team probability (submitted) | 0.344 | 0.355 | 0.721 | 238 |
| Touch-based segmentation, confirmation and probability split | 0.335 | 0.346 | 0.720 | 226 |

DFL truth has 437 possessions per match. Touch-based segmentation changes possession at a touch by
the other team once that team's next touch within 3 s confirms it, and places the change at the
best split under the team probability. It is better than the nearest-player HMM on 7 of 14 match
results, against 13 of 14 for the team probability.

IDSSE has no audio and no Smart Tagging. DFL's own restart and foul events stand in for them during
tuning, at fixed knob values, so the following were set from physics, from distributions over the
whole BAR-ATH match and from the video, and never from the ground truth's full-match statistics:
whistle detection (its frequencies were measured on the two public whistles), `T_WHISTLE`, the
whistle hold and its restart rules, the set-piece lead-in (`T_PRE`, `SET_GATE`), the restart
lookback, the phase nudge, and `GK_SIDE`, since the ATD's slot convention has no IDSSE equivalent.

Several of these knobs (`S0`, `V_STILL`, `STILL_HOLD`, `T_PRE`, `GK_SIDE`, `SET_GATE`, `TOUCH_MODE`,
`CA_MARGIN`) were changed after looking at failures on the public window. That window holds 17
possessions, and their values should be read as fitted to it. Each change and its effect on the public
window is recorded in the comment beside the knob in `starter/possession_hmm.py`.

## 4. Possession rules implemented

- A team has the ball where the team probability favours it and the decoder has no cheaper path.
  The probability reads the nearest home and away distances, how many players of each side are
  within 3 m and 8 m, and the time since and until each side's last and next touch.
- In flight, the ball belongs to the last confirmed toucher's team for up to 3 s. After that the
  decoder keeps the current team until the other side's evidence outweighs the switch cost.
- In a challenge, a touch with the nearest home and away players within 0.5 m of each other has no
  team. A single unconfirmed touch, such as a block that comes straight back, does not change
  possession without about 0.5 s of evidence or a confirmed regain.
- Play is dead from the whistle or from the ball going out until the ball is kicked again. A set
  piece's lead-in counts as dead once a stoppage sign appears. Goal kicks, throw-ins and corners
  after an unseen stoppage are caught by the restart lookback.
- Possession starts at the restart, because leaving D costs less than entering it.

## 5. Performance

On the public window, used as a pipeline check and not to choose the team probability:

| | spellF1 | macroF1 | Possessions (truth 17) |
|---|---|---|---|
| Submitted | 0.607 | 0.918 | 12 |
| Same HMM with nearest-player evidence | 0.575 | 0.924 | 15 |
| `baseline_smooth50` (published floor) | 0.326 | 0.621 | 26 |

One public possession moves spellF1 by about 0.06, so the public difference between the two HMM
versions is about half a possession. Over the whole match the submission has 301 possessions
(median 7.2 s) and labels 31.5% of graded frames D.

## 6. Tried and dropped

- Other decoders: a particle smoother in ball gaps, an explicit-duration decoder, hysteresis
  thresholding, a Bayesian classifier without the HMM, and touch-based segmentation (section 3).
- Stoppage signals: scattered player movement, an offside rule (it fired on 2.5% of in-play public
  frames) and set-piece formation (it added 9 points of D over the match with nothing to show it
  was right).
- Ball-position smoothing and a windowed still-ball test, which corrected a level of tracking noise
  the ATD turned out not to have.

## 7. Limitations

- Possessions under 4 s are mostly missed (5 of 6 on the public window): the switch cost that keeps
  labels stable also drops them.
- About a third of changes of possession land within 1 s of the real change.
- An offside or foul with the ball tracked and still moving is caught only if the whistle is heard
  (both public whistled stoppages are). The detector misses a soft kick-off whistle and one in
  near-silent audio, and its precision over the full match is unknown. Ball gaps that hide a still
  ball are marked dead only when the restart looks like one.
- The full-match dead share is 31.5% D, against 19.6% in the public window, and nothing available
  says whether that is too high.
- The degradation places ball gaps at random, but the ATD loses a still or dead ball more often: in
  the public window the ball is seen on 66% of D frames and 99.9% of H frames. Gaps tied to game
  state would be the next thing to model, followed by a rerun of the knob search.
- Whether DFL's possession flag means what the competition's ground truth means is not established.
  A small audit of short IDSSE spells would settle it.

## 8. Reproduction

Needs Python 3.12 or later (the pinned numpy 2.5.5 requires it) and ffmpeg (for the whistles). The
match data bundle is not in this repository.

```bash
pip install -r requirements.txt              # numpy, pandas (pinned)
export MTAH_DATA_DIR=/path/to/data           # the match data bundle
python starter/possession_hmm.py --name mbg-for-next-yamal   # writes evaluation/results/mbg-for-next-yamal.csv
python evaluation/validate_submission.py evaluation/results/mbg-for-next-yamal.csv
```

The first run detects the referee whistles in the tactical video's audio (`starter/whistle.py`,
about half a minute) and caches them in `evaluation/reports/`. The team-probability weights are in
`starter/team_prob_weights.json`. To refit them and repeat the held-out checks, put the IDSSE XML
files in `data/idsse/` and run:

```bash
python starter/tune_idsse.py fit-team --seed 0     # rewrites the weights
python starter/tune_idsse.py team-check --seed 1   # leave one match out, W_PROB against nearest-player
python starter/team_prob.py && python starter/tune_idsse.py check   # self-checks, no data needed
```
