# Possession from tracking: an HMM with a learned team probability

Submission for the More Than A Hack 2027 possession challenge (BAR-ATH, tactical ATD). Every video
frame is labelled H (Barcelona), A (Athletic) or D (dead ball) by a three-state Viterbi decoder
over evidence from the tracking, Smart Tagging and referee whistles in the video's audio.

The technical document is [mbg-for-next-yamal.md](mbg-for-next-yamal.md).

## Reproduce the submission

Needs **Python 3.12 or later** (the pinned numpy 2.5.5 requires it; the starter scripts exit with a
clear message on older versions) and **ffmpeg 9.0.2** on PATH. The whistle detector decodes the
video's audio with ffmpeg, and another version may shift the samples and so the detected whistles.
`starter/whistle.py` warns when the version differs.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # numpy, pandas (pinned)
export MTAH_DATA_DIR=/path/to/data           # the match data bundle, not in this repository
python starter/possession_hmm.py --name mbg-for-next-yamal   # writes evaluation/results/mbg-for-next-yamal.csv
python evaluation/validate_submission.py evaluation/results/mbg-for-next-yamal.csv
shasum -a 256 evaluation/results/mbg-for-next-yamal.csv      # compare with the hash below
```

### Data layout

`MTAH_DATA_DIR` is the folder that contains `tactical/` (default: `./data`). The pipeline reads
only these four files; the homography and everything in `broadcast/` are not used.

```
$MTAH_DATA_DIR/
`-- tactical/
    |-- BAR-ATH_tactical.mp4                  audio, for the referee whistles
    |-- BAR-ATH_tactical_atd.xml              tracking metadata (teams of the track slots)
    |-- BAR-ATH_tactical_atd.txt              raw tracking, players and ball
    `-- BAR-ATH_tactical_smart_tagging.xml    Smart Tagging
```

### Expected output

The first run prints a summary and writes `evaluation/results/mbg-for-next-yamal.csv`:

- 199,319 data rows plus the `frame,label` header; `validate_submission.py` reports no errors.
- Over the whole match, 301 possessions with a median of 7.2 s, and 31.5% of graded frames
  labelled D (the figures in section 5 of the technical document).
- SHA-256 of the CSV: `TODO: fill in from a clean run` (Python 3.12.x, ffmpeg 9.0.2).

The first run also detects the whistles (about half a minute) and caches them in
`evaluation/reports/`. Delete that folder, or run `python starter/whistle.py --refresh`, to
rescan.

## Layout

- `starter/possession_hmm.py`: the pipeline; every knob is in `PARAMS` with the reason for its value.
- `starter/team_prob.py`, `starter/team_prob_weights.json`: P(home) logistic regression, fitted on
  the open IDSSE matches.
- `starter/whistle.py`: referee whistle detection from the video's audio.
- `starter/tune_idsse.py`: IDSSE degradation, leave-one-match-out checks and the weight fit.
- `games.py` (repository root), `starter/velocity.py`, `starter/baseline.py`,
  `starter/load_smart_tagging.py`, `evaluation/`: loaders and the format validator, from the
  challenge's starter kit (MIT, see [LICENSE](LICENSE)).

No match data, and nothing derived from it, is in this repository.
