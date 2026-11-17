# Possession from tracking: an HMM with a learned team probability

Submission for the More Than A Hack 2027 possession challenge (BAR-ATH, tactical ATD). Every video
frame is labelled H (Barcelona), A (Athletic) or D (dead ball) by a three-state Viterbi decoder
over evidence from the tracking, Smart Tagging and referee whistles in the video's audio.

The technical document is [SOLUTION.md](SOLUTION.md).

## Reproduce the submission

```bash
pip install -r requirements.txt              # numpy, pandas; ffmpeg must be on PATH
export MTAH_DATA_DIR=/path/to/data           # the match data bundle, not in this repository
python starter/possession_hmm.py             # writes evaluation/results/hmm.csv
python evaluation/validate_submission.py evaluation/results/hmm.csv
```

## Layout

- `starter/possession_hmm.py`: the pipeline; every knob is in `PARAMS` with the reason for its value.
- `starter/team_prob.py`, `starter/team_prob_weights.json`: P(home) logistic regression, fitted on
  the open IDSSE matches.
- `starter/whistle.py`: referee whistle detection from the video's audio.
- `starter/tune_idsse.py`: IDSSE degradation, leave-one-match-out checks and the weight fit.
- `starter/velocity.py`, `baseline.py`, `load_smart_tagging.py`, `games.py`, `evaluation/`: loaders
  and the format validator, from the challenge's starter kit (MIT, see [LICENSE](LICENSE)).

No match data, and nothing derived from it, is in this repository.
