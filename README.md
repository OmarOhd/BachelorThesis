# Comparing Input Representations for Transformer-Based Chess Models

Code and results for my bachelor thesis at Leuphana University Lüneburg (2026).

Three models are trained on games from the Top Chess Engine Championship (TCEC).

- h1 sees the current position.
- h8 also sees up to seven earlier positions.
- h8+rep is h8 with repetition flags, trained on a different schedule.

## Folders

```
analysis/       statistics and figures from results/
colab/h8_rep/   h8+rep training, matches and Stockfish analysis
data_pipeline/  TCEC parsing, splits and repetition flags
model/          model, training and evaluation
scripts/        matches, Stockfish analysis and dataset statistics
tests/          unit tests
results/        saved logs, test results, games and Stockfish scores
```

## Setup

Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
```

## Reproduce the numbers

```bash
python -m analysis.reproduce_statistics --json reproduced.json
python -m analysis.make_figures --out figures
```

This recomputes all tables and figures of the thesis from `results/` in a few
seconds. Confidence intervals use a paired bootstrap with 10,000 draws and seed 0.

## Checkpoints

The checkpoints used in the thesis are attached to release v1.0 as
`checkpoints.zip`. Unzip it in the repository folder and check the files with

```bash
shasum -a 256 -c checkpoints/SHA256SUMS.txt
```

## Full pipeline

The full pipeline needs the TCEC games, a CUDA GPU and Stockfish 18. With the
released checkpoints you can skip the training.

### Data

Put the PGN files from [TCEC-Chess/tcecgames](https://github.com/TCEC-Chess/tcecgames)
in `data/tcec/events/full/events/` and run

```bash
python -m data_pipeline.parse_tcec
python -m data_pipeline.build_index
python -m data_pipeline.make_splits
python -m data_pipeline.build_repetition_flags
```

### Training

h1 and h8 were trained for 200,000 steps on one V100 each. Their seed was set to the trainer default (0).

```bash
python -m model.train_model --history 1 --steps 200000 --seed 0 --run-name h1_masked
python -m model.train_model --history 8 --steps 200000 --seed 0 --run-name h8_masked
```

h8+rep was trained on Google Colab, see `colab/h8_rep/README.md`.

### Test set

```bash
python -m model.eval_model --ckpt checkpoints/h1_masked/model_step_60000.pt --split test --by-game h1_60k.csv
python -m model.eval_model --ckpt checkpoints/h8_masked/model_step_30000.pt --split test --by-game h8_30k.csv
```

### Matches

h1's repetition filter is off in the first run and on in the second.

```bash
python -m scripts.match.dump_games --games 100 --no-avoid-repetition --out matches/off
python -m scripts.match.dump_games --games 100 --out matches/on
```

### Stockfish analysis

Run this for `matches/off` and `matches/on`.

```bash
python -m scripts.analysis.analyze_games --pgn matches/off --stockfish /path/to/stockfish \
    --depth 16 --threads 4 --hash 512 --skip-book --out analysis_off_d16
```

GPU training and Stockfish are not fully deterministic, so a rerun gives
slightly different numbers. The files in `results/` are the ones used in the
thesis.
