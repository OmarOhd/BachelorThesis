# Results

These are the files used in the thesis. The scripts in `analysis/` read only
these files.

| Path | Contents |
|---|---|
| `dataset_stats.json` | split sizes, outcomes and game lengths |
| `training_logs/` | validation accuracy every 10,000 steps and training loss every 100 steps |
| `test_eval/` | per-game test results of h1@60k, h8@30k and h8+rep@90k |
| `matches/original/` | games of pairings A, B and C with the filter off and on, and their Stockfish scores |
| `matches/h8_rep/` | the 300 h8+rep games and their Stockfish scores |

In `positions.csv`, `cp_loss` is `max(0, eval_before - eval_after)` for the
player who moved. Scores are capped at ±1,000 centipawns. Rows with
`in_book = True` are the eight opening plies and are not counted.
`matches/original/stockfish/terminal_draw_corrections.csv` lists the 152 final
moves whose score was set to zero because the game had already ended in a draw.
