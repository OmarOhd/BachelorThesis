# h8+rep

`train_h8_rep.py` trains h8+rep on a Google Colab A100 for 100,000 steps.

| Steps | Learning rate |
|---|---|
| 1 to 5,000 | 5e-4 |
| 5,001 to 50,000 | 1e-4 |
| 50,001 to 90,000 | 3.16e-5 |
| 90,001 to 100,000 | 1e-5 |

Apart from the repetition flags and the schedule, the settings are the same as
for h8. The thesis uses the checkpoint at 90,000 steps.

Mount Google Drive in the notebook and run

```
!git clone https://github.com/OmarOhd/BachelorThesis.git /content/code
!pip install -q -r /content/code/requirements.txt
!python /content/code/colab/h8_rep/train_h8_rep.py --data-zip /content/drive/MyDrive/processed.zip
```

`processed.zip` contains `data/tcec/processed` from the data pipeline.
Checkpoints are saved to Google Drive. If Colab disconnects, run the last
command again to continue.

The original run lowered the learning rate twice after loss spikes and resumed
from a saved checkpoint each time. It took seven Colab sessions, which are
listed in `results/training_logs/h8_rep/README.txt`.

`colab_match_h8_rep_90k_2026-11-14.py` plays the 300 h8+rep games, and
`analyze_h8_rep_stockfish.py` runs the Stockfish analysis of these games.
To use the released checkpoint, pass
`--rep checkpoints/h8_rep/model_step_90000.pt` to the match script.
