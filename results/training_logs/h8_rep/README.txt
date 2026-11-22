h8+rep logs, joined from seven Colab sessions.

eval_log.csv   step, raw top-1, legal top-1, positions (every 10,000 steps)
train_log.csv  step, training loss, learning rate (every 100 steps)

Sessions
  steps      1-5000    h8_rep_fp32_seed0_20261110_153312
  steps   5001-6000    h8_rep_fp32_seed0_20261110_153312_lr_recovery_20261110T180342541369Z
  steps   6001-10000   h8_rep_continuation_20261110T184955886017Z
  steps  10001-18000   h8_rep_full_20261110T225010721797Z
  steps  18001-21000   h8_rep_full_20261111T190456306577Z
  steps  21001-50000   h8_rep_full_20261112T160727042700Z
  steps  50001-100000  h8_rep_lr50k_20261113T181553407907Z

Learning rate 5e-4 up to step 5,000, 1e-4 up to 50,000, 3.16e-5 up to 90,000
and 1e-5 up to 100,000. Seed 0, FP32.

The test results of the 90,000-step checkpoint are in
results/test_eval/h8_rep_90k.csv.
