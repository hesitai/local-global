# Online motion-deblurring training

Run commands from the project root in a Python environment with the training
requirements installed, including PyTorch with CUDA support. Prepare the dataset
paths defined at the top of `train_online_2d.py` before launching.

```bash
MAX_ITERS=200000 GPUS=0 BATCH_SIZE=4 python train_online_2d.py
```

The launcher enables grouped kernel-response supervision with the
`joint_residual` controller by default. It stores
new iteration-based runs under `logs/motiondeblur/Online2D/` with `_i200000` in
the experiment name. Existing checkpoints require `RESUME=1`; a fresh run does
not overwrite an existing experiment containing model weights.

## Training budget

| Launcher variable | Default | Meaning |
| --- | --- | --- |
| `MAX_ITERS` | `200000` | Total processed global batches; `0` selects epoch mode |
| `WARMUP_ITERS` | `2820` | Linear warmup batches in iteration mode; `0` disables warmup |
| `EPOCHS` | `500` with response supervision, otherwise `450` | Training budget when `MAX_ITERS=0` |
| `BATCH_SIZE` | `4` | Total batch size across all visible GPUs |
| `GPUS` | `0` | Visible CUDA device IDs |
| `RESPONSE_VARIANT` | `joint_residual` | R controller: `joint_residual` or legacy `serial` |
| `RESUME` | `0` | Set to `1` to resume the same experiment's latest checkpoint |
| `DRY_RUN` | `0` | Set to `1` to print the command without starting training |

One iteration is one processed global batch, regardless of the GPU count.
Automatic mixed precision may skip an optimizer update on overflow; that batch
still counts toward the iteration budget. There is no gradient accumulation.

Each epoch selects one blur variant per GT image from the cached candidate pool.
With 7,520 GT images, global batch size 4, and incomplete batches dropped, each
epoch contains 1,880 iterations. A 200,000-iteration run therefore ends after 106
complete epochs and 720 batches of epoch 107. The final batch triggers validation
(unless explicitly disabled) and saves both the latest and numbered checkpoints.
Regular validation remains every 25 epochs, and candidate pools refresh every
50 epochs.

The learning rate warms up for 2,820 iterations, then follows cosine decay over
the remaining budget. The default initial rate is `3e-4` and its terminal rate
is `1e-6`. Parameter groups preserve their initial learning-rate ratios. Changing
the batch size does not change the warmup iteration count automatically.

## Resume and epoch mode

```bash
RESUME=1 MAX_ITERS=200000 python train_online_2d.py
MAX_ITERS=0 EPOCHS=500 python train_online_2d.py
```

For a previous serial-controller experiment, select its original architecture:

```bash
RESPONSE_VARIANT=serial RESUME=1 python train_online_2d.py
```

New and serial variants have distinct experiment and training-cache names.
Their weights are not interchangeable. Resume validates the controller variant;
it does not silently convert a previous serial model into a new model.

Resume uses the same experiment configuration and training budget. Iteration
checkpoints include the global batch count, learning-rate scheduler, and AMP
scaler state. Nonterminal checkpoints are saved at epoch boundaries; resuming
restarts at the next epoch. The loader's random ordering is not guaranteed to be
bitwise identical for non-EMA runs. Extending an old epoch-based checkpoint to a
new iteration budget requires a separate continuation learning-rate policy and
is not supported by `RESUME=1`.

When calling `train/train_motiondeblur.py` directly, pass `--max_iters 200000
--warmup --warmup_iters 2820` along with the model and dataset arguments.
Its `--max_iters` default is `0` to preserve compatibility with existing epoch
training scripts. A positive value overrides `--nepoch`.

## CPU checks

```bash
python -m unittest discover -s tests -p 'test_training_schedule.py' -v
python -m unittest discover -s tests -p 'test_joint_residual_response.py' -v
```

## Joint residual response control

The model predicts the same 64-channel complex response R using the same labels
and auxiliary loss. z0/z1 and the eight injection locations are unchanged. The
new controller changes how R controls local features, not how blur is generated.

For projected local content C and spatially aligned response Q, it computes:

```text
H = SiLU(Conv1x1(concat(C, Q, C * Q)))
w = softmax(expert_head(H), axis=experts)
alpha = sigmoid(strength_head(H))
output = local + alpha * sum_k(w_k * residual_expert_k(local))
```

C and Q each have 16 channels; H has 32 channels. Both w and alpha vary by
position and channel group (four groups by default). The four experts remain
3x3 depthwise convolutions with dilations 1, 2, 3, 4. Their independently
initialized weights predict residual corrections; no identity center is added.
The explicit skip preserves the original local feature. Small nonzero head
weights permit immediate gradients through R; the strength bias initializes
alpha near 0.1 and sigmoid keeps it between 0 and 1. Joint gating and residual
accumulation use FP32 under AMP; expert convolutions use the outer AMP policy.

`kernel_response_gate_stats.csv` retains the seven original routing fields and
adds `strength_mean`, `strength_spatial_std`, and `relative_update_rms`. These
are per-sample/group summaries of the first logged batch, not full-epoch
averages. `relative_update_rms` measures the RMS correction divided by the RMS
original local feature. Diagnostics do not change training gradients or RNGs.

Train a new model from scratch:

```bash
RESPONSE_VARIANT=joint_residual MAX_ITERS=200000 GPUS=0 BATCH_SIZE=4 \
  python -u train_online_2d.py
```

Do not pass `RESUME=1` or initialization weights for a fresh run. The default
experiment name contains `all8_joint_residual_v1_g4_v2`.

For paired evaluation, explicitly select the new architecture and a new result
directory (replace the checkpoint path with this run's saved model):

```bash
python -u test/test_gopro_hide.py \
  --weights /path/to/new_run/models/model_epoch_107.pth \
  --kernel_response_variant joint_residual \
  --kernel_response_groups 4 \
  --gpus 0 \
  --tile_size 256 --tile_overlap 64 --tile_mode native --tile_blend hann \
  --result_dir results/deblurring/Online2D/joint_residual_i200000_final_fixed_full \
  --existing_results error
```

The paired evaluator keeps `serial` as its low-level default so existing test
commands/checkpoints retain their architecture. Checkpoint metadata is checked
before strict weight loading. For the real-image evaluator, add
`"kernel_response_variant": "joint_residual"` to `model.kwargs` in a separate
config for the new model; existing real-image configs are not changed.

`KERNEL_RESPONSE=0` disables R only; the launcher still enables z0/z1. It is not
an ablation of all degradation guidance.
