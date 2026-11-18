# Incremental Online2D ablations at 200k iterations

Run from the project root in the training Python environment. Prepare the GT
and fixed validation data under the paths in `train_online_2d.py` first.

`train_ablation_2d.py` selects architecture flags explicitly; it does not infer
an ablation by changing `KERNEL_RESPONSE`. The model preset determines whether
R is enabled. Default budget is 200,000 batches, single GPU 0 with global batch 4, LR 3e-4,
warmup 2,820 batches, cosine decay to 1e-6, AdamW weight decay 0.02, and
gradient clipping 1.0 for every architecture. KD is disabled; DDEM models
are trained from scratch. The data generator, pool size and refresh protocol
are shared. Each preset has separate experiment and training-cache paths. Names include
`_b4` to distinguish this protocol from batch-8 experiments.

| Preset | Local branch | CAM | Double process | z0/z1 (DDEM) | R |
| --- | --- | --- | --- | --- | --- |
| `full` | yes | yes | yes | yes | yes |
| `uformer` | no | no | no | no | no |
| `local` | yes | no | no | no | no |
| `local_cam` | yes | yes | no | no | no |
| `local_cam_z0z1` | yes | yes | yes | yes | no |
| `local_cam_double` | yes | yes | yes | no | no |

The original Uformer preset means the base architecture in this codebase,
using the common ablation training protocol. It does not reproduce the
separate author's 789k schedule.

The z0/z1 adapters currently operate in the second process. Consequently,
`local_cam_z0z1` adds both that process and the prior estimator compared with
`local_cam`. Include `local_cam_double` to isolate the effect of adding z0/z1.
Compare `local_cam_z0z1` against `full` to measure adding R.

## Commands

```bash
RESUME=0 MAX_ITERS=200000 RESPONSE_VARIANT=joint_residual RESPONSE_GROUPS=4 GPUS=0 BATCH_SIZE=4 python -u train_ablation_2d.py --model full
RESUME=0 MAX_ITERS=200000 GPUS=0 BATCH_SIZE=4 python -u train_ablation_2d.py --model uformer
RESUME=0 MAX_ITERS=200000 GPUS=0 BATCH_SIZE=4 python -u train_ablation_2d.py --model local
RESUME=0 MAX_ITERS=200000 GPUS=0 BATCH_SIZE=4 python -u train_ablation_2d.py --model local_cam
RESUME=0 MAX_ITERS=200000 GPUS=0 BATCH_SIZE=4 python -u train_ablation_2d.py --model local_cam_z0z1
```

Optional control:

```bash
RESUME=0 MAX_ITERS=200000 GPUS=0 BATCH_SIZE=4 python -u train_ablation_2d.py --model local_cam_double
```

Run each command separately, or let them execute sequentially. Do not run
several training commands concurrently on the same GPUs. Set the same effective
global batch size across machines. For this protocol with 7,520 samples, the
final snapshot is `model_epoch_107.pth` (106 complete epochs plus 720 batches).
Validation is every 25 epochs and on the terminal batch. For model selection,
use the same rule (e.g. final snapshot or validation best) across presets.

`RESUME=1` restores the same preset's latest checkpoint; `RUN_TAG` can name
an independent repeated run. `DRY_RUN=1` prints the fully resolved command
without starting training, but still validates dataset locations.

## Verification

The six presets were checked for exact enabled/disabled flags, budget,
initialization options, and distinct paths. Small actual models for all six
completed forward/backward, clipping and AdamW updates on CPU. The clipping
expression is tested to produce the same threshold with or without DDEM.
This is a configuration/execution check, not a full 200k training result.

```bash
python -m unittest discover -s tests -p test_ablation_presets.py -v
```

At batch size 4, 200k iterations process 800,000 samples, half the count
of batch size 8 at the same iteration budget. Warmup stays at 2,820 iterations;
the LR remains 3e-4. There is no gradient accumulation.
