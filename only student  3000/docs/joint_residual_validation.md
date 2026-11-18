# Joint residual R controller: implementation checks

Validated on 2026-11-14. These checks establish execution and compatibility;
they do not establish a PSNR improvement or replace a full training run.

## Architecture

- Variant: `joint_residual`; legacy `serial` remains explicit and unchanged.
- Joint gate: projected content/response/product, 48 -> 32 -> grouped heads.
- Output: `local + sigmoid(strength) * sum(expert_probability * residual)`.
- Initial strength: approximately 0.1; probabilities initially near uniform.
- Experts: the original four 3x3 depthwise types, dilations 1, 2, 3, 4.
- New parameters relative to the four-group serial model: 13,600 across eight
  controllers. The response head, labels, z0/z1, and injection order are unchanged.

## Completed checks

1. New regression suite: **11/11 passed**, including a CUDA AMP controller test,
   restoration-only gradients into every controller component and R, group
   mapping, window alignment, identity when residuals are zero, and strict
   checkpoint round trips. Diagnostics preserve outputs, gradients, and RNGs.
2. Compatibility against the user's read-only `only student  3000 copy`:
   serial weights/outputs match exactly; shared backbone, DDEM and response-head
   initialization/global RNG are preserved for the new controller.
3. Existing paired-evaluation entrypoint suite: **9/9 passed**.
4. Iteration-schedule suite: **4/4 passed**.
5. Launcher dry runs: default/new, explicit serial, R disabled, and new
   single-group configurations all parse successfully. New runs use independent
   experiment/cache names. R disabled does not disable z0/z1.
6. Production **Uformer_B**, all eight controllers, on an NVIDIA RTX 4090 D:
   **three AMP forward/backward/AdamW updates with batch 4 at 256x256** passed.
   All checked gradients were finite and nonzero, including routing/strength
   heads and experts. GradScaler stayed at 65,536. Peak allocated memory was
   **17,612.2 MiB**. This is a single-GPU check, not a two-GPU performance claim.
   A 128x256 rectangular inference also passed.
7. Actual training script, using its unchanged motion generator and temporary
   synthetic source: 2-iteration fresh training and resume from iteration 1
   passed, with and without validation. Final batch count, scheduler/scaler
   state, variant metadata, best checkpoint, and all ten CSV fields were checked.
8. Actual paired-evaluation `--check_only` strictly loaded the new best model.
   It also strictly loaded the previous production serial 600k checkpoint on
   the existing 1,790-image test set, without creating output directories.
9. Syntax and `git diff --check` passed for the edited sources. Dataset loader
   and online pool orchestration hashes were checked unchanged. The user's
   backup was only read. Smoke-training caches, logs and weights were in `/tmp`.

The older `test_grouped_kernel_response.py` contains references to the absent
historical `/home/xtl/hst/joint2D copy` fixture: two reference-dependent checks
could not run as written. Equivalent controller/model parity checks were added
to the new suite using the user's current backup and passed. The old fixture
paths and archived tests were not rewritten.

## Re-run focused tests

From the project root, in the `joint-cu118-test` environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p 'test_joint_residual_response.py' -v
```

The CUDA check is opt-in; select an idle GPU before enabling it:

```bash
CUDA_VISIBLE_DEVICES=1 RUN_CUDA_TESTS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -p 'test_joint_residual_response.py' -v
```
