#!/bin/sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset/datasets/deblurring/GoPro}"
TRAIN_DIR="${DATA_ROOT}/crop/train"
VAL_DIR="${DATA_ROOT}/test"
GPUS="${GPUS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR_INITIAL="${LR_INITIAL:-0.0002}"

for required_dir in \
    "${TRAIN_DIR}/input_crops" \
    "${TRAIN_DIR}/target_crops" \
    "${VAL_DIR}/input" \
    "${VAL_DIR}/target"; do
    if [ ! -d "${required_dir}" ]; then
        echo "Missing GoPro dataset directory: ${required_dir}" >&2
        exit 1
    fi
done

cd "${PROJECT_ROOT}"

python3 "${PROJECT_ROOT}/train/train_motiondeblur.py" \
    --arch Uformer_B --batch_size "${BATCH_SIZE}" --gpu "${GPUS}" \
    --lr_initial "${LR_INITIAL}" \
    --train_ps 256 --train_dir "${TRAIN_DIR}" \
    --val_ps 256 --val_dir "${VAL_DIR}" --env _omni_local_2gpu_b8 \
    --save_dir "${PROJECT_ROOT}/logs" \
    --mode deblur --nepoch 3000 --checkpoint 500 --dataset GoPro --warmup \
    "$@"
