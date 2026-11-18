#!/bin/sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TEACHER_WEIGHTS="${PROJECT_ROOT}/logs/motiondeblur/GoPro/Uformer_B_omni_stage_last_cam2x_lgddem_featurelocal_stageproj_teacher_scratch_equalLR_b8_e3000/models/model_epoch_3000.pth"

if [ ! -f "${TEACHER_WEIGHTS}" ]; then
    echo "Teacher checkpoint does not exist: ${TEACHER_WEIGHTS}" >&2
    exit 1
fi

export GPUS="${GPUS:-0,1}"
export BATCH_SIZE="${BATCH_SIZE:-8}"

exec sh "${PROJECT_ROOT}/script/train_motiondeblur.sh" \
    --use_local_branch \
    --use_channel_aware \
    --double_process \
    --use_ddem \
    --ddem_stage 2 \
    --ddem_dim 32 \
    --ddem_blocks 2 \
    --ddem_backbone local_global \
    --ddem_heads 1 \
    --ddem_win_size 8 \
    --ddem_stage1_weights "${TEACHER_WEIGHTS}" \
    --ddem_kd_weight 0.0 \
    --ddem_temperature 0.15 \
    --kd_grad_stat_interval 110 \
    --adapter_warmup_epochs 0 \
    --backbone_lr_scale 1.0 \
    --lr_initial 0.0002 \
    --grad_clip 1.0 \
    --disable_validation \
    --checkpoint 500 \
    --nepoch 3000 \
    --env _student_teacher3000_kd0_gradstat110_b8_e3000
