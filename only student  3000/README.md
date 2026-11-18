# Uformer: A General U-Shaped Transformer for Image Restoration (CVPR 2022)
<b><a href='https://zhendongwang6.github.io'>Zhendong Wang</a>, <a href='https://vinthony.github.io'>Xiaodong Cun</a>, <a href='https://jianminbao.github.io/'>Jianmin Bao</a>, <a href='http://staff.ustc.edu.cn/~zhwg/'>Wengang Zhou</a>, <a href='http://people.ucas.ac.cn/~jzliu?language=en'>Jianzhuang Liu</a>, <a href='http://staff.ustc.edu.cn/~lihq/en/'>Houqiang Li </a> </b>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-realblur-j-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-j-trained-on-gopro?p=uformer-a-general-u-shaped-transformer-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-realblur-r-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-realblur-r-trained-on-gopro?p=uformer-a-general-u-shaped-transformer-for)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-denoising-on-dnd)](https://paperswithcode.com/sota/image-denoising-on-dnd?p=uformer-a-general-u-shaped-transformer-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-denoising-on-sidd)](https://paperswithcode.com/sota/image-denoising-on-sidd?p=uformer-a-general-u-shaped-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-gopro)](https://paperswithcode.com/sota/deblurring-on-gopro?p=uformer-a-general-u-shaped-transformer-for) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/deblurring-on-hide-trained-on-gopro)](https://paperswithcode.com/sota/deblurring-on-hide-trained-on-gopro?p=uformer-a-general-u-shaped-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-defocus-deblurring-on-dpd)](https://paperswithcode.com/sota/image-defocus-deblurring-on-dpd?p=uformer-a-general-u-shaped-transformer-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/uformer-a-general-u-shaped-transformer-for/image-enhancement-on-tip-2018)](https://paperswithcode.com/sota/image-enhancement-on-tip-2018?p=uformer-a-general-u-shaped-transformer-for)

Paper link: [[Arxiv]](https://arxiv.org/abs/2106.03106) [[CVPR]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf)


### Update:
* **2024.10.24** Fix expired links.
* **2022.07.06** Upload new codes and models for our Uformer. 
* **2022.04.09** Upload results of Uformer on denoising (SIDD, DND), motion deblurring (GoPro, HIDE, RealBlur-J/-R), and defocus deblurring (DPDD). 
* **2022.03.02** Uformer has been accepted by CVPR 2022! :fire:
* **2021.11.30** Update Uformer in [Arxiv link](https://arxiv.org/abs/2106.03106). The new code, models and results will be uploaded.
* **2021.10.28** Release the results of Uformer32 on SIDD and DND.
* **2021.09.30** Release pre-trained Uformer16 for SIDD denoising.
* **2021.08.19** Release a pre-trained model(Uformer32)! Add a script for FLOP/GMAC calculation.
* **2021.07.29** Add a script for testing the pre-trained model on the arbitrary-resolution images.

<hr>
<i>In this paper, we present Uformer, an effective and efficient Transformer-based architecture, in which we build a hierarchical encoder-decoder network using the Transformer block for image restoration. Uformer has two core designs to make it suitable for this task. The first key element is a local-enhanced window Transformer block, where we use non-overlapping window-based self-attention to reduce the computational requirement and employ the depth-wise convolution in the feed-forward network to further improve its potential for capturing local context. The second key element is that we explore three skip-connection schemes to effectively deliver information from the encoder to the decoder. Powered by these two designs, Uformer enjoys a high capability for capturing useful dependencies for image restoration. Extensive experiments on several image restoration tasks demonstrate the superiority of Uformer, including image denoising, deraining, deblurring and demoireing. We expect that our work will encourage further research to explore Transformer-based architectures for low-level vision tasks.</i>

![Uformer](fig/Uformer.png)

## Package dependencies
The project is built with PyTorch 1.9.0, Python3.7, CUDA11.1. For package dependencies, you can install them by:
```bash
pip install -r requirements.txt
```

## Pretrained model
- Uformer_B: [SIDD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/Ea7hMP82A0xFlOKPlQnBJy0B9gVP-1MJL75mR4QKBMGc2w?e=iOz0zz) |
[GoPro](https://mailustceducn-my.sharepoint.com/:u:/g/personal/zhendongwang_mail_ustc_edu_cn/EfCPoTSEKJRAshoE6EAC_3YB7oNkbLUX6AUgWSCwoJe0oA)

## Results from the pretrained model
- Uformer_B: [SIDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EtcRYRDGWhBIlQa3EYBp4FYBao7ZZT2dPc5k1Qe-CdPh3A) | [DND](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ekv3A5ic_4RChFa9XXquF_MB8M8tFd7spyHGJi_8obycnA) | [GoPro](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EqKY3WMkbfVBlzldiEe4IEUBgr6BQx8mkI9jipWoWrwqQg?e=c5aPIe) | [HIDE](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Eh4p1_kZ95xIopXDNyhl-Q0BrIW0DoAEh_v2g3WhHa51wA?e=qh1l7W) | [RealBlur-J](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EpHFC9FauEpHhJDsFruEmmQB9VVkGnKlWbaOmCmvH6Ar0w?e=vn9Na1) | [RealBlur-R](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Eo2EC8rmkapNu9CxcYLwFpYBAgiqMmaN8JPqnO6eYD5Abw?e=Pybep6) | [DPDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/EvVAI84ZvlNChWsZA6QY4IkBE4TArNRFkHj9VQVFadGWkw?e=eprhQe)


## Data preparation 
### Denoising
For training data of SIDD, you can download the SIDD-Medium dataset from the [official url](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php).
Then generate training patches for training by:
```python
python3 generate_patches_SIDD.py --src_dir ../SIDD_Medium_Srgb/Data --tar_dir ../datasets/denoising/sidd/train
```

For evaluation on SIDD and DND, you can download data from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ev832uKaw2JJhwROKqiXGfMBttyFko_zrDVzfSbFFDoi4Q?e=S3p5hQ).


### Deblurring
For training on GoPro, and evaluation on GoPro, HIDE, RealBlur-J and RealBlur-R, you can download data from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/zhendongwang_mail_ustc_edu_cn/Ev832uKaw2JJhwROKqiXGfMBttyFko_zrDVzfSbFFDoi4Q?e=S3p5hQ).


This workspace uses the following GoPro layout (the loader accepts the shown
folder names directly; no dataset rename is required):

```text
dataset/datasets/deblurring/GoPro/
├── crop/train/
│   ├── input_crops/    # blurred training images
│   └── target_crops/   # sharp training targets
└── test/
    ├── input/          # blurred validation images
    └── target/         # sharp validation targets
```

## Complete-model evaluation

Run the complete 600k-iteration model with the default paired UAV test set:

```bash
python test/test_gopro_hide.py
```

The defaults are relative to this repository, not the shell's working directory:

- Input: `dataset/UAV_single_image_pose_split/fixed_protocol_first_half_60_78_visdrone_all/test_fixed_full`, with matching `input/target` images.
- Weights: the full grouped-response experiment's `models/model_epoch_639.pth` under `logs/motiondeblur/Online2D/Uformer_B_single_image_online_2d_patch256_full_k12_r50_i600000_kr16_c64_all8_serial_g4_v2_complex_w0p01_18eeafbbe4/`.
- Output: `results/deblurring/Online2D/joint2d_i600000_final_fixed_full_Lwf4ZZ`.
- Architecture: Uformer-B with local branch, double process, channel-aware branch, DDEM stage 2 and four-group kernel-response control enabled.
- Inference: GPU 0, AMP FP16, native 256-pixel tiles, 64-pixel overlap, Hann blending; no resizing, TTA or GT input to the stage-2 model.
- Files: full-resolution `.PNG` restorations, `psnr_ssim_per_image.csv`, arithmetic mean metrics in `psnr_ssim.txt`, `test.log`, and `inference_config.json`.

PSNR and SSIM remain computed on fused/clamped float RGB images before PNG quantization, with no evaluation-border crop. SSIM uses the existing scikit-image call (`channel_axis=-1`, `data_range=1.0`, default 7x7 uniform window). This entry requires paired GT; it is not an unpaired DTB70 restoration entry.

Existing results are **not overwritten**: a nonempty output directory is first renamed to a sibling with `_backup_YYYYMMDD_HHMMSS_microseconds`, then a new directory with the requested name is created. This preserves all old restorations, metrics and scene summaries; backups are not automatically deleted. Prefer a new output directory if disk space is limited. Use `--existing_results error` to stop instead of backing up.

For this workspace, activate `joint-cu118-test`. Others should install the project dependencies and place the paired dataset/checkpoint in the indicated locations, or override only their paths:

```bash
python test/test_gopro_hide.py --input_dir /path/to/paired_test --weights /path/to/model.pth --result_dir results/my_test
python test/test_gopro_hide.py --check_only
python test/test_gopro_hide.py --help
```

`--check_only` checks the paired manifest and strictly loads the checkpoint on CPU without inference, writing results or moving old output directories. All previous positive CLI options remain available. Enabled-by-default switches also accept negative forms, such as `--no_save_images` or `--no-save-images`. For an architecture without kernel-response control, also select `--no_use_kernel_response --kernel_response_groups 1` and the appropriate other model switches; incompatible checkpoints fail strict loading rather than silently omitting layers.

## Training

### Online single-image 2D motion blur

See [online training configuration](docs/online_training.md) for iteration budgets,
learning-rate scheduling, checkpoint behavior, and resume instructions.

The UAV pipeline uses only sharp single images. Before each 50-epoch block,
every GT is assigned 12 independently seeded combinations of motion case,
direction, strength, and crop. Every training epoch randomly selects one of
those 12 cached candidates per GT. The motion generator renders a `256x256`
blurry patch directly from the full-resolution image and saves the sharp patch
from the identical coordinates. A crop is used only if all exposure subframes
map inside the source image, preventing black warp borders while preserving
strict Blur/GT registration.

The base trajectories are configured at medium blur strength. Rather than
scaling an entire trajectory with one number, every effective x/y coordinate,
drift, vibration/rotation amplitude, and attitude span independently samples a
factor in `[1.0, 1.8]`. The manifest stores the individual factors in
`strength_parameters`, while `strength` is their mean for summary purposes.
Relative to the earlier baseline, primitive cases 1--4 use `1.4x` spatial
motion and cases 5--7 use `1.8x`. Composite cases 8--15 inherit those calibrated
primitive components and are not multiplied a second time.

Online generation supports a CUDA backend: each physical GPU gets one isolated
process and exposure subframes are warped in batches with PyTorch `grid_sample`.
GPU and CPU outputs are numerically equivalent up to normal interpolation
rounding (at most one output gray level in the regression comparison). This
patch-centric workload remains CPU-default because a local 32-image benchmark
took 3.0 s with 12 CPU workers versus 14.1 s on two RTX 4090 GPUs; full-image
transfer and CUDA process overhead dominate the small 256-pixel output. Set
`BLUR_BACKEND=cuda` to select CUDA explicitly.

The launcher also creates a deterministic validation set once. Do not regenerate
validation every epoch, because changing samples would make validation PSNRs
incomparable. This validation split is for model selection; final generalization
should still be reported on a separate scene or external dataset.

Run from a Python environment that contains this model's PyTorch dependencies:

```sh
python train_online_2d.py
```

Common overrides are environment variables:

```sh
GPUS=0 BATCH_SIZE=4 GEN_WORKERS=16 MAX_ITERS=200000 python train_online_2d.py
```

The fixed protocol uses separate physical GT directories under
`dataset/UAV_single_image_pose_split/fixed_protocol_first_half_60_78_visdrone_all/`.
`DATASET_SCOPE=full` is the default and reads 7,520 GT images from
`train_full_7520`; `DATASET_SCOPE=fifth` reads 1,504 GT images from
`train_fifth_1504`. Each epoch selects one candidate per GT, giving 1,880 or 376
iterations respectively with global batch size 4. Candidate pools refresh every
50 epochs and are deleted after their final use.

The default training budget is 200,000 iterations, including 2,820 warmup
iterations. To use the previous epoch-based stopping rule explicitly:

```sh
MAX_ITERS=0 EPOCHS=500 python train_online_2d.py
```

Pool size and refresh interval can be overridden explicitly:

```sh
POOL_VARIANTS=12 POOL_REFRESH_EPOCHS=50 python train_online_2d.py
```

Set `RESUME=1` to resume from this experiment's `model_latest.pth`. Generated
training caches are removed after the corresponding 50-epoch block; set
`KEEP_CACHE=1` only when inspecting generated pairs.

### Denoising
To train Uformer on SIDD, you can begin the training by:

```sh
sh script/train_denoise.sh
```
### Deblurring
To train Uformer on GoPro, you can begin the training by:

```sh
sh script/train_motiondeblur.sh
```

The script defaults to GPUs 0 and 1, global batch size 8 (4 samples per GPU),
and learning rate `2e-4`. Override them together without editing the file, for
example, for a single-GPU debugging run:

```sh
GPUS=0 BATCH_SIZE=2 LR_INITIAL=0.00005 sh script/train_motiondeblur.sh
```
Extra model arguments are forwarded to Python, so the modified model can be
enabled with:

```sh
sh script/train_motiondeblur.sh --use_local_branch --double_process
```

### Optional Omni local branch and double process

The added experimental paths are disabled by default, so existing Uformer
commands and checkpoints keep the original architecture. Enable both paths for
training with:

```sh
sh script/train_motiondeblur.sh \
  --use_local_branch \
  --double_process \
  --local_expansion 2 \
  --disable_validation
```

With `--disable_validation`, training skips both the initial validation and all
periodic validation. It does not create `model_best.pth` or write
`model_latest.pth` after every epoch; checkpoints are saved only at the
configured `--checkpoint` interval.

`--use_local_branch` uses `DWConv -> PWConv -> PWConv -> DWConv` features as
window-attention K/V and as the Omni-style local residual. `--double_process`
runs a second independently-parameterized attention-residual + LeFF-residual
process only in the last block of each encoder/decoder stage; the bottleneck
remains single-process. Use the same flags when evaluating a checkpoint trained
with these paths enabled.

### MODEM-style DDEM teacher/student training

DDEM stage 1 estimates degradation from the paired blurry/GT input and guides
Uformer Q/K while training the restoration network:

```sh
sh script/train_motiondeblur.sh \
  --use_local_branch --use_channel_aware --double_process \
  --use_ddem --ddem_stage 1 --disable_validation \
  --env _omni_stage_last_cam2x_ddem_teacher_b8_e3000
```

Stage 2 initializes the restorer/student from the stage-1 checkpoint, freezes a
six-channel teacher DDEM, and distills its pre-compression descriptor into the
three-channel blurry-only student DDEM:

```sh
sh script/train_motiondeblur.sh \
  --use_local_branch --use_channel_aware --double_process \
  --use_ddem --ddem_stage 2 \
  --ddem_stage1_weights ./logs/motiondeblur/GoPro/Uformer_B_omni_stage_last_cam2x_ddem_teacher_b8_e3000/models/model_epoch_3000.pth \
  --disable_validation \
  --env _omni_stage_last_cam2x_ddem_student_b8_e3000
```

At inference only the stage-2 student is used. `Z0` applies an identity-centered
affine modulation to every attention Q, and `Z1` applies a sample-adaptive
32x32 matrix to K generated from the Local Branch. V, the local residual, CAM,
and LeFF remain unchanged. Inspect prior variation and each layer's Q/K change:

```sh
python tools/inspect_ddem_priors.py \
  --weights ./logs/motiondeblur/GoPro/Uformer_B_omni_stage_last_cam2x_ddem_student_b8_e3000/models/model_epoch_500.pth \
  --num_samples 8
```


## Evaluation
To evaluate Uformer, you can run:

```sh
sh script/test.sh
```
For evaluate on each dataset, you should uncomment corresponding line.

## Computational Cost

We provide a simple script to calculate the flops by ourselves, a simple script has been added in `model.py`. You can change the configuration and run:

```python
python3 model.py
```

> The manual calculation of GMacs in this repo differs slightly from the main paper, but they do not influence the conclusion. We will correct the paper later.


## Citation
If you find this project useful in your research, please consider citing:

```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zhendong and Cun, Xiaodong and Bao, Jianmin and Zhou, Wengang and Liu, Jianzhuang and Li, Houqiang},
    title     = {Uformer: A General U-Shaped Transformer for Image Restoration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17683-17693}
}
```

## Acknowledgement

This code borrows heavily from [MIRNet](https://github.com/swz30/MIRNet) and [SwinTransformer](https://github.com/microsoft/Swin-Transformer).


## Contact
Please contact us if there is any question or suggestion(Zhendong Wang ZhendongWang6@outlook.com, Xiaodong Cun vinthony@gmail.com).
