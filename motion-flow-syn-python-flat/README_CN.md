

# 自动生成mask
前脚本默认自动识别 car，因为配置中写了：

  text_prompt = "car."

  GroundingDINO 是开放词汇检测模型，可以通过 --text-prompt 改成其他目标。SAM2 不判断类别，只负责把 GroundingDINO 找到的框转换成精确 mask。

  ## 识别人

  当前 UAV 图中的人比较小，需要降低面积过滤阈值：

  CUDA_VISIBLE_DEVICES=0 python script_uav_generate_mask_groundingdino_sam2.py \
      --image dataset/object_motion_A/image/1.jpg \
      --text-prompt "person." \
      --output-dir dataset/object_motion_A/mask_auto_person \
      --debug-dir dataset/object_motion_A/mask_auto_person_debug \
      --min-box-area-ratio 0.0003 \
      --min-mask-area-ratio 0.0003 \
      --max-objects 1

  输出：

  mask_auto_person/1.png
  mask_auto_person_debug/1_overlay.png

  ## 识别船

  建议同时使用 boat 和 ship 两个描述：

  CUDA_VISIBLE_DEVICES=0 python script_uav_generate_mask_groundingdino_sam2.py \
      --image dataset/object_motion_A/image/boat.jpg \
      --text-prompt "boat. ship." \
      --output-dir dataset/object_motion_A/mask_auto_boat \
      --debug-dir dataset/object_motion_A/mask_auto_boat_debug \
      --min-box-area-ratio 0.0003 \
      --min-mask-area-ratio 0.0003 \
      --max-objects 0

  --max-objects 0 表示保留所有符合要求的船，并将它们合并成一张二值 mask。

  ## 同时识别多种目标

  CUDA_VISIBLE_DEVICES=0 python script_uav_generate_mask_groundingdino_sam2.py \
      --image dataset/object_motion_A/image/test.jpg \
      --text-prompt "car. person. boat. ship." \
      --output-dir dataset/object_motion_A/mask_auto_all \
      --debug-dir dataset/object_motion_A/mask_auto_all_debug \
      --min-box-area-ratio 0.0003 \
      --min-mask-area-ratio 0.0003 \
      --max-objects 0

  多个类别之间用英文句号分隔。常见提示词例如：

  person.
  car.
  truck.
  bus.
  motorcycle.
  bicycle.
  boat.
  ship.
  airplane.
  helicopter.

  尽量使用英文类别名称。

  需要注意：当前脚本在选择多个目标时，会把它们合并成一张 mask。如果以后需要：

  1_car_01.png
  1_person_01.png
  1_boat_01.png

  这种“每个实例单独保存”的格式，还需要给脚本增加实例 mask 输出功能。另外，同一张图更换类别运行时，应使用不同的 --output-dir，否则都会写成 1.png，后一次会覆盖前一次。


# 自动生成mask参数含义
--max-objects 0 与面积无关，它表示“不限制最终保留的目标数量”。

  “降低面积过滤阈值”指的是这两个参数：

  --min-box-area-ratio 0.0003
  --min-mask-area-ratio 0.0003

  它们的区别：

   参数                     含义
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   --min-box-area-ratio     检测框面积至少占整张图的比例
  ───────────────────────  ───────────────────────────────────────
   --min-mask-area-ratio    SAM2 最终 mask 面积至少占整张图的比例
  ───────────────────────  ───────────────────────────────────────
   --max-objects            最终最多保留几个目标

  你的图像是 640×480，总计：

  307200 像素

  默认阈值 0.002 相当于：

  307200 × 0.002 ≈ 614 像素

  小于约 614 像素的目标会被过滤。图中的人比较小：

  人的 mask 大约 321 像素
  面积比例约 0.001

  所以默认的 0.002 会把人删除。

  改成 0.0003 后：

  307200 × 0.0003 ≈ 92 像素

  321 像素的人就可以通过面积筛选。

  识别单个人建议：

  python script_uav_generate_mask_groundingdino_sam2.py \
      --image dataset/object_motion_A/image/1.jpg \
      --text-prompt "person." \
      --min-box-area-ratio 0.0003 \
      --min-mask-area-ratio 0.0003 \
      --max-objects 1 \
      --output-dir dataset/object_motion_A/mask_auto_person \
      --debug-dir dataset/object_motion_A/mask_auto_person_debug

  其中：

  --max-objects 1：只保留综合得分最高的一个人
  --max-objects 0：保留所有通过筛选的人，并合并为一张 mask

  面积阈值不能降得太低，否则远处的小纹理、阴影和路面物体也更容易成为误检。


# motion-flow-syn Python 平铺对照版

本工程以 `motion-flow-syn-master.zip` 中的 MATLAB/C++ 源码为转换依据重新编写。

这一版的结构目标是方便和 MATLAB 原文件互相查找：

- MATLAB 的主脚本在 Python 中保留同名文件；
- 你自行编写的 `script_uav_*.m` 仍采用“用户配置区 → 主流程 → 本地函数区”；
- 轨迹、warp、裁边、保存等关键逻辑都留在对应主脚本中；
- 只有 MATLAB 原工程本来就独立的 `gen_mf`、`utils` 和 `conv_opt` 继续独立；
- 输出名称、MAT 变量名和目录语义尽量保持原样。

## 目录结构

```text
motion-flow-syn-python-flat/
├── script_data_gen.py
├── demo_flow_global_blur_no_mex.py
├── script_uav_global_blur_repro_smooth_flow.py
├── script_uav_global_blur_multisubframe_no_depth.py
├── script_uav_object_motion_blur_framework_A.py
├── make_simple_object_test_data.py
├── view_uav_global_blur_flow_by_path.py
├── view_uav_multisubframe_flow_by_path.py
├── view_uav_multisubframe_all_cases_global_value_consistent_v4.py
├── gen_mf.py
├── gen_mf_para.py
├── gen_blurry_img.py
├── conv_opt/
│   └── motion_blur_conv.py
├── utils/
│   ├── mfuv2mo.py
│   ├── flow_to_color.py
│   ├── draw_mf_on_img.py
│   ├── plotflow.py
│   └── matplotlib_cn.py
└── tests/test_core.py
```

## 安装

推荐 Python 3.10 或更新版本。

Linux：

```bash
cd /home/xtl/hst/motion-flow-syn-python-flat
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
cd motion-flow-syn-python-flat
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

第一次运行空间变化卷积时，Numba 会编译原 MEX 的 Python 实现，因此首次调用会稍慢。

## 1. 原始 motion-flow-syn 数据生成

默认输入：

```text
dataset/image/train
dataset/image/test
dataset/image/val
```

运行：

```bash
python script_data_gen.py
```

每张图默认生成 50 个运动模糊样本和 1 个清晰样本。快速测试：

```bash
python script_data_gen.py --num-mf 1 --seed 1
```

## 2. 五类 UAV 空间变化核模糊

```bash
python script_uav_global_blur_repro_smooth_flow.py
```

默认输出：

```text
dataset/data_syn_uav_global_repro_smooth_flow/{train,test,val}
```

默认使用连续 flow。需要模仿原 `script_data_gen.m` 的整数标签时：

```bash
python script_uav_global_blur_repro_smooth_flow.py --quantize-flow
```

仅处理 train 和某一种模糊：

```bash
python script_uav_global_blur_repro_smooth_flow.py \
  --splits train \
  --cases 04_maneuver_spatial_field
```

## 3. 五类 UAV 全局多子帧积分

```bash
python script_uav_global_blur_multisubframe_no_depth.py
```

默认输出：

```text
dataset/data_syn_uav_global_multisubframe_v2/{train,test,val}
```

修改曝光采样数：

```bash
python script_uav_global_blur_multisubframe_no_depth.py --n-subframes 31
```

高频振动建议使用 21 或 31 个子帧。该脚本保存：

- `mfmap`：曝光起点到终点的等效位移；
- `motion_summary.path_len`：完整曝光轨迹长度；
- `motion_summary.span_mag`：轨迹覆盖跨度；
- `para.dx_list/dy_list/theta_list/scale_list`：各子帧参数。

对于往复振动，`mfmap` 可能接近零，但 `path_len` 仍然较大。

## 4. 单目标独立运动方案 A

先生成测试数据：

```bash
python make_simple_object_test_data.py --no-show
```

然后生成五类目标运动模糊：

```bash
python script_uav_object_motion_blur_framework_A.py
```

默认目录：

```text
dataset/object_motion_A/image
dataset/object_motion_A/mask
dataset/object_motion_A/data_syn
```

背景补全可选：

```bash
--inpaint-method biharmonic  # 默认，更接近 MATLAB regionfill 的 PDE 思路
--inpaint-method telea       # 更快，适合批量生成
```

第一阶段使用已知真实背景、跳过 inpaint：

```bash
python script_uav_object_motion_blur_framework_A.py \
  --background-dir dataset/object_motion_A/background_gt \
  --cases 01_obj_linear_motion \
  --n-subframes 11 \
  --no-soft-mask \
  --no-crop-boundary
```

真实背景按输入图的完整文件名匹配；如果扩展名不同，会继续查找同 stem
的 PNG 或其他受支持图像。指定目录后如果背景缺失或尺寸不一致，程序会直接
报错，不会悄悄退回 inpaint。

## 5. 查看结果

空间变化核单样本：

```bash
python view_uav_global_blur_flow_by_path.py \
  --data-dir dataset/data_syn_uav_global_repro_smooth_flow/train \
  --image-stem 8143 \
  --blur-case 04_maneuver_spatial_field
```

多子帧单样本：

```bash
python view_uav_multisubframe_flow_by_path.py \
  --data-dir dataset/data_syn_uav_global_multisubframe_v2/train \
  --image-stem 8143 \
  --blur-case 05_high_frequency_vibration
```

两个单样本查看脚本都可以在文件顶部的“用户配置区”直接修改 `blur_case`，也可以使用命令行 `--blur-case`。可选值为：

```text
01_linear_uniform
02_linear_variable_speed
03_hover_jitter_nonuniform
04_maneuver_spatial_field
05_high_frequency_vibration
```

注意：`05_high_frequency_vibration` 的曝光起点和终点可能重合，因此起止方向无定义。此时应查看 `path_len` 和 `span_mag`，方向页会显示原因说明。

查看脚本的默认输出统一保存在工程的 `dataset` 下：

```text
dataset/flow_vis_output
dataset/flow_vis_output_global_value_consistent
```

保存路径根据脚本自身位置计算，因此从其他工作目录运行也不会改变默认输出位置。

五类统一颜色范围汇总：

```bash
python view_uav_multisubframe_all_cases_global_value_consistent_v4.py \
  --data-dir dataset/data_syn_uav_global_multisubframe_v2/train \
  --image-stem 8143
```

## 输出兼容性

生成脚本保留 MATLAB 风格输出：

```text
*_gtimg.png
*_blurryimg.png
*_mfmap.mat
*_mf_para.mat
```

MAT 文件继续使用 `mfmap`、`para`、`mf`、`traj`、`motion_summary` 等原变量名，Python 查看脚本和 MATLAB 查看脚本都可读取。

## 自动测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 MEX 卷积恒等性、原始运动场、MATLAB Twister 相位、warp、完整轨迹统计和目标多子帧流程。

## MATLAB 与 Python 的数值差异

- 空间变化卷积按原 `.cc` 公式重写，参考样本基本达到逐像素一致；
- MATLAB `interp2` 与 OpenCV `remap` 都使用双线性插值，但内部取整不同，多子帧 PNG 可能相差 1 个灰度级；
- `regionfill` 没有直接的 Python 逐像素等价函数，本工程默认使用 biharmonic 补全；
- 原 `gen_mf_para.m` 每次调用 `rng('shuffle')`，Python 增加了可选 `--seed`，便于复现实验。

详细数值见 `REFERENCE_COMPARISON.md`。



