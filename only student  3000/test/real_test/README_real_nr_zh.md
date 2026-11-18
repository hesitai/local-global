# 真实图像去模糊：无真值评测

评测入口：`evaluate_real_nr.py`。只需填入测试图像目录、模型构造入口与权重。
不需要 GT。输出恢复 PNG、逐图指标、均值/标准差/有效数量，以及人工 MOS 评分模板。

## 1. 文件与入口

| 文件 | 作用 |
| --- | --- |
| `real_nr_config.json` | 填写路径、模型结构与推理设置 |
| `nr_model_adapter.py` | 模型构造、严格加载 checkpoint、推理入口 |
| `evaluate_real_nr.py` | 遍历测试集、保存恢复图、执行评测 |
| `nr_metrics.py` | NIQE、BRISQUE、LV、MUSIQ、CPBD、Entropy |
| `nr_results.py` | 逐图记录、总体统计、人工 MOS 导入 |
| `requirements_real_nr.txt` | 独立评测环境依赖 |

下面命令均在项目根目录运行：

```bash
cd '/home/xtl/hst/joint2D/only student  3000'
```

配置文件中的相对路径一律以该项目根目录为基准；命令行 `--config` 本身的相对路径以当前工作目录为基准。
原有测试脚本和训练依赖文件没有修改。

## 2. 环境

在具有相互匹配的 PyTorch、torchvision 的 Python 3.10/3.11 环境中：

```bash
python -m pip install -r test/real_test/requirements_real_nr.txt
```

NIQE、BRISQUE、MUSIQ 使用固定的 `pyiqa==0.1.13`，首次创建指标时需要下载相应的参数/统计模型，后续可用缓存。
LV、CPBD、Entropy 不需要学习到的模型权重；CPBD 已内置，不需要安装存在旧 SciPy 导入问题的 `cpbd` 包。

**建议将指标环境与训练环境分开。** 本项目训练依赖固定 `timm==0.6.13`，PyIQA 0.1.13 要求 `timm>=0.8`。
可以先在原模型环境运行 `restore_only`，然后在独立指标环境运行 `evaluate`，见第 5 节。
若选择一体化 `restore` 模式，环境必须同时兼容模型与 PyIQA；本次没有安装或升级现有环境。

## 3. 填写测试集和模型

编辑 `test/real_test/real_nr_config.json`，至少填写：

```json
{
  "input_dir": "/你的真实模糊图目录",
  "weights": "/你的模型权重/model.pth",
  "output_dir": "results/real_nr/your_method",
  "model": {
    "factory": "your_model_module:YourModel",
    "kwargs": {},
    "checkpoint_key": "auto",
    "output_key": null,
    "output_index": -1,
    "pad_multiple": 128,
    "square_pad": false,
    "tile_size": 0,
    "tile_overlap": 32,
    "tile_batch_size": 4,
    "amp": false
  }
}
```

以上片段用于说明要修改的字段；请保留配置文件里的其他字段。
`factory` 是“模块名:类名或构造函数名”，`kwargs` 是模型构造参数，必须与训练时的网络结构完全一致。
例如 `model:Uformer` 会从项目的 `model.py` 导入 `Uformer`；需在 `kwargs` 中补齐你训练使用的结构参数。
模型模块应能从项目根目录导入。也可直接修改 `nr_model_adapter.py` 中的 `build_model()` 与 `predict()`，接入特殊网络。

加载器支持裸 state_dict 及 `params_ema`、`params`、`state_dict`、`model_state_dict`、`model`、`net` 包裹格式；
自动模式按这个顺序选择第一个字典，并打印选中的字段。如需指定普通/EMA 权重，用 `checkpoint_key` 指定，支持点分嵌套键。
会去掉统一的 `module.` 前缀，再执行 `strict=True`，结构不匹配会直接报错。
使用 `torch.load(weights_only=True)` 读取 Tensor checkpoint；带自定义对象的旧 checkpoint 需先导出纯 state_dict，或在模型入口按自己的可信格式加载。

模型入口约定：输入/输出均为 RGB、`[N,3,H,W]`、范围 `[0,1]`；N 为一批的图块数。
默认使用 float32，恢复模型可选 CUDA FP16 混合精度；各项指标仍按原精度计算。
若网络需要其他归一化或额外输入，在 `predict()` 内完成，并将返回值还原到上述约定。
多阶段网络返回 tuple/list 时默认取最后一项，可设 `output_index`；返回字典时须设 `output_key`。
TorchScript 文件可设 `factory="torchscript"`，直接填写其 `weights` 路径。

推理保持原图尺寸：右侧/下侧复制边界补齐到 `pad_multiple`，模型推理后去掉补齐部分。
`square_pad=true` 可补到正方形；模型不需要补齐时可设 `pad_multiple=1`。
显存不够可设 `tile_size=256`、`tile_overlap=32` 或 `64`；重叠区域使用等权平均。
分块会改变模型上下文，正式比较时应固定推理设置，并在论文中说明。

### 加速测试

`tile_batch_size=4` 表示将四个图块合并成一次模型前向；图块位置、重叠范围、累加顺序和 FP32 融合方式保持不变。
例如 4000×2250 图像使用 256 图块、64 重叠，一共 252 块，批量为 4 时只需 63 次前向调用。
显存不足时程序会减半图块批量并重试；自定义模型若只接受 batch=1，可设置 `--tile-batch-size 1`。

可先在新目录试跑 3 张，开启恢复模型的混合精度：

```bash
python test/real_test/evaluate_real_nr.py --max-images 3 \
  --tile-batch-size 4 --amp \
  --output-dir results/real_ours_speedcheck
```

`--amp` 只影响恢复模型，可能产生少量数值差异；正式比较应统一精度设置。
如果希望使用 FP32，则将 `--amp` 换成 `--no-amp`。本次没有 GPU 加速倍数实测；请以试跑的实际耗时为准。
图块大小和重叠范围也可通过 `--tile-size`、`--tile-overlap` 覆盖，但修改它们会改变模型上下文及恢复结果。
CPBD 已将独立边缘的扫描向量化；实际 4000×2250 恢复图 CPU 测试从约 1.90 秒降至 1.13 秒，得分与旧版完全一致。

## 4. 运行并查看输出

```bash
python test/real_test/evaluate_real_nr.py --config test/real_test/real_nr_config.json
```

也可以覆盖配置中的路径：

```bash
python test/real_test/evaluate_real_nr.py \
  --input-dir /你的测试集 \
  --weights /你的权重/model.pth \
  --output-dir results/real_nr/your_method
```

输入支持递归读取 PNG/JPG/JPEG/BMP/TIF/TIFF/WebP，按相对路径排序，每次处理一张图。
标准灰度图会转换为 RGB；16 位/浮点图像、透明图像需要先按固定规则转换成 8 位可视图，避免静默改变数值范围。
EXIF 方向会先纠正。一个文件视为一张图，不支持多页 TIFF 栈的逐页评测。
建议输入为至少 `192×192` 的图像，以满足 NIQE 的块统计需要；脚本不会为凑出指标而放大小图。

输出结构：

```text
results/real_nr/your_method/
├── restored/                   # 原尺寸、8 位 RGB 无损 PNG
│   └── scene/a.jpg.png         # 保留原扩展名，避免 a.jpg/a.png 重名覆盖
├── per_image_metrics.csv       # 每图七项指标、尺寸、状态、错误及 MOS 评分人数
├── summary.csv                 # 每项均值、标准差、有效数、缺失数
├── summary.json                # 汇总及配置、依赖版本、设备、权重哈希
├── summary.txt                 # 方便阅读的总体汇总
├── timings.csv                 # 逐图读取、恢复、保存、六项指标与总耗时
├── run_config.json             # 启动配置与环境快照
├── mos_ratings_template.csv    # 待人工填写的评分表
└── errors.log                  # 仅在图像读取/推理失败时生成
```

全部客观指标基于**实际保存后重新读取的 PNG**，而非保存前浮点 Tensor；不对评测图额外 resize。
MUSIQ 等模型会执行其标准内部预处理。输出目录必须在测试集目录之外；非空输出目录会报错，请用新目录保留每次实验。
逐图表在每张图处理后追加保存，因此中途停止也能保留已完成记录。
总体表在首张图、每 `report_every` 张（默认 50）及结束时刷新，结束时包含全部记录。

## 5. 模型推理与指标计算分开运行

第一步，在原模型环境中仅生成恢复图，不导入 PyIQA：

```bash
python test/real_test/evaluate_real_nr.py --mode restore_only \
  --output-dir results/real_nr/your_method_images
```

第二步，在安装上述依赖的评测环境中计算六项客观指标；无须导入网络代码或加载模型权重：

```bash
python test/real_test/evaluate_real_nr.py --mode evaluate \
  --input-dir results/real_nr/your_method_images/restored \
  --output-dir results/real_nr/your_method_metrics
```

`evaluate` 模式直接评价已有恢复图，CSV 的 `restored_path` 指向这些文件，不再复制图像。
其 `image_id` 是相对于这一步输入目录的路径，例如 `scene/a.jpg.png`；请使用**第二步生成的 MOS 模板**保持编号一致。
`restore_only` 的指标列为空，它的均值表不能用于报告客观指标结果。

## 6. MOS：人工评分后再汇总

MOS 是 Mean Opinion Score（平均主观评分），代码不从图像预测或伪造 MOS。
程序自动生成空白评分表；将其复制为 `mos_ratings.csv` 后填写，默认采用 1–5 分，高分表示质量更好。
每位评分者对每张图填一行，同一图像允许多位评分者：

```csv
image_id,rater_id,score
scene/a.jpg,rater01,4
scene/a.jpg,rater02,5
scene/b.png,rater01,3
scene/b.png,rater02,4
```

请保留模板中的 `image_id`；上述值只是格式示例。推荐统一评分标准、显示条件及评分人数，并随机展示图像、隐藏方法名称。
如果采用 0–100 等其他量表，在配置中修改 `mos_min`、`mos_max`，保持分数越高越好。
不填写评分时，逐图 MOS 留空，汇总显示 `N/A`，有效数为 0。

人工评分完成后，仅合并 MOS 和更新均值，无须重新计算客观指标：

```bash
python test/real_test/evaluate_real_nr.py --summarize-only \
  --output-dir results/real_nr/your_method \
  --mos-csv /你的评分表/mos_ratings.csv
```

也可在第一次运行时配置 `mos_csv`。同一图像/评分者重复、评分越界、非有限数值、未知图像编号会报错。
空白评分行忽略。每图 MOS 先对该图评分者取均值，总体 MOS 再对已评分图像等权平均，避免评分人数多的图像获得更大权重。

## 7. 指标定义与统计规则

| 指标 | 方向 | 本代码约定 |
| --- | --- | --- |
| NIQE | ↓ | PyIQA `niqe` 默认统计模型；至少两个完整 `96×96` 块 |
| BRISQUE | ↓ | PyIQA `brisque` 默认版本，非另一个 `brisque_matlab` 变体 |
| LV | ↑ | 8 位灰度，`cv2.Laplacian(..., CV_64F, ksize=1, BORDER_DEFAULT).var()` |
| MUSIQ | ↑ | PyIQA `musiq` 默认 KonIQ-10k 权重 |
| CPBD | ↑ | 内置 python-cpbd 兼容实现，`64×64` 块，至少宽高均为 64 |
| MOS | ↑ | 人工评分均值，默认 1–5 分，附 `MOS_n` |
| Entropy | ↑ | 8 位灰度，256 个灰度等级，Shannon 熵，`log2`，单位 bit，范围 0–8 |

LV、CPBD、Entropy 的灰度转换统一使用 `cv2.COLOR_RGB2GRAY`。
CPBD 保留 Python 参考实现的边缘算法与完整块规则，底部/右侧不足 64 的块不参与统计；
无可用边缘的有效尺寸图像按参考实现返回 0。此实现与专有 MATLAB 边缘检测的数值不保证完全相同。

每项总体均值都是有效图像得分的算术平均；标准差使用总体标准差（`ddof=0`）。
无法计算的指标留空并记录原因，不填零；读取/推理失败的图像会记录为 `error`，不计入任何指标均值。
其他指标成功但个别指标缺失时标为 `partial`，可用指标仍参与各自均值。比较方法时务必核对有效数，必要时在共同有效图像上比较。
返回码：`0` 表示请求的计算全部成功，`2` 表示存在图像失败或请求的客观指标缺失，`1` 表示配置/依赖/模型初始化失败。
MOS 未评分、主动不选择的客观指标，以及 `restore_only` 的空指标不导致返回码 2。

LV 和 Entropy 可能随噪声/锐化增加；这些分数不能单独证明真实细节恢复。建议同步展示相同位置的局部放大图。

## 8. 自检

```bash
python test/real_test/check_nr_results.py
python test/real_test/check_real_nr.py
```

第二个检查需要 NumPy、PyTorch、OpenCV、SciPy、scikit-image、Pillow，但不需要下载 PyIQA 权重。
自检包含 TorchScript 恒等模型的恢复/分块/导出、批量图块与串行输出一致性、边界图块、耗时记录和 MOS 汇总。
CPBD 另与 Python 参考移植在清晰图、模糊图、噪声、常量图和灰度斜坡上核对，差异小于 `1e-14`。
当前 `realblur-test` 环境已安装 PyIQA，已有真实结果记录；本次速度优化未运行完整 GPU 评测。
也可对已有恢复图离线检查三项传统指标：

```bash
python test/real_test/evaluate_real_nr.py --mode evaluate \
  --input-dir /已有恢复图目录 \
  --output-dir results/real_nr/classical_check \
  --metrics LV CPBD Entropy --device cpu --metric-device cpu --max-images 3
```

## 9. 实现来源

- PyIQA 官方项目与接口：https://github.com/chaofengc/IQA-PyTorch
- 固定版本：https://github.com/chaofengc/IQA-PyTorch/tree/v0.1.13
- CPBD 原论文：Narvekar & Karam, IEEE TIP, 2011，DOI: `10.1109/TIP.2011.2131660`
- CPBD Python 参考移植：https://github.com/0x64746b/python-cpbd

CPBD 源代码授权和论文引用要求保留在 `nr_metrics.py` 文件头；发表使用该实现的结果时，应按其要求致谢和引用。
