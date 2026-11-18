# KID：比较生成模糊与真实模糊的分布

## MotionBlur随机轨迹核基线

本地库：`/home/xtl/hst/motionblur/motionblur.py`，来源LeviBorodenko/motionblur。
直接使用`realblur-test`环境中的NumPy、Pillow和SciPy，无需另行安装整个仓库。
输入仍为`dataset/kid/gt`355张清晰图；每图调用原始Kernel生成新核并执行原始applyTo。

```bash
conda activate realblur-test
cd '/home/xtl/hst/joint2D/only student  3000'
python test/kid/generate_motionblur.py --kernel-size 100 --intensity 0.2 --seed 123
```

图像保存到`dataset/kid/motionblur`。默认100×100和intensity0.2沿用仓库示例，不是按KID调优所得。
intensity控制轨迹非线性/抖动，核尺寸也会影响模糊强度，两者应在实验前固定。
设置keep_image_dim=True保留图像尺寸，保留原库same卷积零填充和uint8转换，可能出现边缘变暗；没有另加噪声、缩放或JPEG。
CPU生成，不需要网络权重。输出目录非空会拒绝覆盖。
`dataset/kid/motionblur_metadata`保存逐图随机种子、输入/输出哈希、每图归一化核.npy和总体配置。
若原算法生成非有限或未归一化核，程序停止报错，不会静默跳图/重新抽样。

生成后计算KID：

```bash
python test/kid/evaluate_kid.py --method-name motionblur \
  --generated-dir dataset/kid/motionblur \
  --output-dir results/kid/motionblur --device cuda:0
```

已用临时图像验证原尺寸保存、同种子复现，并核对默认配置355个随机种子对应的核均有效；未生成实际355张图。

## pyblur生成基线

本地仓库 `/home/xtl/hst/pyblur` 为1.3.0，要求Python>=3.10。原`realblur-test`是Python3.8，生成阶段另用环境，KID仍在原环境运行。

首次创建生成环境并安装本地源码：

```bash
conda create -n pyblur-gen python=3.11 pip -y
conda activate pyblur-gen
python -m pip install -e "/home/xtl/hst/pyblur[scipy]" "numpy==1.26.4"
```

输入继续使用同一批355张清晰图`dataset/kid/gt`。默认调用原生`randomized_blur`，等概率选择盒式、散焦、高斯、线性运动、预置PSF五类，然后由库的原生随机函数采样参数：

```bash
cd '/home/xtl/hst/joint2D/only student  3000'
python test/kid/generate_pyblur.py
```

图像保存在`dataset/kid/pyblur_randomized`；清单、每图种子、源文件与库代码哈希记录在旁边的`pyblur_randomized_metadata`。
默认CPU/scipy后端，保留库原始卷积与边界规则，原尺寸RGB PNG输出，不添加噪声或JPEG。
脚本直接导入指定仓库`src/pyblur`，以避免误用另一个安装版本；不会改动库本身。
若希望单独研究PSF或线性运动分支，可分别使用`--mode psf`和`--mode motion`，各自输出到`pyblur_psf`、`pyblur_motion`；它们是不同评测配置，应在表中标清，不能混为同一种方法。
默认预置PSF随机编号0–99；线性运动随机核尺寸由库设置为3/5/7/9，而不是BasicSR的10–20。
不同库的默认模糊强度并不相同；所有生成参数在正式比较前固定，不按测试KID挑选配置。
可显式选择`--backend opencv`（需安装OpenCV），但后端可能改变边缘与舍入结果，使用后需固定并记录，不应与scipy结果混用。

生成完成后切回指标环境：

```bash
conda activate realblur-test
python test/kid/evaluate_kid.py --method-name pyblur_randomized \
  --generated-dir dataset/kid/pyblur_randomized \
  --output-dir results/kid/pyblur_randomized --device cuda:0
```

包装脚本已在临时目录验证randomized、psf、motion三个模式、尺寸保留和固定种子复现；未运行实际355张生成。

## BasicSR方案1：运动/高斯随机混合模糊

生成入口：`test/kid/generate_basicsr_deblur.py`。
默认输入清晰图：`dataset/kid/gt`（已准备355张target图），默认输出：`dataset/kid/basicsr_deblur`。
输入目录只放清晰图，每张输出一张原尺寸PNG，文件名保留（非PNG输入改为.png），不添加缩放、噪声、JPEG或恢复网络。
需要本地 `/home/xtl/hst/BasicSR/scripts/data_preparation/prepare_hifacegan_dataset.py`。

```bash
conda activate realblur-test
cd '/home/xtl/hst/joint2D/only student  3000'
python -m pip install "imgaug==0.4.0" "numpy==1.24.2"
python test/kid/generate_basicsr_deblur.py
```

调用原始`deblur`配置的MotionBlur(k=(10,20))及GaussianBlur(sigma=(3,8))子分支；逐图以各0.5概率选择。
使用明确的分支抽样与每图随机种子，保留原始方法的分布设置，不保证与原脚本随机数调用顺序逐像素一致。
直接调用原始子分支还避免旧imgaug OneOf在部分NumPy版本上的别名兼容问题，不修改全局NumPy属性。
固定seed默认为123，可用`--seed`覆盖。分支实际张数不强制各一半；记录于生成汇总中。
输出文件夹拒绝覆盖；生成清单、源文件哈希、每图分支和种子写入旁边的`dataset/kid/basicsr_deblur_metadata/`。
生成过程在CPU上运行，不需要安装完整BasicSR或加载权重。本次在临时目录验证了两个模糊分支、尺寸保留与固定种子的重复生成一致性，未生成实际355张结果。

生成完成后，用现有真实参照计算KID：

```bash
python test/kid/evaluate_kid.py --method-name basicsr_deblur \
  --generated-dir dataset/kid/basicsr_deblur \
  --output-dir results/kid/basicsr_deblur --device cuda:0
```

数据输入组别为：`dataset/kid/gt`是清晰源图；`dataset/kid/ours`是本文合成模糊；
`dataset/kid/basicsr_deblur`是本基线生成模糊；真实参照仍是配置文件中的`real_dir`。

入口为 `evaluate_kid.py`，配置为 `kid_config.json`。不需要恢复模型或清晰真值，输入两组**模糊图像目录**。
它独立于 `evaluate_real_nr.py`；KID 是图像集合间的分布指标，不计算逐张图像的 KID。
本次没有修改原来的恢复评测脚本及配置。

## 填写入口

编辑 `test/kid/kid_config.json` 中留空的两项：

```json
"real_dir": "/真实模糊图目录",
"generated_dir": "/某一种方法生成的模糊图目录"
```

`method_name` 是方法名称，`output_dir` 为本次输出目录。相对路径均以项目根目录为基准。
只有 `--config` 的相对路径以命令执行目录为基准。

默认设置：每组355张时，每次无放回抽取300张，重复100次；随机种子123。
默认 `require_equal_counts=true` 检查两组图像数量相同。KID数学上允许数量不同，确有需要可显式设为false。
若一组少于300张，会报错提示修改 `subset_size`，不会静默改变方法间的评测设置。
同一方法的目录应仅包含待评价的模糊图，不能混入清晰图、恢复图或可视化拼图。

## 运行

使用已经安装 PyIQA 的环境：

```bash
conda activate realblur-test
cd '/home/xtl/hst/joint2D/only student  3000'
python test/kid/evaluate_kid.py
```

比较本文方法与另外两种生成方法时，保持真实目录和其他配置不变，分别运行：

```bash
python test/kid/evaluate_kid.py --method-name ours \
  --generated-dir '/本文方法生成图目录' --output-dir results/kid/ours

python test/kid/evaluate_kid.py --method-name method_A \
  --generated-dir '/方法A生成图目录' --output-dir results/kid/method_A

python test/kid/evaluate_kid.py --method-name method_B \
  --generated-dir '/方法B生成图目录' --output-dir results/kid/method_B
```

也可用 `--real-dir`、`--device cuda:0` 或 `--device cpu` 覆盖配置。
输出目录必须为空或尚不存在，并放在输入目录之外；重复实验请用新的输出目录。
第一次提取需要下载 FID Inception 权重，后续使用本地权重缓存。特征提取使用FP32，不使用AMP。

## 保持比较一致

- 三种生成方法使用相同清晰源图、相同源图顺序与一图一变体数量，建议保留相同相对文件名。
- 真实组与生成组不要求一一配对；真实图可与清晰源图场景不同，但应注意整体场景构成。
- 脚本按相对路径排序。相同真实列表与seed保证三种方法复用相同真实子集；相同生成相对文件名保证对应源图使用相同抽样索引。
- 默认 `center_crop_size=0` 使用整图。若真实整图为4000×2250、生成图为256×256局部块，建议先确定合理的尺度方案；可统一设置 `center_crop_size=256`，两组都在原像素尺度取中心块后送入Inception。裁块只是一个可选方案，不会自动令不同航高/分辨率的场景尺度一致。
- 中心裁块每个源文件只产生一个评价样本，固定策略，不随机挑选“看起来最好”的区域。图像小于裁块尺寸直接报错。
- 不同方法必须采用相同裁块、缩放与特征配置，不要生成多批后挑最低KID报告。
- 只比较本脚本同一协议计算出的数值，其他论文的预处理/特征权重不同，数值未必可直接比较。

## 输出

```text
results/kid/ours/
├── summary.csv               # 一行方法总体结果，便于填表4-2
├── summary.json              # 配置、版本、权重哈希、列表摘要与结果
├── summary.txt               # 可读汇总
├── subsets.csv               # 100次子集的KID
├── real_manifest.csv         # 实际真实图名单、尺寸、文件SHA256
├── generated_manifest.csv    # 实际生成图名单、尺寸、文件SHA256
└── sampling_indices.npz      # 每轮两侧的抽样索引（对应manifest顺序）
```

表4-2每侧样本总数填355；表注写“每次各抽取300张、重复100次，报告均值与子集标准差”。
`KID_mean` 和 `KID_subset_std` 是原始数值；`KID_mean_x1000` 和 `KID_subset_std_x1000` 是乘1000后的值。
若用放大后的数值填表，列名必须标明 `KID (×10³) ↓`。脚本不填造示例分数。

标准差为100次子集得分的样本标准差（ddof=1），描述有限图集的子集抽样波动，**不是置信区间，也不是100次独立采集实验**。
若每次都取两侧全部355张，子集标准差几乎为0，并不代表总体差异没有不确定性。
无偏估计偶尔可能为负值，脚本保留，不截断到0。

## 特征与计算细节

使用 PyIQA 的 FID Inception-v3 2048维pool3特征，不使用torchvision普通分类权重替代。
解码为8位RGB并纠正EXIF方向，随后按PyIQA `clean`模式逐通道浮点PIL bicubic缩放为299×299，裁剪到[0,255]，再归一化为 `(x-128)/128`。
不透明灰度图会转换为RGB；透明、16位/浮点或多帧图像应预先按统一方案转换，脚本遇到这些文件报错，不会静默跳过。
文件解码失败同样中止，防止方法间实际样本数不一致。

核函数为 `(xᵀy/d+1)³`；使用去除组内对角项的无偏MMD²估计，float64计算核矩阵。
真实与生成抽样分别使用固定、独立的随机流，避免同一源图编号的相关选择。
`batch_size` 仅是提取Inception特征时的GPU批量，与 `subset_size` 无关。

`results/kid_feature_cache/` 自动缓存两组特征。缓存键含文件相对名、文件内容SHA256、裁块规则、提取器权重、软件版本、设备与提取批量。
更换方法时真实图内容及配置相同，就会自动复用真实特征；改变图像内容会重新提取。
为核对缓存，启动时仍会读取图像并计算文件哈希；缓存不能确认各方法是否真的使用同一清晰源图，这一点需在生成阶段保证。

实现来源：[KID原论文](https://arxiv.org/abs/1801.01401)、[PyIQA特征与核计算实现](https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.13/pyiqa/archs/fid_arch.py)。

## 自检

```bash
python test/kid/check_kid.py
```

测试涵盖无偏公式、负值保留、抽样复现、官方预处理一致性、缓存复用与输入变更失效、结果文件输出；不下载权重，不占GPU。
真实数据的KID尚未运行；两个输入目录按要求留空。
