# 核频响监督 DDEM 实验（2026-09-10）

> 本文保留上一版 **1组门控** 的历史配置。2026-09-14起，启动器默认启用核频响并使用4组门控；新实验见 `KERNEL_RESPONSE_GROUPED_EXPERIMENT.md`。要复现/续训本文实验，明确设置 `KERNEL_RESPONSE=1 RESPONSE_GROUPS=1`。底层模型与测试入口仍默认1组，旧测试指令保持兼容。

## 已实现配置

- 原始工作目录 `joint2D`；用户的 `joint2D copy` 未修改，作为改动前参考。
- `KERNEL_RESPONSE=1` 启用核频响；历史版本启动器默认关闭，当前启动器默认开启。底层API在功能关闭时保持兼容。
- DDEM只输入Blur；GT用于Charbonnier；生成器退化标签用于辅助监督。没有teacher/KD，没有测试时真实标签输入。
- DDEM原32通道1/4特征 → 两个stride-2的3×3卷积（SiLU）→ 1×1卷积输出64通道；256输入对应64×16×16。
- 8个半平面方向（0°,22.5°,...157.5°）×4个半径（1/64、1/32、1/16、1/8 cycles/input-pixel）。通道按半径、方向、实部/虚部交错排列。
- 四个Encoder和四个Decoder，各阶段最后block的第二次处理接入。瓶颈不变，不新增Attention。
- 串联位置：Z0 → 原Local Branch → 新控制模块 → Z1 → K/V及原局部残差。
- 新模块：4个3×3 depthwise专家，dilation=1/2/3/4，保留原C通道；内容和预测描述分别映射为16个控制通道，再生成每位置4个softmax权重。输出仅为专家加权和，没有新增外部残差或残差缩放。
- 专家中心权重近1、其余权重小随机值，是近恒等初始化，不是残差旁路。8个阶段独立参数，共享DDEM预测。
- 空间条件先投影和插值，再与当前特征同步window shift；几何增强同时变换频率通道和必要的复共轭。
- 原有参数在同seed下保持初始化一致；新增参数初始化使用独立保存/恢复的Torch RNG上下文。
- 总参数：61,026,991；原配置60,921,999；新增104,992。

## 标签与数据一致性

`kernel_response_labels.py`计算线性RGB逆向采样算子的稀疏局部频响，不是sRGB图像的精确全局OTF，也不是直接逆滤波。

每个16×16网格单元在中心2×2真实整数输出像素计算并平均，避免半整数输出再次插值，保持D4增强对称性。CPU包含OpenCV 1/32像素插值表约定，CUDA使用理想双线性约定；当前正式实验只用CPU。正号对应 B(p)=sum_t w_t S(p+d_t)。

标签计算在原运动、子帧、权重、裁剪全部确定之后执行，不消费随机数，也不修改PNG累加器。每个候选保存自己的 `.npz` 标签。

独立缓存名称包含功能版本、监督模式、权重和标签配置哈希；缓存清单增加标签路径与配置哈希。复用时检查请求、源路径顺序、生成器/运动/标签代码哈希与全部标签。缺失或不匹配会拒绝复用，不会自动删除旧数据或补零。

保留原 `source_path` 和 `Online2DPatchTrainDataset._sample_epoch_rows`：仍按seed、绝对源路径、training_epoch选择同一候选。不要把训练GT切换到复制目录后仍声称抽样完全一致。

标签FP16未压缩为32 KiB/候选；7520×12个候选约2.96 GB（十进制），实际使用npz压缩，另有PNG和文件系统开销。KEEP_CACHE=0沿用原行为，每50轮池消费完才删除该实验自己的缓存。

## 正式训练

```bash
cd "/home/xtl/hst/joint2D/only student  3000"
conda activate joint-cu118-test

GPUS="0,1" \
BATCH_SIZE=8 \
EPOCHS=500 \
DATASET_SCOPE=full \
GEN_WORKERS=24 \
BLUR_BACKEND=cpu \
POOL_VARIANTS=12 \
POOL_REFRESH_EPOCHS=50 \
KEEP_CACHE=0 \
KERNEL_RESPONSE=1 \
RESPONSE_GROUPS=1 \
RESPONSE_LOSS_MODE=complex \
RESPONSE_LOSS_WEIGHT=0.01 \
RESUME=0 \
python train_online_2d.py
```

新功能默认配置：500epoch、CPU 24 worker；学习率3e-4、AdamW、weight decay=.02、warmup3、cosine至1e-6、clip1、patch256；原val间隔25、checkpoint间隔50保持不变。完整训练目录实测7520张。

仅打印本历史版本命令、不生成数据、不训练：`KERNEL_RESPONSE=1 RESPONSE_GROUPS=1 DRY_RUN=1 python train_online_2d.py`。

将 `RESPONSE_LOSS_MODE=magnitude` 可做幅值监督对照；仍输出实虚部，但该辅助损失不约束相位。幅值加epsilon后计算，避免零点导数问题。辅助损失为按全部元素平均的SmoothL1，计算一次；按4频率半径记录低/中/高频误差（中频合并两个半径）。

相同配置的续训用 `RESUME=1`；原有rolling checkpoint仍按验证间隔保存。不要把改动前checkpoint直接作为本结构的resume。重新从头做另一次实验可指定 `RUN_TAG=第二个实验标识`（字母数字/下划线/连字符），防止覆盖。

## 输出

默认实验名：

`Uformer_B_single_image_online_2d_patch256_full_k12_r50_e500_kr16_c64_all8_serial_v1_complex_w0p01_18eeafbbe4`

日志位于 `logs/motiondeblur/Online2D/<实验名>/`：

- `kernel_response_config.json`：模型开关、所有训练参数、标签频率及坐标约定；
- `kernel_response_losses.csv`：每epoch恢复损失、辅助损失、各频段实虚部/幅值误差；
- `validation_results.csv`：原来的验证日志；
- `models/`：原保存规则下的best/latest/周期checkpoint。

现有 `test/test_gopro_hide.py` 新增 `--use_kernel_response`，加载本实验权重时需启用；其余推理分块、融合、指标及图片保存逻辑保持。正式测试不要求核频响标签。

## 验证记录与边界

1. 新测试套件9项通过（CPU，joint-cu118-test）：
   - 15类运动：改动前/功能关闭/功能开启，Blur与GT逐字节一致，运动、强度、裁剪、seed、子帧数一致；
   - 12候选池：epoch 1..500选择一致，抽样epoch的配对增强一致；
   - 实际GT、cache epoch51、12候选、2个worker、真实生成环境：新旧PNG及生成字段一致；
   - 缓存匹配复用成功，错误seed请求被拒绝且旧缓存未删除；
   - identity、解析平面波、8种D4增强、标签验证、窗口移动、两种loss；
   - 功能关闭与改动前模型输出逐元素一致，公共参数同seed初始化一致；
   - 8个控制模块和预测头梯度有效、有限，支持矩形输入。
2. 原生成器现有3项online2d单元测试通过。
3. 真实CPU生成环境为OpenCV5.0，32个频点的平面波验证最大误差约4.20e-8。
4. 完整生产Uformer-B在CPU完成batch1、256×256前向/反向、clip1和AdamW更新；输出标签1×64×16×16，所有梯度/更新参数有限。
5. AST核对：运动随机化、prepare_task、build_homographies、choose_valid_crop、render_task、源图排序、候选选择、50epoch映射均未改变；整个轨迹定义文件与用户备份逐字节一致。

复查命令：

```bash
python -m unittest discover -s tests -p test_kernel_response.py -v
```

没有生成全部7520×12×10个候选逐张验证；上述是明确范围的回归检查，配合原随机/生成核心不变的代码核对。没有启动正式500epoch训练。

当前代理执行环境无法访问NVIDIA驱动，torch.cuda.is_available()为False：**尚未实测双卡AMP、batch8 GPU显存和GPU训练速度，也未验证新结构实际涨分。** CPU完整模型测试峰值进程RSS约9.43 GiB，不能用于推算GPU需求。
