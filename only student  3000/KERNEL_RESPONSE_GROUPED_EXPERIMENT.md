# 4组核频响条件门控实验（2026-09-14）

## 改动范围

仅在原 `joint2D` 修改。`joint2D copy` 为用户复制的上一版，不修改其源码、数据或权重。

主要结构变化在 `kernel_response.py::ResponseConditionedLocal`：

- 旧版：每位置4个专家权重，所有通道共用；新版：4组通道，每组每位置4个专家权重。
- 控制输入仍为16通道内容投影与16通道核频响投影的拼接；gate从Conv1×1(32,4)变为Conv1×1(32,16)。
- 16个logit按`[batch, group=4, expert=4, height, width]`排列，只沿expert维softmax。
- 每个专家仍为原来的逐通道深度卷积，不改成4组普通卷积；4个专家dilation=1/2/3/4不变。
- 连续通道按组分配。每个专家仅计算一次，结果按通道组加权；不堆叠4份完整专家输出。
- 8个encoder/decoder阶段最后block的第二次处理插入位置不变；瓶颈不变。保持串联，无新增外部残差。
- DDEM、16×16×64预测、Z0/Z1、Local Branch、CAM、attention和恢复损失不变。
- 参数量从61,026,991增至61,030,159，增加3,168；这不是显存增量估算。

## 初始化与旧版本兼容

先按原顺序构造完整1组控制器，再在保存/恢复RNG的上下文里扩展gate。将原4通道gate的权重和偏置重复4次，因此各组从同一门控出发；原专家、投影、后续阶段和预测头的初始化保持一致。不同通道特征/专家参数使各组梯度可以不同。

模型API、`options.py`及现有测试入口的`kernel_response_groups`默认1；旧权重和旧测试指令仍兼容。新模型测试必须同时指定：

```text
--use_kernel_response --kernel_response_groups 4
```

不自动将旧权重改形状加载进4组模型。不同组数的直接严格加载会失败；续训也核对保存配置的组数。上一版配置缺少组数字段时解释为1组。

启动器 `train_online_2d.py` 默认改为`KERNEL_RESPONSE=1 RESPONSE_GROUPS=4`，因此无其他环境变量覆盖时为完整7520张、500epoch、CPU24worker。仍建议正式运行使用下方显式命令。

## 正式训练命令

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
RESPONSE_GROUPS=4 \
RESPONSE_LOSS_MODE=complex \
RESPONSE_LOSS_WEIGHT=0.01 \
RESPONSE_GATE_LOG_INTERVAL=1 \
RUN_TAG= \
RESUME=0 \
DRY_RUN=0 \
python train_online_2d.py
```

从头训练，不加载上一版权重。其他设置保持：AdamW、lr=3e-4、weight_decay=.02、warmup3后cosine、eta_min=1e-6、clip1、patch256、seed1234；训练DataLoader仍4worker，24worker是模糊生成进程数。每25epoch验证、每50epoch周期保存；每epoch7520/8=940个训练batch。

把`DRY_RUN=0`改成1，只打印命令/检查GT数量，不生成数据或启动训练。启动时明确打印GT数量、epoch、batch、生成worker、分组数、源路径和实验名；数量不符直接报错，不默默继续训练。

## 缓存和结果隔离

新实验名：

`Uformer_B_single_image_online_2d_patch256_full_k12_r50_e500_kr16_c64_all8_serial_g4_v2_complex_w0p01_18eeafbbe4`

位置：`logs/motiondeblur/Online2D/<实验名>/`。缓存另用带`serial_g4_v2`后缀的根目录。旧模型、日志、恢复图、验证集和测试集不覆盖。

`KEEP_CACHE=0`只按原逻辑清理本实验已经消费完的候选池。此次代码修改和测试没有删除用户缓存或结果，也没有启动正式训练。

新增`kernel_response_gate_stats.csv`：每epoch**第一个训练batch**，逐阶段、逐组记录4个专家平均权重、归一化熵、空间标准差和组间偏离。它不是全epoch统计。组间偏离在每个位置与各组均值比较后取绝对值均值，避免只比较全局权重均值而漏掉局部差异。统计只使用detach后的数据，不增加损失、不额外前向、不抽随机样本；以`[batch,group,metric]`返回便于双卡按样本聚合。可设`RESPONSE_GATE_LOG_INTERVAL=0`关闭。

## 模糊生成和选择一致性的依据

仅使用原目录不足以无条件保证一致，还需要相同GT内容/路径、seed、生成配置和软件环境。本次保留这些来源与协议：

- 生成器、轨迹定义、核频响标签生成器均未修改，与用户备份逐字节一致。
- `online_2d.py`和整个`dataset/dataset_motiondeblur.py`未修改，生成调用、50epoch映射、候选选择、增强代码均不变。
- 实际使用原`train_full_7520`源路径，未切换到复制目录。
- 与备份启动器按原项目身份生成的命令比较：除模型组数、只读统计、新实验/缓存目录外，训练参数相同；生成子进程参数除缓存输出位置外相同。
- **全部7520个实际GT路径×500个epoch**已通过元数据选择回归：将缓存路径从旧目录换为新目录，所选variant_index逐项一致。共核对3,760,000次源图/epoch选择，未生成这些全部图像。
- 15类运动新旧Blur/GT逐字节一致；实际GT在cache epoch51生成12候选的PNG及运动/强度/裁剪/子帧字段一致；抽样epoch的配对增强一致。
- 模型新增初始化不推进原随机流，统计也不消费随机数。固定DataLoader和从头训练条件保持。

这里保证的是受控条件下的数据生成/选择一致，不是不同模型训练输出逐位一致，也不是每张历史缓存都重新生成过。更换GT内容、绝对路径、生成依赖或中断续训的随机状态，不能自动延伸这个保证。

## 已执行验证

- 新7项分组测试通过：旧1组精确兼容；4组复制初始化与原RNG一致；softmax维度及通道映射；统计不改变输出/梯度/RNG；全部8处/全部4组梯度；完整源路径500epoch选择；命令及文件一致性。
- 原核频响9项测试通过；生成器原有3项测试通过。
- 生产配置Uformer-B在CPU完成batch1、256×256前向/反向、clip1及AdamW更新，所有梯度/参数有限，预测形状和8处统计符合约定。
- 训练入口help、源文件语法和正式命令dry-run通过。

复核：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_grouped_kernel_response.py -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_kernel_response.py -v
```

当前代理执行环境torch.cuda.is_available()为False，未实测GPU/AMP、双卡通信、batch8显存或速度；这些不能用CPU测试代替。尚无新模型PSNR结果，不能保证达到27 dB。
