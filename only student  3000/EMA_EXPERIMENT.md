# v2 + EMA

使用原训练环境，在本目录启动：

```bash
python train_online_2d_ema.py
# 查看命令，不生成数据、不训练
DRY_RUN=1 python train_online_2d_ema.py
# 从此 EMA 实验的 model_latest.pth 恢复
RESUME=1 python train_online_2d_ema.py
```

保留 v2 启动器所有原参数和环境变量。实验目录在原名称后增加
`_ema_e4_d0p999`，原 `train_online_2d.py` 仍运行无 EMA 的 v2。
模糊生成、候选池路径、候选抽样、模型结构、损失和学习率策略不变。
缓存路径沿用 v2；不要让共享同一缓存的两个训练进程同时生成或清理缓存。
需要并行实验时用不同 RUN_TAG（这是原启动器已有功能）。

第 1–3 epoch 不启用 EMA；第 4 epoch 首个 batch 前复制普通模型，
此后每次成功的优化器更新执行 `ema = 0.999 * ema + 0.001 * model`。
AMP 溢出跳过优化器时也跳过 EMA。浮点 buffer 同步做 EMA，整数 buffer 直接复制。
EMA 模型始终 eval 且不参与梯度计算。

验证频率保持每 25 epoch；同一固定验证集分别验证普通模型和已激活的 EMA。
`validation_results.csv` 中 `periodic` 为普通模型，`periodic_ema` 为 EMA；
两者独立维护 best。额外 EMA 验证保存并恢复随机数状态，避免改变后续抽样。

- `models/model_best.pth`：普通模型最佳权重。
- `models/model_best_ema.pth`：EMA 最佳权重，标准 state_dict，可用原推理加载器。
- `models/model_latest.pth`：每 epoch 保存完整续训状态，包括未启用 EMA 的预热阶段。
- `models/model_epoch_N.pth`：保留原周期备份频率，同时包含完整续训状态。

续训请使用 latest 或 epoch checkpoint，不能用仅含推理权重的 EMA best。
EMA checkpoint 包含普通模型、优化器、EMA 权重及更新次数、两套最佳分数、
AMP scaler、warmup/cosine 调度器和 Python/NumPy/PyTorch/CUDA 随机数状态。
原始非 EMA checkpoint 不作为 EMA 续训 checkpoint，防止静默重置 EMA。
若使用 RUN_TAG 等环境变量，续训时保持一致。中途退出从最后一个完整 epoch 恢复。

验证命令：

```bash
python -m unittest discover -s tests -p test_ema.py -v
python -m unittest discover -s tests -p 'test*kernel_response.py' -v
```
