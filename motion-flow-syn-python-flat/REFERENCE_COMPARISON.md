# MATLAB 参考结果对照报告

参考输入和输出来自用户上传的 `motion-flow-syn-master.zip`：

```text
dataset/image/train/8143.jpg
dataset/data_syn_uav_global_repro_smooth_flow/train
dataset/data_syn_uav_global_multisubframe_v2/train
```

输入图尺寸为 `321×481`。

## 1. 五类 smooth-flow 运动场

直接比较 Python 生成的未裁剪 `mf.mu/mf.mv` 与 MATLAB `*_mf_para.mat`：

| 类型 | mu 最大绝对误差 | mv 最大绝对误差 |
|---|---:|---:|
| 01 linear uniform | 0 | 0 |
| 02 linear variable speed | 0 | 0 |
| 03 hover jitter | 1.78e-15 | 1.78e-15 |
| 04 maneuver field | 0 | 0 |
| 05 high-frequency vibration | 1.78e-15 | 1.78e-15 |

第 3、5 类的 MATLAB Twister 随机相位也与 Python 完全一致。

## 2. 五类 smooth-flow 最终 PNG

| 类型 | 输出尺寸一致 | 像素完全相同比例 | 最大灰度差 |
|---|---|---:|---:|
| 01 | 是 | 100% | 0 |
| 02 | 是 | 100% | 0 |
| 03 | 是 | 100% | 0 |
| 04 | 是 | 99.9972% | 4/255 |
| 05 | 是 | 100% | 0 |

第 4 类只有极少数像素受接近特殊核角度的浮点舍入影响。

## 3. 五类多子帧轨迹统计

起止 `mu/mv` 最大误差不超过 `5.69e-14`，裁剪后的 `path_len` 最大误差不超过 `1.96e-13`。输出裁边尺寸全部与 MATLAB 一致。

## 4. 五类多子帧最终 PNG

| 类型 | 平均灰度差 | 最大灰度差 | 完全相同比例 |
|---|---:|---:|---:|
| 01 | 0.0554/255 | 1/255 | 94.46% |
| 02 | 0.0566/255 | 1/255 | 94.34% |
| 03 | 0.0576/255 | 1/255 | 94.24% |
| 04 | 0.0626/255 | 1/255 | 93.74% |
| 05 | 0.0778/255 | 1/255 | 92.22% |

差异只在相邻灰度级之间，原因是 MATLAB `interp2` 与 OpenCV `remap` 的双线性插值和内部取整实现不同。

## 5. 其他端到端检查

- 原 `script_data_gen`：已用 1 个随机模糊样本 + 1 个 sharp 样本运行；
- 目标方案 A：自动测试数据上的五类目标运动均成功生成；
- 三个查看脚本：均能读取 Python 输出的 PNG/MAT 并生成可视化；
- 自动测试：6 项全部通过。

