# MATLAB → Python 文件对应表

| MATLAB 源文件 | Python 文件 | 说明 |
|---|---|---|
| `script_data_gen.m` | `script_data_gen.py` | 原始批量数据生成 |
| `demo_flow_global_blur_no_mex.m` | `demo_flow_global_blur_no_mex.py` | 单图 flow 曝光积分演示 |
| `script_uav_global_blur_repro_smooth_flow.m` | `script_uav_global_blur_repro_smooth_flow.py` | 单文件保留全部 UAV 空间场函数 |
| `script_uav_global_blur_multisubframe_no_depth.m` | `script_uav_global_blur_multisubframe_no_depth.py` | 单文件保留轨迹、warp、summary、裁边和保存 |
| `script_uav_object_motion_blur_framework_A.m` | `script_uav_object_motion_blur_framework_A.py` | 单文件保留 mask、补全、目标轨迹与合成 |
| `make_simple_object_test_data.m` | `make_simple_object_test_data.py` | 自动生成方案 A 测试图和 mask |
| `view_uav_global_blur_flow_by_path.m` | `view_uav_global_blur_flow_by_path.py` | 空间 flow 单样本查看 |
| `view_uav_multisubframe_flow_by_path.m` | `view_uav_multisubframe_flow_by_path.py` | 多子帧单样本查看 |
| `view_uav_multisubframe_all_cases_global_value_consistent_v4.m` | `view_uav_multisubframe_all_cases_global_value_consistent_v4.py` | 五类统一色标汇总 |
| `gen_mf.m` | `gen_mf.py` | 逐式转换 |
| `gen_mf_para.m` | `gen_mf_para.py` | 增加可选可复现 RNG |
| `gen_blurry_img.m` | `gen_blurry_img.py` | 调用 Python 空间变化卷积 |
| `utils/mfuv2mo.m` | `utils/mfuv2mo.py` | 逐式转换 |
| `utils/flowToColor.m`、`computeColor.m` | `utils/flow_to_color.py` | 合并为一个 Middlebury 色轮模块 |
| `utils/draw_mf_on_img.m` | `utils/draw_mf_on_img.py` | 稀疏运动线叠加显示 |
| `utils/plotflow.m` | `utils/plotflow.py` | 保留 vector/rgb/hsv/bw/mag 显示模式 |
| `motionBlurConv.cc/.mexw64` | `conv_opt/motion_blur_conv.py::motion_blur_conv` | Numba 重写，不再需要 MEX |
| `motionBlurConv_mirror.cc/.mexw64` | `conv_opt/motion_blur_conv.py::motion_blur_conv_mirror` | 修复原固定 25×25 缓冲区风险 |
| `makemexfiles.m` | 不需要 | Python 版本不编译 MATLAB MEX |

## 结构约定

三个 `script_uav_*.py` 没有把核心逻辑抽进包中。打开任意脚本即可从顶部配置一直看到下方本地函数，查找方式与对应 MATLAB 文件一致。
