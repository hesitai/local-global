"""模型入口：填写 JSON 中 model.factory / model.kwargs，或修改 build_model。

接口约定：输入和输出均为 RGB Tensor [N, 3, H, W]、范围 [0, 1]。
默认 float32，可通过主程序 --amp 启用混合精度；N 为一次并行处理的图块数。
归一化、额外网络输入等特殊逻辑可写在 predict() 中。
"""
import importlib
from collections.abc import Mapping
from pathlib import Path


def load_checkpoint_strict(model, weights, checkpoint_key="auto"):
    """加载常见 checkpoint 格式；严格匹配权重，避免随机参数参与测试。"""
    import torch

    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    state = checkpoint
    if checkpoint_key != "auto":
        for key in checkpoint_key.split("."):
            state = state[key]
    elif isinstance(checkpoint, Mapping):
        for key in ("params_ema", "params", "state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                state = checkpoint[key]
                print("Checkpoint state dictionary:", key)
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint 中没有有效的 state_dict，请配置 model.checkpoint_key。")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("选中的对象不是纯 Tensor state_dict，请配置 model.checkpoint_key。")
    # DataParallel / DistributedDataParallel checkpoints.
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def build_model(weights, device, model_config):
    """用户模型入口，返回已加载权重且处于 eval 模式的网络。

    方法一：JSON 填 factory="model:Uformer"、kwargs={...训练时的完整结构参数...}。
    方法二：将下面网络创建部分替换为你自己的代码，例如：

        from my_network import MyNetwork
        model = MyNetwork(...)
        load_checkpoint_strict(model, weights, model_config.get("checkpoint_key", "auto"))
        return model.to(device).eval()

    方法三：factory="torchscript"，直接载入导出的 TorchScript 文件。
    """
    import torch

    factory_name = model_config.get("factory", "").strip()
    if not factory_name:
        raise NotImplementedError(
            "请填写 real_nr_config.json 的 model.factory 和 model.kwargs，"
            "或修改 test/real_test/nr_model_adapter.py 中的 build_model()。")
    if not weights or not Path(weights).is_file():
        raise FileNotFoundError("请填写有效的 weights 模型权重路径。")
    if factory_name == "torchscript":
        return torch.jit.load(str(weights), map_location=device).eval()
    if ":" not in factory_name:
        raise ValueError("model.factory 格式应为 模块名:函数或类名，例如 model:Uformer。")
    module_name, attribute = factory_name.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    model = factory(**model_config.get("kwargs", {}))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model.factory 必须返回 torch.nn.Module。")
    load_checkpoint_strict(model, weights, model_config.get("checkpoint_key", "auto"))
    return model.to(device).eval()


def predict(model, image, model_config):
    """用户推理入口；多输出网络用 output_key / output_index 选择恢复图。"""
    import torch

    output = model(image)
    if isinstance(output, Mapping):
        key = model_config.get("output_key")
        if key is None:
            raise ValueError("模型返回字典，请填写 model.output_key 选择恢复图。")
        output = output[key]
    elif isinstance(output, (list, tuple)):
        output = output[model_config.get("output_index", -1)]
    if not torch.is_tensor(output):
        raise TypeError("predict() 必须返回恢复图 Tensor。")
    if output.shape != image.shape:
        raise ValueError("恢复图形状 {} 与输入 {} 不一致。".format(tuple(output.shape), tuple(image.shape)))
    if model_config.get("check_finite", True) and not torch.isfinite(output).all().item():
        raise ValueError("模型输出包含 NaN 或 Inf。")
    return output.float()
