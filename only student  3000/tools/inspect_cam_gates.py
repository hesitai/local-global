import argparse
import os
import sys
from collections import OrderedDict

import torch
from PIL import Image
from torchvision.transforms import functional as TF


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import ChannelAwareModule, Uformer  # noqa: E402
from utils.model_utils import load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Uformer CAM gate distributions")
    parser.add_argument("--input_dir", required=True, help="GoPro test root or its input folder")
    parser.add_argument("--weights", required=True, help="Model checkpoint")
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=256)
    return parser.parse_args()


def resolve_input_dir(path):
    candidate = os.path.join(path, "input")
    return candidate if os.path.isdir(candidate) else path


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    input_dir = resolve_input_dir(args.input_dir)
    names = sorted(
        name for name in os.listdir(input_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg")))[:args.num_images]
    if not names:
        raise RuntimeError("No test images found in {}".format(input_dir))

    model = Uformer(
        img_size=args.patch_size,
        embed_dim=32,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        win_size=8,
        token_projection="linear",
        token_mlp="leff",
        modulator=True,
        dd_in=3,
        use_local_branch=True,
        use_channel_aware=True,
        double_process=True,
        local_expansion=2,
    )
    load_checkpoint(model, args.weights)
    model.cuda().eval()

    stats = OrderedDict()
    handles = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            gate = output.detach().float()
            item = stats.setdefault(name, {
                "sum": 0.0, "sum_sq": 0.0, "count": 0,
                "min": float("inf"), "max": float("-inf"),
                "lt025": 0, "lt050": 0, "gt075": 0,
            })
            item["sum"] += gate.sum().item()
            item["sum_sq"] += gate.square().sum().item()
            item["count"] += gate.numel()
            item["min"] = min(item["min"], gate.min().item())
            item["max"] = max(item["max"], gate.max().item())
            item["lt025"] += (gate < 0.25).sum().item()
            item["lt050"] += (gate < 0.50).sum().item()
            item["gt075"] += (gate > 0.75).sum().item()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, ChannelAwareModule):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for index, name in enumerate(names, 1):
            image = Image.open(os.path.join(input_dir, name)).convert("RGB")
            image = TF.center_crop(image, [args.patch_size, args.patch_size])
            tensor = TF.to_tensor(image).unsqueeze(0).cuda()
            with torch.cuda.amp.autocast():
                model(tensor)
            print("Processed {}/{}: {}".format(index, len(names), name))

    for handle in handles:
        handle.remove()

    print("\nlayer\tmean\tstd\tmin\tmax\t<0.25\t<0.50\t>0.75")
    for name, item in stats.items():
        count = item["count"]
        mean = item["sum"] / count
        variance = max(item["sum_sq"] / count - mean * mean, 0.0)
        print(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.1%}\t{:.1%}\t{:.1%}".format(
                name, mean, variance ** 0.5, item["min"], item["max"],
                item["lt025"] / count, item["lt050"] / count,
                item["gt075"] / count,
            )
        )


if __name__ == "__main__":
    main()
