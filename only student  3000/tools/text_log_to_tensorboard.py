import argparse
import os
import re


EPOCH_RE = re.compile(
    r"Epoch:\s*(\d+)\s+Time:\s*([0-9.]+)\s+Loss:\s*([0-9.]+)"
    r"\s+LearningRate\s+([0-9.eE+-]+)")
VAL_RE = re.compile(
    r"\[Ep\s+(\d+)\s+it\s+(\d+)\s+PSNR\s+\w+:\s*([0-9.]+).*?"
    r"Best_PSNR_\w+\s+([0-9.]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Uformer text training log to TensorBoard events")
    parser.add_argument("--log", required=True, help="Uformer .txt training log")
    parser.add_argument("--output", required=True, help="TensorBoard event directory")
    parser.add_argument("--iters_per_epoch", type=int, default=55,
                        help="Training iterations per epoch (440 samples / batch 8 = 55)")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SystemExit(
            "TensorBoard is not installed. Run: python -m pip install tensorboard"
        ) from exc

    os.makedirs(args.output, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output)
    epoch_count = 0
    val_count = 0

    with open(args.log, "r", encoding="utf-8") as handle:
        for line in handle:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epoch_time = float(epoch_match.group(2))
                loss = float(epoch_match.group(3))
                learning_rate = float(epoch_match.group(4))
                step = epoch * args.iters_per_epoch
                writer.add_scalar("train/epoch_loss", loss, step)
                writer.add_scalar("train/learning_rate", learning_rate, step)
                writer.add_scalar("train/epoch_time_seconds", epoch_time, step)
                epoch_count += 1
                continue

            val_match = VAL_RE.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                iteration = int(val_match.group(2))
                psnr = float(val_match.group(3))
                best_psnr = float(val_match.group(4))
                step = (epoch - 1) * args.iters_per_epoch + iteration
                writer.add_scalar("validation/psnr_crop", psnr, step)
                writer.add_scalar("validation/best_psnr_crop", best_psnr, step)
                val_count += 1

    writer.flush()
    writer.close()
    print("Converted {} epoch records and {} validation records to {}".format(
        epoch_count, val_count, args.output))


if __name__ == "__main__":
    main()
