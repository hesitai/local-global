#!/usr/bin/env python3
"""Launch a named Online2D architecture ablation with the shared protocol."""

import argparse

from train_online_2d import ABLATION_PRESETS, main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(ABLATION_PRESETS), required=True)
    args = parser.parse_args()
    raise SystemExit(main(ablation=args.model))
