#!/usr/bin/env python3
"""The v2 launcher with delayed EMA and an independent experiment directory."""
from train_online_2d import main

if __name__ == '__main__':
    raise SystemExit(main(ema=True))
