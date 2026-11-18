#!/usr/bin/env python3
"""Generate the local motionblur trajectory-kernel baseline, one kernel per image.

Defaults follow the repository example (100x100, intensity0.2), not parameters
selected by KID. Native applyTo(..., keep_image_dim=True) retains zero padding
and original rounding. Does not train a network or change library source.
"""
import argparse
import csv
import hashlib
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default=str(ROOT / 'dataset/kid/gt'))
    parser.add_argument('--output-dir', default=str(ROOT / 'dataset/kid/motionblur'))
    parser.add_argument('--motionblur-root', default=str(ROOT.parents[1] / 'motionblur'))
    parser.add_argument('--kernel-size', type=int, default=100)
    parser.add_argument('--intensity', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--expected-count', type=int, default=355)
    args = parser.parse_args(argv)
    if args.kernel_size < 3 or not 0 <= args.intensity <= 1:
        parser.error('kernel-size须>=3，intensity须在[0,1]范围内。')
    if not 0 <= args.seed < 2**32 or args.expected_count < 1:
        parser.error('seed须在[0,2**32-1]范围内，expected-count须为正数。')
    import numpy as np
    from PIL import Image, ImageOps

    source = Path(args.input_dir).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve()
    library = Path(args.motionblur_root).expanduser().resolve() / 'motionblur.py'
    if not source.is_dir():
        raise FileNotFoundError('清晰图目录不存在：{}'.format(source))
    if not library.is_file():
        raise FileNotFoundError('motionblur源码不存在：{}'.format(library))
    if source == destination or source in destination.parents:
        raise ValueError('输出目录必须在输入目录之外。')
    metadata_dir = destination.parent / (destination.name + '_metadata')
    for directory in (destination, metadata_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('目录非空，拒绝覆盖：{}'.format(directory))
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    files = sorted((p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in suffixes),
                   key=lambda p: p.relative_to(source).as_posix())
    if len(files) != args.expected_count:
        raise ValueError('输入{}张，与expected-count={}不一致。'.format(len(files), args.expected_count))
    outputs = [p.relative_to(source).with_suffix('.png') for p in files]
    if len(set(outputs)) != len(outputs):
        raise ValueError('转换为PNG后存在同名输出。')

    def load_image(path):
        with Image.open(path) as image:
            if image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('输入须为单帧8位RGB图像：{}'.format(path))
            image.load()
            result = ImageOps.exif_transpose(image).copy()
            if min(result.size) < args.kernel_size:
                raise ValueError('图像小于模糊核：{}'.format(path))
            return result

    for path in files:
        load_image(path)
    spec = importlib.util.spec_from_file_location('local_motionblur_generator', library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination.mkdir(parents=True, exist_ok=True)
    (metadata_dir / 'kernels').mkdir(parents=True, exist_ok=True)
    seeds = np.random.RandomState(args.seed).randint(0, 2**31 - 1, size=len(files))
    fields = ['source_id', 'output_id', 'image_seed', 'kernel_size', 'intensity', 'width', 'height',
              'source_sha256', 'output_sha256', 'kernel_file', 'kernel_sha256']
    with (metadata_dir / 'generation_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (path, relative, seed) in enumerate(zip(files, outputs, seeds), 1):
            image = load_image(path)
            np.random.seed(int(seed))
            kernel = module.Kernel(size=(args.kernel_size, args.kernel_size), intensity=args.intensity)
            matrix = kernel.kernelMatrix
            if not np.isfinite(matrix).all() or np.any(matrix < 0) or not np.isclose(matrix.sum(), 1):
                raise ValueError('原始算法生成了无效核：{} seed={}，停止而不静默重采样。'.format(path, seed))
            blurred = kernel.applyTo(image, keep_image_dim=True)
            if blurred.size != image.size or blurred.mode != 'RGB':
                raise ValueError('输出图像尺寸或模式变化：{}'.format(path))
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            blurred.save(output, format='PNG')
            kernel_path = metadata_dir / 'kernels' / ('{:06d}.npy'.format(index - 1))
            np.save(kernel_path, matrix, allow_pickle=False)
            writer.writerow({'source_id': path.relative_to(source).as_posix(), 'output_id': relative.as_posix(),
                             'image_seed': int(seed), 'kernel_size': args.kernel_size, 'intensity': args.intensity,
                             'width': image.width, 'height': image.height, 'source_sha256': digest(path),
                             'output_sha256': digest(output), 'kernel_file': kernel_path.relative_to(metadata_dir).as_posix(),
                             'kernel_sha256': digest(kernel_path)})
            stream.flush()
            print('[{}/{}] {}'.format(index, len(files), relative), flush=True)
    versions = {'python': sys.version.split()[0]}
    for package in ('numpy', 'scipy', 'Pillow'):
        versions[package] = metadata.version(package)
    report = {'method': 'LeviBorodenko/motionblur random trajectory PSF', 'input_dir': str(source),
              'output_dir': str(destination), 'image_count': len(files), 'kernel_size': args.kernel_size,
              'intensity': args.intensity, 'seed': args.seed, 'keep_image_dim': True,
              'source_library': str(library), 'source_library_sha256': digest(library), 'versions': versions,
              'scope': 'native kernel generation and applyTo; original zero padding and uint8 conversion',
              'sampling': 'one new kernel per image; fixed intensity and support size; independent recorded image seeds'}
    (metadata_dir / 'generation_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done:', destination, '| metadata:', metadata_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
