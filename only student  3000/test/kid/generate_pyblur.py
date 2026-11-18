#!/usr/bin/env python3
"""Generate pyblur baselines from shared sharp images (Python >=3.10).

Calls the downloaded library directly. Default: its native five-way randomized
blur dispatcher. Optional: only PSF or linear-motion random blur. No resize,
noise, JPEG or restoration. Keep backend and mode fixed across comparisons.
"""
import argparse
import csv
import hashlib
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
    parser.add_argument('--output-dir')
    parser.add_argument('--pyblur-root', default=str(ROOT.parents[1] / 'pyblur'))
    parser.add_argument('--mode', choices=['randomized', 'psf', 'motion'], default='randomized')
    parser.add_argument('--backend', choices=['scipy', 'opencv', 'numpy'], default='scipy')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--expected-count', type=int, default=355)
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error('本地pyblur需要Python>=3.10；请使用pyblur-gen环境，不要在Python3.8的realblur-test中安装。')
    if not 0 <= args.seed < 2**32 or args.expected_count < 1:
        parser.error('seed须在[0,2**32-1]内，expected-count须为正数。')
    import numpy as np
    from PIL import Image, ImageOps

    source = Path(args.input_dir).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / ('dataset/kid/pyblur_' + args.mode)
    repository = Path(args.pyblur_root).expanduser().resolve()
    package = repository / 'src/pyblur'
    if not (package / '__init__.py').is_file():
        raise FileNotFoundError('找不到本地pyblur源码：{}'.format(package))
    sys.path.insert(0, str(package.parent))
    import pyblur
    if Path(pyblur.__file__).resolve() != (package / '__init__.py').resolve():
        raise RuntimeError('实际导入的不是指定pyblur源码，请检查环境。')
    pyblur.get_backend(args.backend)  # fail before output writes if backend unavailable
    if not source.is_dir():
        raise FileNotFoundError('清晰图目录不存在：{}'.format(source))
    if source == destination or source in destination.parents:
        raise ValueError('输出目录必须放在输入目录之外。')
    metadata_dir = destination.parent / (destination.name + '_metadata')
    for directory in (destination, metadata_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('目录非空，拒绝覆盖：{}'.format(directory))
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    files = sorted((p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in suffixes),
                   key=lambda p: p.relative_to(source).as_posix())
    if len(files) != args.expected_count:
        raise ValueError('找到{}张清晰图，expected-count={}。'.format(len(files), args.expected_count))
    names = [p.relative_to(source).with_suffix('.png') for p in files]
    if len(set(names)) != len(files):
        raise ValueError('源图文件名转换为PNG后存在冲突。')

    def read_rgb(path):
        with Image.open(path) as image:
            if image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('输入须为单帧8位RGB清晰图：{}'.format(path))
            image.load()
            return ImageOps.exif_transpose(image).copy()

    for path in files:
        read_rgb(path)
    function = {'randomized': pyblur.randomized_blur, 'psf': pyblur.psf_blur_random,
                'motion': pyblur.linear_motion_blur_random}[args.mode]
    source_hashes = {p.relative_to(repository).as_posix(): digest(p)
                     for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in ('.py', '.npz')}
    seeds = np.random.RandomState(args.seed).randint(0, 2**31 - 1, size=len(files))
    destination.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    fields = ['source_id', 'output_id', 'mode', 'backend', 'image_seed', 'width', 'height',
              'source_sha256', 'output_sha256']
    with (metadata_dir / 'generation_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (path, relative, seed) in enumerate(zip(files, names, seeds), 1):
            image = read_rgb(path)
            np.random.seed(int(seed))
            blurred = function(image, backend=args.backend)
            if blurred.mode != 'RGB' or blurred.size != image.size:
                raise ValueError('pyblur改变了输出尺寸或颜色模式：{}'.format(path))
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            blurred.save(output, format='PNG')
            writer.writerow({'source_id': path.relative_to(source).as_posix(), 'output_id': relative.as_posix(),
                             'mode': args.mode, 'backend': args.backend, 'image_seed': int(seed),
                             'width': image.width, 'height': image.height,
                             'source_sha256': digest(path), 'output_sha256': digest(output)})
            stream.flush()
            print('[{}/{}] {} {}'.format(index, len(files), relative, args.mode), flush=True)
    versions = {'python': sys.version.split()[0]}
    for name in ('numpy', 'Pillow', 'scipy', 'opencv-python', 'pyblur'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    report = {'method': 'pyblur ' + args.mode, 'function': function.__name__,
              'image_count': len(files), 'seed': args.seed, 'input_dir': str(source),
              'output_dir': str(destination), 'repository': str(repository), 'backend': args.backend,
              'source_files_sha256': source_hashes, 'versions': versions,
              'repository_project_metadata_sha256': digest(repository / 'pyproject.toml'),
              'scope': 'native random APIs; per-image seeding; RGB PNG; no resize/noise/JPEG',
              'boundary': 'native library backend behavior retained; scipy convolution uses white padding'}
    (metadata_dir / 'generation_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done:', destination, '| metadata:', metadata_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
