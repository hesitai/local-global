#!/usr/bin/env python3
"""Generate blur-only baseline from BasicSR's HiFaceGAN deblur template.

No restoration network, model weights, noise, JPEG, resize or training needed.
Equal-probability branch selection matches the distribution of original OneOf;
separate deterministic per-image seeds are used for reproducibility. We call
the original MotionBlur/GaussianBlur children directly, avoiding legacy OneOf
NumPy alias incompatibilities. Not bit-identical to original RNG scheduling.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', default=str(ROOT / 'dataset/kid/gt'))
    parser.add_argument('--output-dir', default=str(ROOT / 'dataset/kid/basicsr_deblur'))
    parser.add_argument('--basicsr-root', default='/home/xtl/hst/BasicSR')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--expected-count', type=int, default=355)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32 - 1 or args.expected_count < 1:
        parser.error('seed须在[0,2**32-2]范围内，expected-count须为正数。')
    import cv2
    import numpy as np
    try:
        import imgaug
    except ImportError as exc:
        raise RuntimeError('请先在realblur-test中安装：python -m pip install "imgaug==0.4.0"') from exc

    source = Path(args.input_dir).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve()
    script = Path(args.basicsr_root).expanduser().resolve() / 'scripts/data_preparation/prepare_hifacegan_dataset.py'
    if not source.is_dir():
        raise FileNotFoundError('清晰图目录不存在：{}；请用--input-dir指定实际目录。'.format(source))
    if not script.is_file():
        raise FileNotFoundError('BasicSR原始退化脚本不存在：{}；请用--basicsr-root指定仓库根目录。'.format(script))
    if source == destination or source in destination.parents:
        raise ValueError('输出目录必须位于输入目录之外。')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('输出目录非空，拒绝覆盖：{}'.format(destination))
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    files = sorted((p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in suffixes),
                   key=lambda p: p.relative_to(source).as_posix())
    if len(files) != args.expected_count:
        raise ValueError('输入图像{}张，与expected-count={}不一致。'.format(len(files), args.expected_count))
    relative_outputs = [p.relative_to(source).with_suffix('.png') for p in files]
    if len(set(relative_outputs)) != len(files):
        raise ValueError('存在仅扩展名不同的同名源图，PNG输出会重名。')
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('需要8位三通道清晰图：{}'.format(path))

    # Load just the standalone script, without importing/training BasicSR.
    spec = importlib.util.spec_from_file_location('basicsr_hifacegan_degradation_source', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mixture = module.DegradationSimulator().default_deg_templates['deblur']
    children = mixture.get_children_lists()[0]
    if len(children) != 2 or [type(child).__name__ for child in children] != ['MotionBlur', 'GaussianBlur']:
        raise ValueError('下载的BasicSR deblur配置与预期不同，请先核对原始代码。')
    print('Source template:', mixture, flush=True)
    metadata_dir = destination.parent / (destination.name + '_metadata')
    if metadata_dir.exists() and any(metadata_dir.iterdir()):
        raise FileExistsError('元数据目录非空，请使用新output-dir：{}'.format(metadata_dir))
    destination.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    branch_rng = np.random.RandomState(args.seed)
    seed_rng = np.random.RandomState(args.seed + 1)
    manifest = metadata_dir / 'generation_manifest.csv'
    fields = ['source_id', 'output_id', 'branch', 'augmentation_seed', 'width', 'height', 'source_sha256', 'output_sha256']
    counts = {'MotionBlur': 0, 'GaussianBlur': 0}
    with manifest.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (path, relative) in enumerate(zip(files, relative_outputs), 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            branch = int(branch_rng.randint(0, 2))
            seed = int(seed_rng.randint(0, 2**31 - 1))
            augmenter = children[branch].deepcopy()
            augmenter.seed_(seed)
            blurred = augmenter.augment_image(image)
            if blurred.shape != image.shape or blurred.dtype != np.uint8:
                raise ValueError('退化输出尺寸或类型变化：{}'.format(path))
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), blurred):
                raise OSError('保存失败：{}'.format(output))
            branch_name = type(augmenter).__name__
            counts[branch_name] += 1
            writer.writerow({'source_id': path.relative_to(source).as_posix(), 'output_id': relative.as_posix(),
                             'branch': branch_name, 'augmentation_seed': seed,
                             'width': image.shape[1], 'height': image.shape[0],
                             'source_sha256': sha256(path), 'output_sha256': sha256(output)})
            stream.flush()
            print('[{}/{}] {} {}'.format(index, len(files), relative, branch_name), flush=True)
    report = {'method': 'BasicSR HiFaceGAN deblur template (motion/Gaussian mixture)',
              'input_dir': str(source), 'output_dir': str(destination), 'source_script': str(script),
              'source_script_sha256': sha256(script), 'seed': args.seed, 'image_count': len(files),
              'branch_counts': counts, 'branch_probabilities': [0.5, 0.5],
              'template': str(mixture), 'imgaug_version': imgaug.__version__,
              'numpy_version': np.__version__, 'opencv_version': cv2.__version__,
              'scope': 'original blur children only; equal-probability per-image branch; no resize/noise/JPEG',
              'rng_note': 'Distribution matches original OneOf; per-image seeding is independently scheduled.'}
    (metadata_dir / 'generation_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done:', destination, '| counts:', counts, '| metadata:', metadata_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
