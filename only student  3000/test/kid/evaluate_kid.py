#!/usr/bin/env python3
"""Compare two image distributions, not image pairs: python test/kid/evaluate_kid.py.

Inception features: PyIQA 0.1.13 FID Inception-v3, pool3 (2048 dims).
Preprocessing: float-channel PIL bicubic 299x299, clip0..255, (x-128)/128,
matching PyIQA clean mode. Optional fixed center crop before resizing.
Estimator: unbiased polynomial MMD^2, k(x,y)=(x@y/d+1)^3.
Sources: https://arxiv.org/abs/1801.01401 and
https://github.com/chaofengc/IQA-PyTorch/blob/v0.1.13/pyiqa/archs/fid_arch.py
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_image(path, crop_size):
    with Image.open(path) as source:
        if source.mode not in ('1', 'L', 'P', 'RGB') or source.info.get('transparency') is not None:
            raise ValueError('只接受不透明的8位图像，请先统一转换：{} ({})'.format(path, source.mode))
        if getattr(source, 'n_frames', 1) != 1:
            raise ValueError('不支持多帧图像：{}'.format(path))
        image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = image.size
        if crop_size:
            if min(width, height) < crop_size:
                raise ValueError('图像小于 center_crop_size，不能静默放大：{}'.format(path))
            left, top = (width - crop_size) // 2, (height - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
        return np.asarray(image).copy(), (width, height)


def image_manifest(directory, crop_size):
    paths = sorted((p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES),
                   key=lambda p: p.relative_to(directory).as_posix())
    if len(paths) < 2:
        raise ValueError('每组至少需要2张图像：{}'.format(directory))
    records = []
    for path in paths:
        _, size = read_image(path, crop_size)  # fail before any feature extraction on corrupt images
        records.append({'image_id': path.relative_to(directory).as_posix(),
                        'path': str(path), 'sha256': sha256(path), 'width': size[0], 'height': size[1]})
    return paths, records


class ImageDataset:
    def __init__(self, paths, crop_size):
        self.paths, self.crop_size = paths, crop_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import torch
        rgb, _ = read_image(self.paths[index], self.crop_size)
        channels = []
        for channel in range(3):
            plane = Image.fromarray(rgb[:, :, channel].astype(np.float32), mode='F')
            plane = plane.resize((299, 299), resample=Image.BICUBIC)
            channels.append(np.asarray(plane).clip(0, 255))
        image = (np.stack(channels, axis=0).astype(np.float32) - 128) / 128
        return torch.from_numpy(image.copy())


def build_extractor(device):
    from pyiqa.archs.inception import InceptionV3
    return InceptionV3(output_blocks=[3]).to(device).eval()


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def extract_features(paths, records, model, signature, config):
    import torch
    from torch.utils.data import DataLoader
    identity = {'images': [(r['image_id'], r['sha256']) for r in records],
                'extractor': signature, 'crop_size': config['center_crop_size']}
    encoded = json.dumps(identity, sort_keys=True).encode()
    cache_path = Path(config['cache_dir']) / (hashlib.sha256(encoded).hexdigest() + '.npz')
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            features = cache['features']
        if features.shape == (len(paths), 2048) and np.isfinite(features).all():
            print('Reuse feature cache:', cache_path.name, flush=True)
            return features, str(cache_path)
        raise ValueError('特征缓存无效，请检查：{}'.format(cache_path))
    loader = DataLoader(ImageDataset(paths, config['center_crop_size']),
                        batch_size=config['batch_size'], num_workers=config['num_workers'],
                        shuffle=False, drop_last=False, pin_memory=config['device'].startswith('cuda'))
    parts = []
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            features = model(batch.to(config['device']), False, False)[0].flatten(1)
            array = features.cpu().numpy()
            if array.shape[1] != 2048 or not np.isfinite(array).all():
                raise ValueError('Inception 特征非有限或维度不为2048。')
            parts.append(array)
            print('Features {}/{} batches'.format(index, len(loader)), flush=True)
    features = np.concatenate(parts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, features=features)
    temporary.replace(cache_path)
    return features, str(cache_path)


def kid_subsets(real, generated, subset_size, num_subsets, seed):
    """Return individual unbiased KID estimates and reproducible sampling indices.

    Independent random streams for real and synthetic subsets avoid coupling
    same-index content; identical real lists/seeds yield shared real subsets
    across methods. Subset SD is descriptive, not a confidence interval.
    """
    real, generated = np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1] or real.shape[1] < 1:
        raise ValueError('两组特征须为相同维度的二维数组。')
    if not np.isfinite(real).all() or not np.isfinite(generated).all():
        raise ValueError('特征包含NaN/Inf。')
    if not 2 <= subset_size <= min(len(real), len(generated)) or num_subsets < 2:
        raise ValueError('需要2 <= subset_size <= 两组最小图像数，num_subsets >=2。')
    rng_real, rng_gen = np.random.RandomState(seed), np.random.RandomState(seed + 1)
    real_indices = np.stack([rng_real.choice(len(real), subset_size, replace=False) for _ in range(num_subsets)])
    gen_indices = np.stack([rng_gen.choice(len(generated), subset_size, replace=False) for _ in range(num_subsets)])
    dimension = real.shape[1]
    def kernel(x, y):
        return (x @ y.T / dimension + 1) ** 3
    # Small sets: compute the Gram matrices once. Large sets: bounded subset memory.
    full = max(len(real), len(generated)) <= 2000
    if full:
        rr, gg, rg = kernel(real, real), kernel(generated, generated), kernel(real, generated)
    scores = []
    for ri, gi in zip(real_indices, gen_indices):
        if full:
            a, b, c = rr[np.ix_(ri, ri)], gg[np.ix_(gi, gi)], rg[np.ix_(ri, gi)]
        else:
            x, y = real[ri], generated[gi]
            a, b, c = kernel(x, x), kernel(y, y), kernel(x, y)
        m = subset_size
        scores.append((a.sum() - np.trace(a) + b.sum() - np.trace(b)) / (m * (m - 1))
                      - 2 * c.mean())
    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError('KID出现非有限结果。')
    return scores, real_indices, gen_indices


def write_csv(path, records, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(config):
    import torch
    real_dir, gen_dir = Path(config['real_dir']), Path(config['generated_dir'])
    output = Path(config['output_dir'])
    if real_dir == gen_dir:
        raise ValueError('真实与生成目录相同，请检查路径。')
    for directory in (real_dir, gen_dir):
        if not directory.is_dir():
            raise NotADirectoryError(str(directory))
        for artifact_dir in (output, Path(config['cache_dir'])):
            if directory == artifact_dir or directory in artifact_dir.parents:
                raise ValueError('输出/缓存目录须放在输入目录之外。')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('输出目录非空，请换新目录：{}'.format(output))
    real_paths, real_records = image_manifest(real_dir, config['center_crop_size'])
    gen_paths, gen_records = image_manifest(gen_dir, config['center_crop_size'])
    n, m = len(real_paths), len(gen_paths)
    if config['require_equal_counts'] and n != m:
        raise ValueError('两组数量不一致：real={} generated={}。请检查缺图或显式关闭require_equal_counts。'.format(n, m))
    if not 2 <= config['subset_size'] <= min(n, m):
        raise ValueError('subset_size={} 超出有效范围[2,{}]；请显式修改，脚本不会自动改变抽样设置。'.format(config['subset_size'], min(n, m)))
    if config['subset_size'] == n == m:
        print('注意：每次子集均为全体样本，重复抽样标准差接近0，不代表总体不确定性为0。')
    for label, records in [('real', real_records), ('generated', gen_records)]:
        duplicates = len(records) - len({r['sha256'] for r in records})
        if duplicates:
            print('注意：{} 发现{}个相同文件内容的重复项。'.format(label, duplicates))
    device = config['device']
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用；可设device=cpu。')
    config = dict(config, device=str(torch.device(device)))
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    versions = {name: metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'pyiqa')}
    model = build_extractor(config['device'])
    signature = {'versions': versions, 'weights_sha256': model_digest(model),
                 'preprocess': 'RGB+EXIF; float-channel PIL bicubic299; clip; (x-128)/128',
                 'feature': 'PyIQA FID Inception-v3 pool3 2048', 'script_sha256': sha256(__file__),
                 'device': config['device'], 'batch_size': config['batch_size']}
    print('Method={} real={} generated={} subset={} repeats={}'.format(
        config['method_name'], n, m, config['subset_size'], config['num_subsets']), flush=True)
    real_features, real_cache = extract_features(real_paths, real_records, model, signature, config)
    gen_features, gen_cache = extract_features(gen_paths, gen_records, model, signature, config)
    scores, real_indices, gen_indices = kid_subsets(real_features, gen_features,
        config['subset_size'], config['num_subsets'], config['seed'])
    summary = {'method': config['method_name'], 'real_count': n, 'generated_count': m,
               'subset_size': config['subset_size'], 'num_subsets': config['num_subsets'],
               'KID_mean': float(scores.mean()), 'KID_subset_std': float(scores.std(ddof=1)),
               'KID_mean_x1000': float(scores.mean() * 1000),
               'KID_subset_std_x1000': float(scores.std(ddof=1) * 1000)}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'summary.csv', [summary], list(summary))
    write_csv(output / 'subsets.csv', [{'subset': i + 1, 'KID': float(s), 'KID_x1000': float(s * 1000)}
              for i, s in enumerate(scores)], ['subset', 'KID', 'KID_x1000'])
    for name, records in [('real', real_records), ('generated', gen_records)]:
        write_csv(output / (name + '_manifest.csv'), records, list(records[0]))
    np.savez_compressed(output / 'sampling_indices.npz', real=real_indices, generated=gen_indices)
    report = {'summary': summary, 'config': config, 'extractor': signature,
              'real_features_cache': real_cache, 'generated_features_cache': gen_cache,
              'real_manifest_digest': hashlib.sha256(json.dumps(real_records, sort_keys=True).encode()).hexdigest(),
              'generated_image_ids_digest': hashlib.sha256(json.dumps([r['image_id'] for r in gen_records]).encode()).hexdigest(),
              'timestamp': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'std_definition': 'sample SD (ddof=1) over overlapping random subsets; not a CI',
              'kernel': '(x@y/d+1)^3; unbiased within-set diagonal-excluded MMD^2; no negative clipping'}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    message = '{}: KID = {:.8f} +/- {:.8f}\nKID x1000 = {:.4f} +/- {:.4f}\nreal={} generated={} subset={} repeats={}\nSD is subset sampling variation, NOT a confidence interval.\n'.format(
        config['method_name'], summary['KID_mean'], summary['KID_subset_std'],
        summary['KID_mean_x1000'], summary['KID_subset_std_x1000'], n, m,
        config['subset_size'], config['num_subsets'])
    (output / 'summary.txt').write_text(message, encoding='utf-8')
    print(message)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('kid_config.json')))
    for name in ('real-dir', 'generated-dir', 'output-dir', 'method-name', 'device'):
        parser.add_argument('--' + name)
    options = parser.parse_args(argv)
    try:
        config = json.loads(Path(options.config).expanduser().read_text(encoding='utf-8'))
        for name in ('real_dir', 'generated_dir', 'output_dir', 'method_name', 'device'):
            if getattr(options, name) is not None:
                config[name] = getattr(options, name)
        for name in ('real_dir', 'generated_dir', 'output_dir', 'cache_dir'):
            if not config.get(name, '').strip():
                raise ValueError('请填写kid_config.json的{}或传入对应命令行参数。'.format(name))
            config[name] = str(project_path(config[name]))
        for name, minimum in [('batch_size', 1), ('num_workers', 0), ('subset_size', 2),
                              ('num_subsets', 2), ('seed', 0), ('center_crop_size', 0)]:
            if type(config[name]) is not int or config[name] < minimum:
                raise ValueError('{}须为>={}的整数。'.format(name, minimum))
        if config['seed'] >= 2 ** 32 - 1:
            raise ValueError('seed必须小于2**32-1。')
        if type(config['require_equal_counts']) is not bool:
            raise ValueError('require_equal_counts必须为布尔值。')
        run(config)
        return 0
    except Exception as exc:
        print('KID评测失败：{}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
