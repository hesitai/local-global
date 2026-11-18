#!/usr/bin/env python3
"""Restore response labels only after exact independent validation-pair parity.

Run with the motion generator's .venv Python. Existing validation images,
manifests, source images, labels, training code and checkpoints are read-only.
Newly rendered pairs and labels are written only into a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Keep each CPU worker bounded; parallelism is controlled by --workers.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_ROOT = PROJECT_ROOT.parent / 'motion-flow-syn-python-flat'
sys.path.insert(0, str(GENERATOR_ROOT))
import generate_online_2d_epoch as generator
import kernel_response_labels as labels

DEFAULT_INPUT = PROJECT_ROOT / (
    'dataset/UAV_single_image_pose_split/fixed_protocol_first_half_60_78_visdrone_all/'
    'validation_fixed_online_2d/epoch_000000')
BASE_SEED = 20260810


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def infer_task(row, output_dir, summary):
    sample_id = row['sample_id']
    prefix = sample_id.split('_', 1)[0]
    if not prefix.isdigit():
        raise ValueError('Sample does not contain a numeric source index: ' + sample_id)
    source_index = int(prefix)
    epoch, variant = int(row['epoch']), int(row['variant_index'])
    base_seed = int(row['seed']) - epoch * 1_000_003 - source_index * 10_007 - variant * 97
    if base_seed != BASE_SEED:
        raise ValueError(f'Unexpected inferred base seed {base_seed}: {sample_id}')
    if Path(row['input_path']).stem != sample_id:
        raise ValueError('Input filename/sample identity mismatch: ' + sample_id)
    patch_size = int(row['patch_size'])
    if patch_size != int(summary['patch_size']):
        raise ValueError('Manifest/summary patch size differs')
    return generator.PatchTask(
        source_index=source_index, variant_index=variant,
        source_path=row['source_path'],
        staging_input_path=str(output_dir / 'input' / Path(row['input_path']).name),
        staging_target_path=str(output_dir / 'target' / Path(row['target_path']).name),
        output_name=Path(row['input_path']).name,
        epoch=epoch, base_seed=base_seed, patch_size=patch_size,
        # A forced single case would change random-number consumption.
        cases=tuple(generator.motion.blur_cases),
        strength_min=float(summary['strength_range'][0]),
        strength_max=float(summary['strength_range'][1]),
        adaptive_subframes=bool(summary['adaptive_subframes']), fixed_subframes=31,
        interpolation_margin=2, crop_trials=256,
        kernel_response=True, response_backend='cpu',
    )


def verify_one(row, output_dir, summary):
    cv2.setNumThreads(1)
    started = time.monotonic()
    original_hashes = {key: sha256(row[key])
                       for key in ('input_path', 'target_path', 'source_path')}
    task = infer_task(row, Path(output_dir), summary)
    rendered = generator.render_task(task)
    differences = []
    for key in ('sample_id', 'epoch', 'variant_index', 'case_name', 'strength',
                'strength_band', 'direction_deg', 'crop_top', 'crop_bottom',
                'crop_left', 'crop_right', 'patch_size', 'n_subframes', 'seed'):
        if str(rendered[key]) != str(row[key]):
            differences.append(dict(field=key, original=row[key], rendered=rendered[key]))
    if json.loads(rendered['strength_parameters']) != json.loads(row['strength_parameters']):
        differences.append(dict(field='strength_parameters', original=row['strength_parameters'],
                                rendered=rendered['strength_parameters']))
    if Path(rendered['source_path']).resolve() != Path(row['source_path']).resolve():
        differences.append(dict(field='source_path', original=row['source_path'],
                                rendered=rendered['source_path']))
    parity = {}
    for name, regenerated in [('input_path', task.staging_input_path),
                              ('target_path', task.staging_target_path)]:
        original = cv2.imread(row[name], cv2.IMREAD_UNCHANGED)
        current = cv2.imread(regenerated, cv2.IMREAD_UNCHANGED)
        if original is None or current is None:
            raise ValueError('Could not decode comparison images for ' + row['sample_id'])
        equal = original.shape == current.shape and np.array_equal(original, current)
        parity[name] = dict(pixel_identical=bool(equal), original_shape=list(original.shape),
                            regenerated_shape=list(current.shape),
                            original_sha256=original_hashes[name],
                            regenerated_sha256=sha256(regenerated))
        if not equal:
            differences.append(dict(field=name, reason='decoded pixels differ'))
    for key, before in original_hashes.items():
        if sha256(row[key]) != before:
            differences.append(dict(field=key, reason='original file changed during verification'))
    label_path = Path(rendered['kernel_response_path'])
    expected_metadata = {key: row[key] for key in (
        'sample_id', 'seed', 'variant_index', 'crop_top', 'crop_left', 'source_path')}
    value = labels.load_label(label_path, int(row['patch_size']), 'cpu', expected_metadata)
    if rendered['kernel_response_hash'] != labels.config_hash('cpu'):
        raise ValueError('Response label schema differs')
    return dict(
        sample_id=row['sample_id'], differences=differences, image_parity=parity,
        source_sha256=original_hashes['source_path'], label_shape=list(value.shape),
        kernel_response_path=str(label_path.resolve()),
        kernel_response_hash=rendered['kernel_response_hash'],
        kernel_response_sha256=sha256(label_path), elapsed_seconds=time.monotonic() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_images', type=int, default=0,
                        help='0: all pairs; otherwise first N pairs for a parity pilot')
    args = parser.parse_args()
    if args.workers < 1 or args.max_images < 0:
        parser.error('Require workers >= 1 and max_images >= 0')
    original_dir, output = args.input_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error('Output must be a new directory; existing files are never overwritten')
    if output == original_dir or original_dir in output.parents:
        parser.error('Output must not be inside the original validation directory')
    manifest_path, summary_path = original_dir / 'pairs.tsv', original_dir / 'summary.json'
    with manifest_path.open(newline='') as source:
        original_rows = list(csv.DictReader(source, delimiter='\t'))
    summary = json.loads(summary_path.read_text())
    if summary.get('backend') != 'cpu':
        raise ValueError('Only the original CPU/OpenCV validation backend is supported')
    if len(original_rows) != int(summary['pair_count']):
        raise ValueError('Manifest/summary image counts differ')
    if not original_rows or len({r['sample_id'] for r in original_rows}) != len(original_rows):
        raise ValueError('Manifest is empty or has duplicate sample IDs')
    rows = original_rows[:args.max_images] if args.max_images else original_rows
    for row in rows:
        for key in ('input_path', 'target_path', 'source_path'):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        infer_task(row, output, summary)
    original_manifest_hash = sha256(manifest_path)
    original_summary_hash = sha256(summary_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'input').mkdir()
    (output / 'target').mkdir()
    source_files = [Path(__file__), Path(generator.__file__), Path(generator.motion.__file__),
                    Path(labels.__file__)]
    report = dict(
        status='running', expected_count=len(rows), verified_count=0,
        source_pair_count=len(original_rows), mismatches=[], failures=[],
        input_dir=str(original_dir), output_dir=str(output),
        label_manifest=str(output / 'pairs.tsv'), base_seed=BASE_SEED,
        source_manifest_sha256=original_manifest_hash,
        source_summary_sha256=original_summary_hash,
        generator_sources={str(p.resolve()): sha256(p) for p in source_files},
        response_specification=labels.specification('cpu'),
        kernel_response_hash=labels.config_hash('cpu'),
        settings=dict(workers=args.workers, max_images=args.max_images, fixed_subframes=31,
                      interpolation_margin=2, crop_trials=256, cases='all'),
        verified_samples=[],
    )
    write_json(output / 'parity_report.json', report)
    started = time.monotonic()
    results = {}
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        jobs = {pool.submit(verify_one, row, str(output), summary): row for row in rows}
        for completed, future in enumerate(as_completed(jobs), 1):
            row = jobs[future]
            try:
                result = future.result()
                if result['differences']:
                    report['mismatches'].append(result)
                else:
                    results[row['sample_id']] = result
                    report['verified_count'] += 1
                report['verified_samples'].append(result)
            except Exception as error:
                report['failures'].append(dict(sample_id=row['sample_id'],
                                               error_type=type(error).__name__, message=str(error)))
            if completed % 25 == 0 or completed == len(rows):
                report['elapsed_seconds'] = time.monotonic() - started
                write_json(output / 'parity_report.json', report)
                print(f'{completed}/{len(rows)} complete; verified={report["verified_count"]}; '
                      f'mismatches={len(report["mismatches"])}; failures={len(report["failures"])}; '
                      f'seconds={report["elapsed_seconds"]:.1f}', flush=True)
    for path, expected in [(manifest_path, original_manifest_hash), (summary_path, original_summary_hash)]:
        if sha256(path) != expected:
            report['failures'].append(dict(path=str(path), message='Original manifest/summary changed'))
    # A usable manifest is published only when every requested sample passed.
    report['elapsed_seconds'] = time.monotonic() - started
    if report['mismatches'] or report['failures'] or len(results) != len(rows):
        report['status'] = 'failed'
        write_json(output / 'parity_report.json', report)
        raise SystemExit('Parity verification failed; see parity_report.json. No usable label manifest published.')
    fields = list(original_rows[0])
    for key in ('kernel_response_path', 'kernel_response_hash'):
        if key not in fields:
            fields.append(key)
    with (output / 'pairs.tsv').open('x', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for row in rows:
            result = results[row['sample_id']]
            writer.writerow(dict(row, kernel_response_path=result['kernel_response_path'],
                                 kernel_response_hash=result['kernel_response_hash']))
    report['status'] = 'verified'
    report['label_manifest_sha256'] = sha256(output / 'pairs.tsv')
    report['verified_samples'].sort(key=lambda r: r['sample_id'])
    write_json(output / 'parity_report.json', report)
    print(f'VERIFIED {len(rows)}/{len(rows)} original pairs; labels: {output / "pairs.tsv"}', flush=True)


if __name__ == '__main__':
    main()
