"""Versioned, deterministic frequency-response labels; never draws random numbers.

The response is for linear-RGB backward sampling B(p)=sum w*S(p+d).
Four integer pixels around each stride-16 cell centre are averaged. This grid
is exactly closed under flips/transposes; half-integer output pixels are NOT
silently subjected to another interpolation. CPU labels include OpenCV's
1/32-pixel interpolation table convention; CUDA uses ideal bilinear weights.
This is a sparse local operator descriptor, not an sRGB/global inverse OTF.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

STRIDE = 16
RADII = (1 / 64, 1 / 32, 1 / 16, 1 / 8)
FREQUENCIES = np.asarray([
    (r * np.cos(j * np.pi / 8), r * np.sin(j * np.pi / 8))
    for r in RADII for j in range(8)
], dtype=np.float64)
CHANNELS = 2 * len(FREQUENCIES)


def specification(backend='cpu'):
    if backend not in ('cpu', 'cuda'):
        raise ValueError('Unsupported response backend')
    return dict(schema='kernel_response_v1', stride=STRIDE,
                frequencies=FREQUENCIES.tolist(), channels='interleaved_real_imag',
                grid='mean_of_centre_2x2_integer_pixels', sign='+',
                domain='linear_rgb_backward_sampling', backend=backend,
                interpolation='opencv_table32' if backend == 'cpu' else 'bilinear')


def config_hash(backend='cpu'):
    payload = json.dumps(specification(backend), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def response_from_homographies(homographies, weights, top, left, size, backend='cpu'):
    specification(backend)
    if size % STRIDE:
        raise ValueError('Response patch must be divisible by 16')
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid exposure weights')
    g = size // STRIDE
    yy, xx = np.meshgrid(np.arange(g) * STRIDE, np.arange(g) * STRIDE, indexing='ij')
    points = np.stack([
        np.stack((xx + left + dx, yy + top + dy), -1)
        for dy in (7, 8) for dx in (7, 8)
    ]).reshape(-1, 2).astype(np.float64)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1).T
    result = np.zeros((len(points), len(FREQUENCIES)), dtype=np.complex128)
    for h, w in zip(homographies, weights):
        if np.allclose(h, np.eye(3), rtol=0, atol=1e-12):
            result += float(w)
            continue
        q = (np.linalg.inv(h) @ homogeneous).T
        q = q[:, :2] / q[:, 2:3]
        if backend == 'cpu':
            q = np.rint(q * 32) / 32
        base = np.floor(q)
        frac = q - base
        phase = 2 * np.pi * ((base - points) @ FREQUENCIES.T)
        # Separable bilinear sum, one large exponential instead of four.
        bx = 1-frac[:, 0:1] + frac[:, 0:1] * np.exp(2j*np.pi*FREQUENCIES[:, 0])
        by = 1-frac[:, 1:2] + frac[:, 1:2] * np.exp(2j*np.pi*FREQUENCIES[:, 1])
        result += float(w) * np.exp(1j * phase) * bx * by
    result = result.reshape(4, g, g, -1).mean(0)
    label = np.stack((result.real, result.imag), -1).reshape(g, g, CHANNELS)
    return label.transpose(2, 0, 1).astype(np.float32)


def save_label(path, response, metadata, backend='cpu'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, response=response.astype(np.float16),
                        config_hash=np.asarray(config_hash(backend)),
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_label(path, size, backend='cpu', expected_metadata=None):
    with np.load(path, allow_pickle=False) as data:
        if str(data['config_hash'].item()) != config_hash(backend):
            raise ValueError(f'Response schema mismatch: {path}')
        value = data['response'].astype(np.float32)
        meta = json.loads(str(data['metadata'].item()))
    if value.shape != (CHANNELS, size // STRIDE, size // STRIDE) or not np.isfinite(value).all():
        raise ValueError(f'Invalid response label: {path}')
    if np.max(np.abs(value)) > 1.002:
        raise ValueError(f'Unbounded response label: {path}')
    for key, expected in (expected_metadata or {}).items():
        if str(meta.get(key)) != str(expected):
            raise ValueError(f'Response/sample mismatch {key}: {path}')
    return value


def augmentation_channels(mode):
    """T'(f)=T(A^T f); missing negative frequencies use conjugation."""
    a = np.eye(2)
    if mode & 1:
        a = np.diag([-1, 1]) @ a
    if mode & 2:
        a = np.diag([1, -1]) @ a
    if mode & 4:
        a = np.asarray([[0, 1], [1, 0]]) @ a
    indices, signs = [], []
    for f in FREQUENCIES @ a:
        distances = np.linalg.norm(FREQUENCIES - f, axis=1)
        reverse = np.linalg.norm(FREQUENCIES + f, axis=1)
        if distances.min() < 1e-9:
            j, sign = int(distances.argmin()), 1
        elif reverse.min() < 1e-9:
            j, sign = int(reverse.argmin()), -1
        else:
            raise ValueError('Frequency set not closed under D4')
        indices.extend((2*j, 2*j+1))
        signs.extend((1, sign))
    return indices, signs
