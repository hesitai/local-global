"""No-reference image metrics, evaluated on the exact exported 8-bit RGB image.

NIQE/BRISQUE/MUSIQ use PyIQA's named default models (``niqe``, ``brisque``,
``musiq``). Input is NCHW RGB float32 in [0, 1]; no external resize is applied.
PyIQA performs each metric's own standard internal preprocessing. Record the
installed PyIQA version when comparing experiments.

LV = population variance of cv2.Laplacian(gray_uint8, CV_64F, ksize=1,
BORDER_DEFAULT). Entropy = Shannon entropy (bits) of a 256-bin gray histogram.
Gray conversion is cv2.COLOR_RGB2GRAY on uint8 pixels for LV/Entropy/CPBD.

CPBD is a bundled compatibility adaptation of the python-cpbd reference port:
https://github.com/0x64746b/python-cpbd (cpbd/compute.py, cpbd/octave.py).
It retains that port's Canny/Sobel/Marziliano conventions, 64x64 blocks,
beta=3.6, contrast-dependent JNB widths 5/3 and histogram bins 0..63.
Its numerical results are not claimed to be identical to proprietary MATLAB
edge detection. A CPBD of zero on an edgeless image follows the reference
convention; it does not establish whether the image has good overall quality.

Original metric: N. D. Narvekar and L. J. Karam, IEEE TIP 20(9), 2678-2683,
2011, doi:10.1109/TIP.2011.2131660. Publications using this code must acknowledge
the IVU Lab and cite the original software/papers as required below.

CPBD LICENSE (retained from the reference implementation)
Copyright (c) 2009-2010 Arizona Board of Regents. All Rights Reserved.
Contact: Lina Karam (karam@asu.edu) and Niranjan Narvekar (nnarveka@asu.edu)
Image, Video, and Usabilty (IVU) Lab, http://ivulab.asu.edu, Arizona State University
This copyright statement may not be removed from any file containing it or from
modifications to these files. This copyright notice must also be included in any
file or product that is derived from the source files.
Redistribution and use of this code in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
- Redistribution's of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
- Redistribution's in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- The Image, Video, and Usability Laboratory (IVU Lab, http://ivulab.asu.edu)
  is acknowledged in any publication that reports research results using this
  code, copies of this code, or modifications of this code.
The code and our papers are to be cited in the bibliography as:
N. D. Narvekar and L. J. Karam, "CPBD Sharpness Metric Software",
http://ivulab.asu.edu/Quality/CPBD
N. D. Narvekar and L. J. Karam, "A No-Reference Image Blur Metric Based on the
Cumulative Probability of Blur Detection (CPBD)," IEEE Transactions on Image
Processing, 2011.
N. D. Narvekar and L. J. Karam, "An Improved No-Reference Sharpness Metric Based
on the Probability of Blur Detection," International Workshop on Video
Processing and Quality Metrics for Consumer Electronics (VPQM), January 2010,
http://www.vpqm.org (pdf)
N. D. Narvekar and L. J. Karam, "A No Reference Perceptual Quality Metric based
on Cumulative Probability of Blur Detection," First International Workshop on
the Quality of Multimedia Experience (QoMEX), pp. 87-91, July 2009.
DISCLAIMER:
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall the Arizona Board of Regents, Arizona State
University, IVU Lab members, authors or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages (including,
but not limited to, procurement of substitute goods or services; loss of use,
data, or profits; or business interruption) however caused and on any theory
of liability, whether in contract, strict liability, or tort (including
negligence or otherwise) arising in any way out of the use of this software,
even if advised of the possibility of such damage.
"""

import math
import time

import cv2
import numpy as np


METRIC_NAMES = ('NIQE', 'BRISQUE', 'LV', 'MUSIQ', 'CPBD', 'Entropy')
METRIC_DIRECTIONS = {
    'NIQE': 'lower', 'BRISQUE': 'lower', 'LV': 'higher',
    'MUSIQ': 'higher', 'CPBD': 'higher', 'Entropy': 'higher',
}
_PYIQA_NAMES = {'NIQE': 'niqe', 'BRISQUE': 'brisque', 'MUSIQ': 'musiq'}


def _validate_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise TypeError('Metrics require an RGB uint8 numpy array.')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Expected nonempty H x W x 3 RGB image.')


def laplacian_variance(gray):
    """LV on gray [0,255], population variance, 3x3 cross Laplacian kernel."""
    return float(cv2.Laplacian(
        gray, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_DEFAULT).var())


def information_entropy(gray):
    """256-bin Shannon grayscale entropy in bits (range [0, 8])."""
    histogram = np.bincount(gray.reshape(-1), minlength=256)
    probabilities = histogram[histogram > 0].astype(np.float64) / gray.size
    return float(abs(-np.sum(probabilities * np.log2(probabilities))))


def _cpbd_sobel_edges(image):
    """Octave-style vertical Sobel and simple thinning used by python-cpbd."""
    from scipy.ndimage import convolve

    kernel = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]) / 8.
    strength = np.square(convolve(image, kernel.T))
    threshold = 2 * np.sqrt(np.mean(strength))
    strength[strength <= threshold] = 0
    padded = np.pad(strength, 1, mode='constant')
    horizontal = ((strength > padded[1:-1, :-2]) &
                  (strength > padded[1:-1, 2:]))
    vertical = ((strength > padded[:-2, 1:-1]) &
                (strength > padded[2:, 1:-1]))
    return horizontal | vertical


def _cpbd_edge_widths(edges, image):
    """Marziliano widths, preserving the public Python port's conventions."""
    gradient_y, gradient_x = np.gradient(image)
    angles = np.zeros(image.shape, dtype=np.float64)
    nonzero_x = gradient_x != 0
    angles[nonzero_x] = np.degrees(np.arctan2(
        gradient_y[nonzero_x], gradient_x[nonzero_x]))
    angles[(gradient_x == 0) & (gradient_y == np.pi / 2)] = 90
    widths = np.zeros(image.shape, dtype=np.float64)
    # Preserve the Python port's all-zero-angle behavior for reproducibility.
    if not np.any(angles):
        return widths
    quantized = 45 * np.round(angles / 45)
    height, width = image.shape
    candidates = edges & ((quantized == 0) | (np.abs(quantized) == 180))
    candidates[[0, -1], :] = False
    candidates[:, [0, -1]] = False
    rows, cols = np.nonzero(candidates)
    rising = quantized[rows, cols] == 0
    left = _cpbd_scan_widths(image, rows, cols, rising, -1)
    right = _cpbd_scan_widths(image, rows, cols, rising, 1)
    widths[rows, cols] = left + right
    return widths


def _cpbd_scan_widths(image, rows, cols, rising, step):
    """Run the original <=100-pixel monotonic scan over all active edges.

    Each array element stops on exactly the same border or intensity test as
    the scalar reference loop. Only the independent edge scans are vectorized;
    the 101-step limit, integer widths and threshold behavior are unchanged.
    """
    side_widths = np.full(rows.size, 101, dtype=np.int64)
    active = np.arange(rows.size)
    for margin in range(101):
        if not active.size:
            break
        outer = cols[active] + step * (margin + 2)
        in_bounds = (outer >= 0) & (outer < image.shape[1])
        side_widths[active[~in_bounds]] = margin + 1
        active = active[in_bounds]
        outer = outer[in_bounds]
        if not active.size:
            break
        inner = cols[active] + step * (margin + 1)
        difference = image[rows[active], outer] - image[rows[active], inner]
        if step < 0:
            stopped = np.where(rising[active], difference >= 0, difference <= 0)
        else:
            stopped = np.where(rising[active], difference <= 0, difference >= 0)
        side_widths[active[stopped]] = margin + 1
        active = active[~stopped]
    return side_widths


def cpbd_score(gray):
    """CPBD on grayscale uint8; at least one complete 64x64 block is required.

    Incomplete bottom/right border blocks are omitted as in the reference.
    Edgeless valid images return 0, as in the reference. Smaller images raise
    ValueError rather than reporting an unsupported zero score.
    """
    from skimage.feature import canny

    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('CPBD expects a two-dimensional uint8 grayscale image.')
    if min(gray.shape) < 64:
        raise ValueError('CPBD requires height and width >= 64; no resizing applied.')
    image = gray.astype(np.float64)
    canny_edges = canny(image)
    widths = _cpbd_edge_widths(_cpbd_sobel_edges(image), image)
    histogram = np.zeros(101, dtype=np.int64)
    for top in range(0, (image.shape[0] // 64) * 64, 64):
        for left in range(0, (image.shape[1] // 64) * 64, 64):
            rows, cols = slice(top, top + 64), slice(left, left + 64)
            if np.count_nonzero(canny_edges[rows, cols]) <= 64 * 64 * 0.002:
                continue
            local_widths = widths[rows, cols]
            local_widths = local_widths[local_widths != 0]
            block = image[rows, cols]
            contrast = int(block.max() - block.min())
            jnb_width = 5. if contrast <= 50 else 3.
            probabilities = 1 - np.exp(-np.abs(local_widths / jnb_width) ** 3.6)
            buckets = np.rint(probabilities * 100).astype(np.int64)
            histogram += np.bincount(buckets, minlength=101)
    count = int(histogram.sum())
    return float(histogram[:64].sum() / count) if count else 0.


class MetricSuite:
    """Compute selected automatic metrics, keeping failures explicitly missing.

    Initialization/import/download failures for requested PyIQA models are
    fatal. Per-image failures return None and populate ``last_errors``; they
    must be excluded from means and counted in the report. NIQE needs at
    least two complete 96x96 blocks to estimate image feature covariance.
    No smaller image is enlarged merely to make a score computable.
    MOS is human-rated and intentionally does not belong to this class.
    ``last_timings`` contains per-metric elapsed seconds (including failures).
    CUDA work is synchronized around each learned metric. The first learned
    metric also includes the shared RGB tensor transfer; grayscale conversion
    happens once outside individual metric timings.
    """

    def __init__(self, names=None, device='cpu'):
        self.names = tuple(METRIC_NAMES if names is None else names)
        if len(set(self.names)) != len(self.names):
            raise ValueError('Duplicate metric names are not allowed.')
        unknown = set(self.names) - set(METRIC_NAMES)
        if unknown:
            raise ValueError('Unknown automatic metrics: ' + ', '.join(sorted(unknown)))
        self.device = str(device)
        self.models = {}
        self.last_errors = {}
        self.last_timings = {}
        self.torch = None
        learned = [name for name in self.names if name in _PYIQA_NAMES]
        if learned:
            try:
                import pyiqa
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    'NIQE/BRISQUE/MUSIQ require PyIQA and PyTorch. Install the '
                    'evaluation requirements, or select LV CPBD Entropy for '
                    'an offline classical-metric smoke check.') from exc
            self.torch = torch
            for name in learned:
                try:
                    self.models[name] = pyiqa.create_metric(
                        _PYIQA_NAMES[name], device=self.device).eval()
                except Exception as exc:
                    raise RuntimeError(
                        'Unable to initialize {} (PyIQA {}). Check dependencies '
                        'and pretrained-weight availability: {}'.format(
                            name, getattr(pyiqa, '__version__', 'unknown'), exc)) from exc

    def __call__(self, rgb_uint8):
        _validate_rgb(rgb_uint8)
        self.last_errors = {}
        self.last_timings = {}
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        tensor = None
        results = {}
        classical = {'LV': laplacian_variance, 'CPBD': cpbd_score,
                     'Entropy': information_entropy}
        for name in self.names:
            started = time.perf_counter()
            try:
                sync_cuda = name in self.models and self.device.startswith('cuda')
                if sync_cuda:
                    self.torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                if name in classical:
                    value = classical[name](gray)
                else:
                    height, width = rgb_uint8.shape[:2]
                    if name == 'NIQE' and (
                            min(height, width) < 96 or
                            (height // 96) * (width // 96) < 2):
                        raise ValueError(
                            'NIQE requires at least two complete 96x96 blocks; '
                            'image is {}x{}. No resizing applied.'.format(width, height))
                    if tensor is None:
                        tensor = self.torch.from_numpy(np.ascontiguousarray(
                            rgb_uint8.transpose(2, 0, 1))).unsqueeze(0)
                        tensor = tensor.to(device=self.device,
                                           dtype=self.torch.float32) / 255.
                    with self.torch.inference_mode():
                        value = self.models[name](tensor).item()
                if sync_cuda:
                    self.torch.cuda.synchronize(self.device)
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError('Metric returned a nonfinite score.')
                results[name] = value
            except Exception as exc:
                results[name] = None
                self.last_errors[name] = '{}: {}'.format(type(exc).__name__, exc)
            finally:
                self.last_timings[name] = time.perf_counter() - started
        return results
