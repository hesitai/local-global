"""Plot saved forward trajectories matching the existing strength-1.8 images."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'image_use/3.4_data'
SAVED = ROOT / 'motion-flow-syn-python-flat/output/strength_1.8'
STEM = '1658131893.746021619'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / 'image_use/3.3/trajectory_plots/provenance.json').read_text())
    point = np.array([*prior['reference_pixel_xy'], 1.])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10, 6.2), layout='constrained')
    records = []
    for case, color, style in zip((2, 6, 10, 13),
            ('#3277B5', '#D18B25', '#379367', '#AF5684'), ('-', '--', '-.', ':')):
        paths = list(SAVED.glob(f'{STEM}_{case:02d}_*_mf_para.mat'))
        assert len(paths) == 1
        path = paths[0]
        params = loadmat(path, simplify_cells=True)['para']
        tau = np.asarray(params['tau_list']).reshape(-1)
        matrices = np.asarray(params['homography_source_gt_to_subframe_list'])
        transformed = matrices @ point
        xy = transformed[:, :2] / transformed[:, 2:3] - point[:2]
        assert matrices.shape == (len(tau), 3, 3)
        assert np.isfinite(xy).all() and np.all(np.diff(tau) > 0)
        filename = path.name.replace('_mf_para.mat', '_blurryimg.png')
        selected = ROOT / 'image_use/3.3/1.8' / filename
        original = SAVED / filename
        assert hashlib.sha256(selected.read_bytes()).digest() == hashlib.sha256(original.read_bytes()).digest()
        if case != 13:
            old = np.loadtxt(ROOT / f'image_use/3.3/trajectory_plots/type_{case:02d}_strength_1.8.csv', delimiter=',', skiprows=1)
            assert np.allclose(old[:, :3], np.column_stack((tau, xy)), atol=1e-9)
        weights = np.asarray(params['temporal_weights']).reshape(-1)
        assert np.isclose(weights.sum(), 1)
        np.savetxt(OUT / f'type_{case:02d}.csv', np.column_stack((tau, xy, weights)),
                   delimiter=',', header='tau,dx_px,dy_px,integration_weight', comments='')
        ax.plot(*xy.T, color=color, lw=2.3, ls=style, label=f'Type {case}')
        display_tau = np.linspace(0, 1, 11)
        marks = np.column_stack([np.interp(display_tau, tau, xy[:, i]) for i in range(2)])
        ax.scatter(*marks.T, color=color, s=24, zorder=3)
        ax.scatter(*xy[0], facecolors='white', edgecolors=color, s=85, zorder=4)
        ax.scatter(*xy[-1], color=color, marker='>', s=90, zorder=4)
        records.append(dict(case=case, name=str(params['case_name']), parameter_file=str(path),
                            image=str(selected), image_matches_saved_output=True,
                            case_seed=int(params['case_seed']), subframes=len(tau)))
    ax.set(xlabel=r'Horizontal displacement $\Delta x$ (px)',
           ylabel=r'Vertical displacement $\Delta y$ (px)',
           title='Different motion types, strength = 1.8')
    ax.set_aspect('equal', adjustable='datalim'); ax.invert_yaxis()
    ax.grid(alpha=.18); ax.axhline(0, color='0.7', lw=.7); ax.axvline(0, color='0.7', lw=.7)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=4, frameon=False)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(ROOT / f'image_use/3.4.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'provenance.json').write_text(json.dumps(dict(records=records,
        reference_pixel_xy=point[:2].tolist(), strength=1.8,
        convention='Forward H(t)p-p, full-image coordinates; y down. Equal-time display dots interpolated from saved integration nodes.'), indent=2))
    print('Exported 3.4.png/pdf/svg; image hashes and prior type 2/6/10 curves verified.')

if __name__ == '__main__':
    main()
