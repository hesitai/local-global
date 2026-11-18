"""Controlled type-3/type-4 comparison; leaves generator defaults unchanged."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import script_uav_global_blur_multisubframe_no_depth as gen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'image_use/3.3/curved_speed'
SOURCE = ROOT / 'motion-flow-syn-python-flat/dataset/image/5/1658131893.746021619.jpg'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    image = gen.read_image_double(SOURCE)
    curves, renders, boxes, records = [], [], [], []
    tau = np.linspace(0, 1, 1001)
    for gamma in (1, 2, 4, 8):
        case = '03_heading_turning_uniform' if gamma == 1 else '04_heading_turning_variable_speed'
        trajectory = gen.make_uav_trajectory(case, 1234)
        trajectory['time_mapping']['gamma'] = float(gamma)
        gen.scale_trajectory_motion_amplitudes(trajectory, 1.8)
        xy = np.array([gen.evaluate_trajectory_at_time(trajectory, float(t))[:2] for t in tau])
        curves.append(xy)
        name = f'type_{3 if gamma == 1 else 4:02d}_gamma_{gamma}'
        np.savetxt(OUT / f'{name}.csv', np.column_stack((tau, tau**gamma, xy)),
                   delimiter=',', header='tau,normalized_arc_progress,dx_px,dy_px', comments='')
        blur, _, params, _ = gen.synthesize_blur_by_multisubframe_v2(image, trajectory, 101, 2)
        renders.append((name, blur))
        boxes.append([int(v) for v in params['crop_rectangle_top_bottom_left_right']])
        records.append(dict(name=name, gamma=gamma, strength=1.8, seed=1234,
                            subframes=int(params['N_subframes_actual']), trajectory=trajectory))
        print(name, 'subframes:', params['N_subframes_actual'], flush=True)
        assert np.allclose(xy[500], 0)
    top = max(b[0] for b in boxes); bottom = min(b[1] for b in boxes)
    left = max(b[2] for b in boxes); right = min(b[3] for b in boxes)
    gen.write_image_uint8(OUT / 'sharp.png', image[top:bottom, left:right])
    for name, blur in renders:
        gen.write_image_uint8(OUT / f'{name}.png', blur[top:bottom, left:right])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'svg.fonttype': 'none'})
    colors = ['#3277B5', '#D18B25', '#379367', '#AF5684']
    fig, ax = plt.subplots(figsize=(10, 6.5), layout='constrained')
    for gamma, xy, color, style in zip((1, 2, 4, 8), curves, colors, ('-', '--', '-.', ':')):
        label = f'Type 3: uniform ($\\gamma=1$)' if gamma == 1 else f'Type 4: $\\gamma={gamma}$'
        ax.plot(*xy.T, color=color, lw=2.3, ls=style, label=label)
        # Equally spaced time marks, not the adaptive integration nodes.
        ax.scatter(*xy[::100].T, color=color, s=25, zorder=3)
        ax.scatter(*xy[0], facecolors='white', edgecolors=color, s=90, zorder=4)
        ax.scatter(*xy[-1], color=color, marker='>', s=95, zorder=4)
    ax.set(xlabel=r'Horizontal displacement $\Delta x$ (px)',
           ylabel=r'Vertical displacement $\Delta y$ (px)', title='Curved motion with different speed profiles')
    ax.set_aspect('equal', adjustable='datalim'); ax.invert_yaxis()
    ax.grid(alpha=.18); ax.axhline(0, color='0.7', lw=.7); ax.axvline(0, color='0.7', lw=.7)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, frameon=False)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(ROOT / f'image_use/3.3.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'metadata.json').write_text(json.dumps(dict(source=str(SOURCE),
        common_crop=[top,bottom,left,right], configs=records,
        note='s=tau^gamma; gamma is a speed-profile exponent, not constant acceleration. '
             'Trajectories are relative to exposure midpoint, explaining their translations. '
             'Dots mark equal time intervals; identical geometric path and strength across cases.'),
        indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x)))

if __name__ == '__main__':
    main()
