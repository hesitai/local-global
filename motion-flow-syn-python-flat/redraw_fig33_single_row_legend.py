"""Replot saved trajectories without regenerating blur images."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'image_use/3.3/curved_speed'
TARGET = ROOT / 'image_use/3-3.png'

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10, 6.5), layout='constrained')
    for gamma, color, style in zip((1, 2, 4, 8),
                                  ('#3277B5', '#D18B25', '#379367', '#AF5684'),
                                  ('-', '--', '-.', ':')):
        kind = 3 if gamma == 1 else 4
        xy = np.loadtxt(DATA / f'type_{kind:02d}_gamma_{gamma}.csv',
                        delimiter=',', skiprows=1)[:, 2:4]
        label = (r'Type 3: uniform ($\gamma=1$)' if gamma == 1
                 else f'Type 4: $\\gamma={gamma}$')
        ax.plot(*xy.T, color=color, lw=2.3, ls=style, label=label)
        ax.scatter(*xy[::100].T, color=color, s=25, zorder=3)
        ax.scatter(*xy[0], facecolors='white', edgecolors=color, s=90, zorder=4)
        ax.scatter(*xy[-1], color=color, marker='>', s=95, zorder=4)
    ax.set(xlabel=r'Horizontal displacement $\Delta x$ (px)',
           ylabel=r'Vertical displacement $\Delta y$ (px)',
           title='Curved motion with different speed profiles')
    ax.set_aspect('equal', adjustable='datalim')
    ax.invert_yaxis()
    ax.grid(alpha=.18)
    ax.axhline(0, color='0.7', lw=.7)
    ax.axvline(0, color='0.7', lw=.7)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=4,
              frameon=False, fontsize=11, columnspacing=1.3, handlelength=2.2)
    backup = TARGET.with_name('3-3_before_single_row_legend.png')
    if not backup.exists():
        shutil.copy2(TARGET, backup)
    fig.savefig(TARGET, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
