import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, PowerNorm
from PythonFunction.Base_function import *
from matplotlib.patches   import Patch



def plot_multi_markers_speed_color(
        *,                # force keyword-only
        time,
        fs,
        thr=0.1,
        gap_tol=1.0,
        cutoff=6,
        order=2,
        cmap=('blue', 'red'),
        **xyz_mm):
    """
    Draw one horizontal bar per marker:
    - red  where filtered speed > thr  (m/s)
      and where blue gaps < gap_tol are merged into red;
    - blue elsewhere.

    Pass the markers as keyword arguments, e.g.:
        plot_multi_markers_speed_color(time=t, fs=100, thr=0.12,
                                       RANK=RANK, LANK=LANK, HEAD=HEAD)
    """
    blue, red = cmap
    time = np.asarray(time)
    n = len(xyz_mm)

    fig, axes = plt.subplots(nrows=n,
                             figsize=(10, 1.8 * n),
                             sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, xyz) in zip(axes, xyz_mm.items()):
        # 1) speed in m/s + filtering
        speed = compute_speed(time, xyz) /1000
        speed_f = butter_lowpass_filter(speed, cutoff=cutoff,
                                        fs=fs, order=order)

        # 2) threshold + gap merge
        red_mask = speed_f > thr
        red_idx = np.where(red_mask)[0]
        for k in range(len(red_idx) - 1):
            if time[red_idx[k+1]] - time[red_idx[k]] < gap_tol:
                red_mask[red_idx[k]:red_idx[k+1]+1] = True

        # 3) build coloured segments
        y = np.zeros_like(time)
        pts = np.array([time, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        colors = [red if f else blue for f in red_mask[:-1]]
        lc = LineCollection(segs, colors=colors, linewidth=8)
        ax.add_collection(lc)

        # cosmetics
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_title(name, loc='left')
        ax.spines[['left', 'right', 'top']].set_visible(False)

    # shared X axis
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(time.min(), time.max())
    axes[-1].ticklabel_format(useOffset=False)      # no scientific offset

    # legend once, in the first axis
    axes[0].legend(handles=[Patch(color=red,  label=f'> {thr} m/s or merged'),
                            Patch(color=blue, label=f'≤ {thr} m/s')],
                   loc='upper right')

    plt.tight_layout()
    plt.show()
    return fig


def build_segments(time):
    y   = np.zeros_like(time)
    pts = np.array([time, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)

# --- main ------------------------------------------------------------------
def plot_multi_markers_speed_color2(
        *, time, fs,
        thr=0.1, gap_tol=1.0,
        cutoff=6, order=2,
        cmap=('blue', 'red'),
        linewidth=6,
        **xyz_mm):

    blue, red = cmap
    t   = np.asarray(time)
    rng = t.max() - t.min()
    n   = len(xyz_mm)

    # figure plus compacte : 1.2 cm de hauteur par barre ≈ 0.47 inch
    fig_h = 0.47 * n + 0.7           # 0.7 inch de marge bas/haut
    fig, axes = plt.subplots(nrows=n, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, xyz) in zip(axes, xyz_mm.items()):
        # ---------- vitesse filtrée -----------------------------------
        speed   = compute_speed(t, xyz) /1000                       # m/s
        speed_f = butter_lowpass_filter(speed, cutoff, fs, order)

        # ---------- masque rouge + fusion des gaps -------------------
        red_mask = speed_f > thr
        idx = np.where(red_mask)[0]
        for k in range(len(idx) - 1):
            if t[idx[k+1]] - t[idx[k]] < gap_tol:
                red_mask[idx[k]:idx[k+1]+1] = True

        # ---------- segments colorés ---------------------------------
        segs   = build_segments(t)
        colors = [red if f else blue for f in red_mask[:-1]]
        ax.add_collection(LineCollection(segs, colors=colors, linewidth=linewidth))

        # ---------- esthétique ---------------------------------------
        ax.set_ylim(-0.3, 0.3)          # barre plus fine, moins d’espace vide
        ax.set_yticks([])
        ax.spines[['left', 'right', 'top']].set_visible(False)

        # Label du marqueur à gauche, centré verticalement
        ax.text(-0.02, 0.5, name, ha='right', va='center',
                transform=ax.transAxes, fontsize=9)

    # axe X commun
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(t.min(), t.max())
    axes[-1].ticklabel_format(useOffset=False)

    # légende unique
    axes[0].legend(handles=[Patch(color=red,  label=f'> {thr} m/s or merged'),
                            Patch(color=blue, label=f'≤ {thr} m/s')],
                   loc='upper right', fontsize=8)

    # espace vertical réduit entre subplots
    fig.subplots_adjust(hspace=0.15)
    plt.tight_layout()
    plt.show()
    return fig