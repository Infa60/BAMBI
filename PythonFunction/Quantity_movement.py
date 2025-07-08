import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from setuptools.sandbox import save_path

from PythonFunction.Base_function import *
from matplotlib.patches   import Patch
from typing import Iterable
import os


def build_segments(time):
    y   = np.zeros_like(time)
    pts = np.array([time, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)

def mask_to_intervals(mask, time):
    """
    Convert a boolean mask to a list of (start, end) intervals.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    # find breakpoints where contiguous run ends
    breaks = np.where(np.diff(idx) > 1)[0]
    splits = np.split(idx, breaks + 1)

    intervals = [(time[s[0]], time[s[-1]]) for s in splits]
    return intervals


def plot_multi_markers_speed_color(
        *, time, fs,
        thr=0.1, gap_tol=1.0,
        cutoff=6, order=2,
        cmap=('blue', 'red'),
        linewidth=6,
        show_common='intersection',
        plot_name=None,
        save_path=None,
        **xyz_mm):

    """
    Draw one horizontal bar per marker showing speed > `thr` (red)
    vs. ≤ `thr` (blue); tiny blue gaps shorter than `gap_tol` seconds
    between two red blocks are merged into red.

    Parameters
    ----------
    time      : (N,) array – shared time stamps (s)
    fs        : float      – sampling frequency (Hz)
    thr       : float      – speed threshold (m/s) after filtering
    gap_tol   : float      – merge blue gaps shorter than this (s)
    cutoff    : float      – Butterworth cut-off (Hz)
    order     : int        – Butterworth filter order
    cmap      : (str,str)  – colors for (blue, red)
    linewidth : int/float  – bar thickness
    show_common : {'intersection', 'union', False}
        * 'intersection' → all markers above the threshold
        * 'union'        → at least one marker above the threshold
        * False          → no extra common bar
    plot_name : str or None, default None
        Base filename (without extension).
        If *None*, a name is built from all marker keys
        (e.g. ``'RANK_LANK_LHIP'``).
    save_path : str or None
        Folder in which the PNG file is written.  If *None*, nothing is saved.
    **xyz_mm  : keyword trajectories, e.g. RANK=array, LANK=array, …
                 positions must be in millimetres
    """

    # ------------------------------------------------------------------
    # Build a default file name if none supplied
    # ------------------------------------------------------------------
    if plot_name is None:
        # insertion order is preserved (Py ≥ 3.7)
        plot_name = "_".join(xyz_mm.keys())

    blue, red = cmap
    t = np.asarray(time)
    n_markers = len(xyz_mm)
    extra = 1 if show_common else 0          # +1 row for common bar

    # Figure height: ~0.47 inch per bar + margins
    fig_h = 0.47 * (n_markers + extra) + 0.7
    fig, axes = plt.subplots(nrows=n_markers + extra,
                             figsize=(10, fig_h),
                             sharex=True)
    if n_markers + extra == 1:
        axes = [axes]

    red_masks = []   # store each marker’s red mask for later

    # --------------------------------------------------------------
    # Iterate through markers
    # --------------------------------------------------------------
    for ax, (name, xyz) in zip(axes, xyz_mm.items()):
        # 1) speed in m/s + low-pass filter
        speed   = compute_speed(t, xyz)/1000
        speed_f = butter_lowpass_filter(speed, cutoff, fs, order)

        # 2) threshold and gap merging
        red_mask = speed_f > thr
        idx = np.where(red_mask)[0]
        for k in range(len(idx) - 1):
            if t[idx[k+1]] - t[idx[k]] < gap_tol:
                red_mask[idx[k]:idx[k+1]+1] = True
        red_masks.append(red_mask)

        # 3) build and add colored segments
        segs   = build_segments(t)
        colors = [red if f else blue for f in red_mask[:-1]]
        ax.add_collection(LineCollection(segs, colors=colors,
                                         linewidth=linewidth))

        # aesthetics
        ax.set_ylim(-0.3, 0.3)
        ax.set_yticks([])
        ax.text(-0.02, 0.5, name, ha='right', va='center',
                transform=ax.transAxes, fontsize=9)
        ax.spines[['left', 'right', 'top']].set_visible(False)

    # --------------------------------------------------------------
    # Common (intersection or union) bar
    # --------------------------------------------------------------
    if show_common == 'intersection':
        common_mask = np.logical_and.reduce(red_masks)
        common_intervals = mask_to_intervals(common_mask, t)
        ax_common = axes[-1]                       # last row
        segs = build_segments(t)
        colors = [red if f else blue for f in common_mask[:-1]]
        ax_common.add_collection(LineCollection(segs, colors=colors,
                                                linewidth=linewidth))
        ax_common.set_ylim(-0.3, 0.3)
        ax_common.set_yticks([])
        ax_common.text(-0.02, 0.5, 'COMMON', ha='right', va='center',
                       transform=ax_common.transAxes, fontsize=9,
                       fontweight='bold')
        ax_common.spines[['left', 'right', 'top']].set_visible(False)

    if show_common == 'union':
        # use logical OR instead of AND  ➜ at least one marker is red
        union_mask = np.logical_or.reduce(red_masks)
        common_intervals = mask_to_intervals(union_mask, t)

        ax_common = axes[-1]  # last row (extra bar)
        segs = build_segments(t)
        colors = [red if f else blue for f in union_mask[:-1]]
        ax_common.add_collection(
            LineCollection(segs, colors=colors, linewidth=linewidth)
        )

        ax_common.set_ylim(-0.3, 0.3)
        ax_common.set_yticks([])
        ax_common.text(-0.02, 0.5, 'UNION', ha='right', va='center',
                       transform=ax_common.transAxes, fontsize=9,
                       fontweight='bold')
        ax_common.spines[['left', 'right', 'top']].set_visible(False)

    # --------------------------------------------------------------
    # Shared X axis and legend
    # --------------------------------------------------------------
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(t.min(), t.max())
    axes[-1].ticklabel_format(useOffset=False)

    axes[0].legend(
        handles=[Patch(color=red,  label=f'> {thr} m/s (merged)'),
                 Patch(color=blue, label=f'≤ {thr} m/s')],
        loc='upper right', fontsize=8)

    # Compact vertical spacing
    fig.subplots_adjust(hspace=0.15)
    plt.tight_layout()

    # Save file if a folder is provided
    if save_path is not None:
        fname = f"{plot_name}_{show_common}_quantity_movement.png"
        plt.savefig(os.path.join(save_path, fname), dpi=300)

    plt.close()
    return common_intervals


def marker_velocity_outcome(
        marker_velocity: np.ndarray,
        row: dict,
        marker_name: str,
        ndigits: int = 2):
    if marker_velocity.size == 0:
        vals = dict(mean=np.nan, std=np.nan, skew=np.nan, max=np.nan)
    else:
        vals = {
            "mean": np.nanmean(marker_velocity),
            "std": np.nanstd(marker_velocity),
            "skew": skew(marker_velocity, nan_policy="omit", bias=False),
            "max": np.nanmax(marker_velocity),
        }

    # round & store
    for k, v in vals.items():
        row[f"{k}_{marker_name}_velocity"] = round(float(v), ndigits) if np.isfinite(v) else np.nan


def plot_marker_trajectory_mean(
    marker: np.ndarray,
    time,
    marker_name: str = "MARKER",
    win: int = 500,
    k: float = 2.0,
    component_names: Iterable[str] | None = None,
    figsize: tuple[int, int] = (10, 9),
    save_path: str = None,
    plot_name: str = None,
) -> None:
    """
    Plot the three spatial components of a motion-capture marker with a
    rolling mean and ±k·σ envelope.

    Parameters
    ----------
    marker : np.ndarray
        Array whose shape is (n_frames, 3) **or** (3, n_frames).
    marker_name : str, default "MARKER"
        Name used in titles and legends (e.g. "RANK", "LHIP", "head").
    win : int, default 500
        Size of the centred rolling window (in frames).
    k : float, default 2.0
        Envelope half-width expressed in standard deviations.
    component_names : Iterable[str] or None, default None
        Custom labels for the three axes.  If *None*, a default of
        ``[f"{marker_name}_x", "{marker_name}_y", "{marker_name}_z"]``
        is used.
    figsize : (int, int), default (10, 9)
        Figure size passed to `plt.subplots`.

    Returns
    -------
    None
        The function draws the figure; nothing is returned.
    """
    # ---------------------------------------------------------------
    # 1) Standardise orientation → (n_frames, 3)
    # ---------------------------------------------------------------
    if marker.ndim != 2:
        raise ValueError("`marker` must be a 2-D array with three components")

    if marker.shape[0] == 3 and marker.shape[1] != 3:
        marker = marker.T  # turn (3, N) into (N, 3)

    if marker.shape[1] != 3:
        raise ValueError("`marker` must have exactly three columns (X, Y, Z)")

    # ---------------------------------------------------------------
    # 2) Build DataFrame so that pandas can do rolling stats
    # ---------------------------------------------------------------
    if component_names is None:
        component_names = [f"{marker_name}_x",
                           f"{marker_name}_y",
                           f"{marker_name}_z"]

    if len(component_names) != 3:
        raise ValueError("`component_names` must contain exactly three labels")

    df = pd.DataFrame(marker, columns=component_names)

    for c in component_names:
        df[f"{c}_mean"]  = df[c].rolling(win, center=True).mean()
        df[f"{c}_std"]   = df[c].rolling(win, center=True).std()
        df[f"{c}_upper"] = df[f"{c}_mean"] + k * df[f"{c}_std"]
        df[f"{c}_lower"] = df[f"{c}_mean"] - k * df[f"{c}_std"]

    # ---------------------------------------------------------------
    # 3) Plot: three stacked axes sharing X
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.25}, constrained_layout=True
    )

    for ax, c in zip(axes, component_names):
        ax.plot(time, df[c],                 lw=0.8, alpha=0.8, label=c)
        ax.plot(time, df[f"{c}_mean"], '--', lw=1.2,
                label=f"mean ({win})")
        ax.fill_between(time,
                        df[f"{c}_lower"], df[f"{c}_upper"],
                        color="gray", alpha=0.2, label=f"±{k}σ")
        ax.set_ylabel(f" Trajectory ({c.split("_")[-1]})")          # show x / y / z only
        ax.grid(True)
        ax.legend(loc="upper right", fontsize="small")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{marker_name} trajectory – rolling window {win} frames, ±{k}σ",
        y=0.95
    )
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{plot_name}_{marker_name}_TMW.png"), dpi=300)
    plt.close()
