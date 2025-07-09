import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections.abc import Mapping, MutableMapping
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
        bambiID=None,
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
        speed   = compute_speed(t, xyz)
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
        fname = f"{bambiID}_{plot_name}_{show_common}_quantity_movement.png"
        plt.savefig(os.path.join(save_path, fname), dpi=300)

    plt.close()
    return common_intervals


def marker_outcome(
        marker_velocity: np.ndarray,
        row: dict,
        marker_name: str,
        type_value: str,
        ndigits: int = 2):
    if marker_velocity.size == 0:
        vals = dict(mean=np.nan, std=np.nan, skew=np.nan, max=np.nan)
    else:
        vals = {
            "mean (m/s)": np.nanmean(marker_velocity),
            "std": np.nanstd(marker_velocity),
            "skew": skew(marker_velocity, nan_policy="omit", bias=False),
            "max": np.nanmax(marker_velocity),
        }

    # round & store
    for k, v in vals.items():
        row[f"{k}_{marker_name}_{type_value}"] = round(float(v), ndigits) if np.isfinite(v) else np.nan


def plot_marker_trajectory_mean(
    marker: np.ndarray,
    time: np.ndarray,
    marker_name: str = "MARKER",
    win: int = 500,
    k: float = 2.0,
    component_names: Iterable[str] | None = None,
    figsize: tuple[int, int] = (10, 9),
    save_path: str | None = None,
    plot_name: str | None = None,
    data_type: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Plot the three spatial components of a motion-capture marker with their
    rolling mean, ±k·σ envelope and green fills showing the deviation areas.

    Returns
    -------
    dict
        Nested dict with absolute integrated areas (trajectory vs. mean, and
        trajectory vs. envelope) for each component.
    """
    # ---------------------------------------------------------------
    # 1) Sanity checks & reshape to (n_frames, 3)
    # ---------------------------------------------------------------
    if marker.ndim != 2:
        raise ValueError("`marker` must be 2-D (n_frames × 3 or 3 × n_frames).")

    if marker.shape[0] == 3 and marker.shape[1] != 3:
        marker = marker.T
    if marker.shape[1] != 3:
        raise ValueError("`marker` must have exactly three columns (X, Y, Z).")

    if component_names is None:
        component_names = [f"{marker_name}_x",
                           f"{marker_name}_y",
                           f"{marker_name}_z"]
    if len(component_names) != 3:
        raise ValueError("`component_names` must contain exactly three labels.")

    # ---------------------------------------------------------------
    # 2) Build rolling statistics with pandas
    # ---------------------------------------------------------------
    df = pd.DataFrame(marker, columns=component_names)
    for c in component_names:
        df[f"{c}_mean"]  = df[c].rolling(win, center=True).mean()
        df[f"{c}_std"]   = df[c].rolling(win, center=True).std()
        df[f"{c}_upper"] = df[f"{c}_mean"] + k * df[f"{c}_std"]
        df[f"{c}_lower"] = df[f"{c}_mean"] - k * df[f"{c}_std"]

    # ---------------------------------------------------------------
    # 3) Compute integrated areas            ⟶  output for the caller
    # ---------------------------------------------------------------
    dt = np.mean(np.diff(time))                     # constant sample step (s)
    areas: dict[str, dict[str, float]] = {}

    for c in component_names:
        # |trajectory − mean|
        diff_mean = np.abs(df[c] - df[f"{c}_mean"])
        area_mean = np.nansum(diff_mean) * dt       # integrate via Riemann sum

        # overshoot outside ±kσ envelope
        upper, lower = df[f"{c}_upper"], df[f"{c}_lower"]
        overshoot = np.where(df[c] > upper, df[c] - upper,
                     np.where(df[c] < lower, lower - df[c], 0.0))
        area_env = np.nansum(overshoot) * dt

        axis_key = c.split("_")[-1]                 # "x" | "y" | "z"
        areas[axis_key] = {"mean": area_mean, "envelope": area_env}

    # ---------------------------------------------------------------
    # 4) Plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.25}, constrained_layout=True
    )

    for ax, c in zip(axes, component_names):
        # Trajectory & stats
        ax.plot(time, df[c],                 lw=0.8, alpha=0.8, label=c)
        ax.plot(time, df[f"{c}_mean"], '--', lw=1.2, label=f"mean ({win})")
        ax.fill_between(time,
                        df[f"{c}_lower"], df[f"{c}_upper"],
                        color="gray", alpha=0.2, label=f"±{k}σ")

        # 4-A) Light-green: deviation from mean (always present)
        ax.fill_between(time,
                        df[c], df[f"{c}_mean"],
                        where=~np.isnan(df[f"{c}_mean"]),
                        color="green", alpha=0.35, label="|traj-mean|" )

        # 4-B) Darker green: overshoot outside envelope
        upper, lower = df[f"{c}_upper"], df[f"{c}_lower"]
        ax.fill_between(time, df[c], upper,
                        where=df[c] > upper, color="green", alpha=0.70,
                        label="outside + kσ")
        ax.fill_between(time, df[c], lower,
                        where=df[c] < lower, color="green", alpha=0.70)

        # Cosmetics
        ax.set_ylabel(f"{c.split('_')[-1].upper()}")   # show X / Y / Z
        ax.grid(True)
        ax.legend(loc="upper right", fontsize="x-small")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{marker_name} {data_type} – rolling window {win} frames, ±{k}σ",
        y=0.97
    )

    if save_path and plot_name:
        fname = f"{plot_name}_{data_type}_{marker_name}_outside_mean_std.png"
        plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.close(fig)

    return areas


def compute_area_outside_mean_std(
    markers: Mapping[str, np.ndarray],
    time: np.ndarray,
    freq: float,
    row: MutableMapping[str, float] | None = None,
    *,
    win_mult: float = 2.0,        # rolling-window length in *seconds*
    k: float = 1.0,               # half-width of the envelope in σ
    save_path: str | None = None,
    plot_name: str | None = None,
    data_type: str
):
    """
    Plot every marker trajectory, compute the deviation areas and (optionally)
    write the totals into *row*.

    Parameters
    ----------
    markers : dict[str, np.ndarray]
        Keys are marker names; each value is an (n_frames, 3) array.
    time : np.ndarray
        1-D array of timestamps in seconds (length n_frames).
    freq : float
        Sampling frequency in Hz.
    row : MutableMapping[str, float] or None
        A dictionary-like object (e.g. a Pandas Series) that will be updated
        with two new keys per marker:

            "<marker>_area_outside_mean"
            "<marker>_area_outside_std"

        If *None*, no insertion is done.
    win_mult : float, default 2.0
        Rolling window length expressed in seconds (win = win_mult × freq).
    k : float, default 1.0
        Number of standard deviations for the envelope (± k σ).
    save_path, plot_name : str or None
        Forwarded to `plot_marker_trajectory_mean`.

    Returns
    -------
    dict
        {
          'RANK': {
              'mean_total':     float,   # |traj – mean| integrated over x-y-z
              'envelope_total': float,   # area outside ± k σ integrated over x-y-z
              'per_axis': { 'x': {...}, 'y': {...}, 'z': {...} }
          },
          'LWRA': { … },
          ...
        }
    """
    results = {}
    win = int(win_mult * freq)  # convert seconds → frames

    for name, data in markers.items():
        # 1) Plot and retrieve per-axis areas

        if data_type == "Velocity":
            data = derivative(data, 1/freq)
            data = butter_lowpass_filter(data, cutoff = 6, fs = freq)

        per_axis = plot_marker_trajectory_mean(
            data,
            time,
            marker_name=name,
            win=win,
            k=k,
            save_path=save_path,
            plot_name=plot_name,
            data_type=data_type,
        )

        # 2) Aggregate across the three axes
        mean_total     = round(sum(v['mean']     for v in per_axis.values()),2)
        envelope_total = round(sum(v['envelope'] for v in per_axis.values()),2)

        results[name] = {
            "mean_total":     mean_total,
            "envelope_total": envelope_total,
            "per_axis":       per_axis,
        }

        # 3) Optionally write into the provided row
        if row is not None:
            row[f"{data_type}_{name}_area_outside_mean"] = mean_total
            row[f"{data_type}_{name}_area_outside_std"]  = envelope_total

    return results


def marker_pos_to_jerk(marker_xyz, cutoff, fs):
    dt = 1 / fs

    # Step 2 jerk
    velocity = derivative(marker_xyz, dt)/1000
    velocity_f = butter_lowpass_filter(velocity, cutoff=cutoff, fs=fs)

    # 2) Accélération
    acceleration = derivative(velocity_f, dt)
    acceleration_f = butter_lowpass_filter(acceleration, cutoff=cutoff, fs=fs)

    # 3) Jerk
    jerk = derivative(acceleration_f, dt)
    jerk_f = butter_lowpass_filter(jerk, cutoff=cutoff, fs=fs)

    jerk_mag = np.linalg.norm(jerk_f, axis=1)

    return jerk_mag