#!/usr/bin/env python3
"""Offline analysis for three-boat Part 6 trajectory logger CSV files."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend('Agg')


DESIRED_DISTANCE_M = 5.0
DEFAULT_CONVERGENCE_TOLERANCE_M = 0.5
DEFAULT_CONVERGENCE_DWELL_S = 5.0
DEFAULT_TURN_RATE_THRESHOLD_RAD_S = 0.05

REQUIRED_COLUMNS = [
    'time_s',
    'r1_x_m',
    'r1_y_m',
    'r1_speed_mps',
    'r1_yaw_rad',
    'r2_x_m',
    'r2_y_m',
    'r2_speed_mps',
    'r2_yaw_rad',
    'r3_x_m',
    'r3_y_m',
    'r3_speed_mps',
    'r3_yaw_rad',
    'reference_x_m',
    'reference_y_m',
    'leader_tracking_error_m',
    'd12_euclidean_m',
    'd23_euclidean_m',
    'e12_euclidean_m',
    'e23_euclidean_m',
    'r2_path_gap_m',
    'r3_path_gap_m',
    'r2_path_gap_error_m',
    'r3_path_gap_error_m',
]


def finite_values(values):
    """Return the finite numeric values from an array."""
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def mean_or_nan(values):
    values = finite_values(values)
    return float(np.mean(values)) if values.size else float('nan')


def max_or_nan(values):
    values = finite_values(values)
    return float(np.max(values)) if values.size else float('nan')


def min_or_nan(values):
    values = finite_values(values)
    return float(np.min(values)) if values.size else float('nan')


def rmse(values):
    values = finite_values(values)
    return (
        float(np.sqrt(np.mean(np.square(values))))
        if values.size
        else float('nan')
    )


def mean_absolute_error(values):
    values = finite_values(values)
    return (
        float(np.mean(np.abs(values)))
        if values.size
        else float('nan')
    )


def percentile_95(values):
    values = finite_values(values)
    return (
        float(np.percentile(values, 95))
        if values.size
        else float('nan')
    )


def convergence_time(times, errors, tolerance, dwell):
    """Return confirmation time for a consecutive in-band sample run."""
    times = np.asarray(times, dtype=float)
    errors = np.asarray(errors, dtype=float)
    inside = np.isfinite(times) & np.isfinite(errors)
    inside &= np.abs(errors) <= tolerance

    run_start = None
    for index, is_inside in enumerate(inside):
        if not is_inside:
            run_start = None
            continue

        if run_start is None:
            run_start = index

        if times[index] - times[run_start] >= dwell:
            return float(times[run_start])

    return None


def correlation(x_values, y_values):
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = np.asarray(x_values, dtype=float)[mask]
    y_values = np.asarray(y_values, dtype=float)[mask]
    if (
        x_values.size < 2
        or np.ptp(x_values) == 0.0
        or np.ptp(y_values) == 0.0
    ):
        return float('nan')
    return float(np.corrcoef(x_values, y_values)[0, 1])


def peak_and_time(times, values):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(times) & np.isfinite(values)
    if not np.any(valid):
        return float('nan'), float('nan')
    valid_indices = np.flatnonzero(valid)
    peak_index = valid_indices[np.argmax(values[valid])]
    return float(values[peak_index]), float(times[peak_index])


def validate_and_load(input_csv):
    try:
        frame = pd.read_csv(input_csv)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            f'Could not read input CSV {input_csv}: {error}'
        ) from error

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            'Input CSV is missing required columns: '
            + ', '.join(missing)
        )
    if frame.empty:
        raise ValueError('Input CSV contains no data rows.')

    for column in REQUIRED_COLUMNS:
        original = frame[column]
        numeric = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            rows = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
            raise ValueError(
                f'Column {column!r} contains non-numeric values '
                f'on CSV row(s) {rows}.'
            )
        frame[column] = numeric.astype(float)

    times = frame['time_s'].to_numpy(dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError('Column time_s must contain only finite values.')
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError('Column time_s must be strictly increasing.')
    return frame


def save_figure(figure, output_path):
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def plot_trajectories(frame, output_dir):
    figure, axis = plt.subplots(figsize=(9, 8))
    series = [
        ('Reference', 'reference_x_m', 'reference_y_m', 'k--'),
        ('R1', 'r1_x_m', 'r1_y_m', 'b-'),
        ('R2', 'r2_x_m', 'r2_y_m', 'orange'),
        ('R3', 'r3_x_m', 'r3_y_m', 'g-'),
    ]
    for label, x_column, y_column, style in series:
        axis.plot(
            frame[x_column],
            frame[y_column],
            style,
            label=label,
            linewidth=1.5,
        )
    axis.set_title('Three-boat trajectories and reference')
    axis.set_xlabel('X [m]')
    axis.set_ylabel('Y [m]')
    axis.set_aspect('equal', adjustable='datalim')
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure(figure, output_dir / '01_trajectories.png')


def plot_leader_error(frame, output_dir):
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(
        frame['time_s'],
        frame['leader_tracking_error_m'],
        label='Leader tracking error',
    )
    axis.set_title('R1 leader tracking error')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Error [m]')
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure(figure, output_dir / '02_leader_tracking_error.png')


def plot_euclidean_distances(frame, output_dir):
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(frame['time_s'], frame['d12_euclidean_m'], label='d12 Euclidean')
    axis.plot(frame['time_s'], frame['d23_euclidean_m'], label='d23 Euclidean')
    axis.axhline(
        DESIRED_DISTANCE_M,
        color='black',
        linestyle='--',
        label='Desired 5.0 m',
    )
    axis.set_title('Euclidean inter-robot distances')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Distance [m]')
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure(figure, output_dir / '03_euclidean_distances.png')

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(frame['time_s'], frame['e12_euclidean_m'], label='e12 Euclidean')
    axis.plot(frame['time_s'], frame['e23_euclidean_m'], label='e23 Euclidean')
    axis.axhline(0.0, color='black', linestyle='--', label='Zero error')
    axis.set_title('Euclidean distance errors (separate from path gaps)')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Signed error [m]')
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure(figure, output_dir / '04_euclidean_distance_errors.png')


def path_gap_available(values):
    return bool(np.isfinite(np.asarray(values, dtype=float)).any())


def plot_path_gaps(frame, output_dir):
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(
        frame['time_s'],
        frame['r2_path_gap_m'],
        label='R2 controller path gap',
    )
    axis.plot(
        frame['time_s'],
        frame['r3_path_gap_m'],
        label='R3 controller path gap',
    )
    axis.axhline(
        DESIRED_DISTANCE_M,
        color='black',
        linestyle='--',
        label='Desired 5.0 m',
    )
    axis.set_title('Controller breadcrumb path-gap measurements')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Path gap [m]')
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure(figure, output_dir / '05_path_gap.png')


def plot_error_propagation(frame, output_dir, follower_error_sources):
    times = frame['time_s'].to_numpy(dtype=float)
    leader_error = frame['leader_tracking_error_m'].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(times, leader_error, label='Leader tracking error')

    for follower, source in follower_error_sources.items():
        axis.plot(
            times,
            np.abs(frame[source].to_numpy(dtype=float)),
            label=f'|{follower} error| ({source})',
        )

    axis.set_title('Leader and follower error time series')
    axis.set_xlabel('Time [s]')
    axis.set_ylabel('Absolute error [m]')
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure(figure, output_dir / '06_error_propagation.png')


def make_turn_regions(axis, times, turn_mask):
    indices = np.flatnonzero(turn_mask)
    if not indices.size:
        return
    starts = [indices[0]]
    ends = []
    for previous, current in zip(indices[:-1], indices[1:]):
        if current != previous + 1:
            ends.append(previous)
            starts.append(current)
    ends.append(indices[-1])
    for start, end in zip(starts, ends):
        axis.axvspan(
            times[start],
            times[end],
            color='gold',
            alpha=0.22,
            label='TURN region' if start == starts[0] else None,
        )


def plot_turn_stability(
    frame,
    output_dir,
    yaw_rate,
    turn_mask,
    turn_error_sources,
):
    times = frame['time_s'].to_numpy(dtype=float)
    figure, error_axis = plt.subplots(figsize=(11, 5))
    rate_axis = error_axis.twinx()
    make_turn_regions(error_axis, times, turn_mask)

    for label, column in turn_error_sources.items():
        error_axis.plot(
            times,
            frame[column].to_numpy(dtype=float),
            label=label,
        )

    rate_axis.plot(
        times,
        yaw_rate,
        color='gray',
        alpha=0.65,
        linewidth=1.0,
        label='R1 yaw rate',
    )
    error_axis.set_title('Spacing errors and R1 turn regions')
    error_axis.set_xlabel('Time [s]')
    error_axis.set_ylabel('Signed spacing error [m]')
    rate_axis.set_ylabel('Yaw rate [rad/s]')
    error_axis.grid(True, alpha=0.3)

    handles, labels = error_axis.get_legend_handles_labels()
    rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
    error_axis.legend(handles + rate_handles, labels + rate_labels, loc='best')
    save_figure(figure, output_dir / '07_turn_stability.png')


def metric_value(value):
    if value is None:
        return 'not converged'
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return 'not available'
    return value


def format_metric(value, unit=''):
    value = metric_value(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f'{float(value):.4f}{unit}'
    return str(value)


def write_metrics_csv(metrics, output_path):
    pd.DataFrame(
        [
            {'metric': name, 'value': metric_value(value)}
            for name, value in metrics.items()
        ]
    ).to_csv(output_path, index=False)


def write_summary(
    frame,
    metrics,
    output_path,
    convergence_tolerance,
    convergence_dwell,
    turn_threshold,
    turn_sample_count,
    straight_sample_count,
    propagation_sources,
):
    lines = [
        'Guidebook Part 6 - Three-boat performance analysis',
        f'Samples: {len(frame)}',
        (
            'Experiment duration: '
            f'{frame["time_s"].iloc[-1] - frame["time_s"].iloc[0]:.3f} s'
        ),
        '',
        'Q33 - Leader trajectory tracking',
        (
            'Tracking error [m]: '
            f'mean={format_metric(metrics["leader_tracking_mean_m"])}, '
            f'max={format_metric(metrics["leader_tracking_max_m"])}, '
            f'RMSE={format_metric(metrics["leader_tracking_rmse_m"])}, '
            f'p95={format_metric(metrics["leader_tracking_p95_m"])}.'
        ),
        (
            'These values describe the recorded distance from R1 to the '
            'nearest reference point; no pass/fail threshold was applied.'
        ),
        '',
        'Q34 - Follower distance maintenance',
        'Euclidean spacing (separate from controller path gap):',
    ]
    for pair in ('e12', 'e23'):
        lines.append(
            f'  {pair}: signed mean='
            f'{format_metric(metrics[f"{pair}_mean_signed_m"])} m, '
            f'MAE={format_metric(metrics[f"{pair}_mean_abs_m"])} m, '
            f'max |error|={format_metric(metrics[f"{pair}_max_abs_m"])} m, '
            f'RMSE={format_metric(metrics[f"{pair}_rmse_m"])} m, '
            f'distance range=['
            f'{format_metric(metrics[f"d{pair[1]}{pair[2]}_min_m"])}, '
            f'{format_metric(metrics[f"d{pair[1]}{pair[2]}_max_m"])}] m.'
        )
    lines.extend([
        (
            'Euclidean convergence uses tolerance '
            f'{convergence_tolerance:.3f} m and dwell '
            f'{convergence_dwell:.3f} s; reported time is the start of the '
            'first in-band run later confirmed to satisfy the dwell.'
        ),
        'Controller breadcrumb path-gap measurements:',
        (
            '  R2: mean gap='
            f'{format_metric(metrics["r2_path_gap_mean_m"])} m, '
            f'MAE error='
            f'{format_metric(metrics["r2_path_gap_mean_abs_error_m"])} m, '
            f'max |error|='
            f'{format_metric(metrics["r2_path_gap_max_abs_error_m"])} m, '
            f'RMSE error={format_metric(metrics["r2_path_gap_rmse_m"])} m, '
            f'convergence='
            f'{format_metric(metrics["r2_path_gap_convergence_time_s"], " s")}.'
        ),
        (
            '  R3: mean gap='
            f'{format_metric(metrics["r3_path_gap_mean_m"])} m, '
            f'MAE error='
            f'{format_metric(metrics["r3_path_gap_mean_abs_error_m"])} m, '
            f'max |error|='
            f'{format_metric(metrics["r3_path_gap_max_abs_error_m"])} m, '
            f'RMSE error={format_metric(metrics["r3_path_gap_rmse_m"])} m, '
            f'convergence='
            f'{format_metric(metrics["r3_path_gap_convergence_time_s"], " s")}.'
        ),
        (
            'Path-gap metrics exclude NaN samples; Euclidean metrics use '
            'their own columns and are not replaced by path-gap values.'
        ),
        '',
        'Q35 - Error propagation',
        (
            'Peak leader tracking error='
            f'{format_metric(metrics["leader_tracking_peak_m"])} m at '
            f'{format_metric(metrics["leader_tracking_peak_time_s"], " s")}.'
        ),
        (
            f'Peak R2 error ({propagation_sources["R2"]})='
            f'{format_metric(metrics["r2_propagation_peak_m"])} m at '
            f'{format_metric(metrics["r2_propagation_peak_time_s"], " s")}; '
            f'Pearson correlation with leader error='
            f'{format_metric(metrics["leader_r2_error_correlation"])}.'
        ),
        (
            f'Peak R3 error ({propagation_sources["R3"]})='
            f'{format_metric(metrics["r3_propagation_peak_m"])} m at '
            f'{format_metric(metrics["r3_propagation_peak_time_s"], " s")}; '
            f'Pearson correlation with leader error='
            f'{format_metric(metrics["leader_r3_error_correlation"])}.'
        ),
        (
            'Correlations describe association in these recorded samples and '
            'do not establish causality.'
        ),
        '',
        'Q36 - Stability during turns',
        (
            'Turn definition: R1 yaw is unwrapped, yaw rate is estimated by '
            'finite differences, and TURN means '
            f'|yaw_rate| > {turn_threshold:.4f} rad/s. '
            f'TURN samples={turn_sample_count}; STRAIGHT samples='
            f'{straight_sample_count}.'
        ),
        (
            'TURN RMSE [m]: leader tracking='
            f'{format_metric(metrics["turn_leader_tracking_rmse_m"])}, '
            f'R1-R2 Euclidean spacing error='
            f'{format_metric(metrics["turn_e12_rmse_m"])}, '
            f'R2-R3 Euclidean spacing error='
            f'{format_metric(metrics["turn_e23_rmse_m"])}.'
        ),
        (
            'STRAIGHT RMSE [m]: leader tracking='
            f'{format_metric(metrics["straight_leader_tracking_rmse_m"])}, '
            f'R1-R2 Euclidean spacing error='
            f'{format_metric(metrics["straight_e12_rmse_m"])}, '
            f'R2-R3 Euclidean spacing error='
            f'{format_metric(metrics["straight_e23_rmse_m"])}.'
        ),
        (
            'STRAIGHT controller path-gap error RMSE [m]: '
            f'R2={format_metric(metrics["straight_r2_path_gap_rmse_m"])}, '
            f'R3={format_metric(metrics["straight_r3_path_gap_rmse_m"])}.'
        ),
        (
            'During turns: minimum d12='
            f'{format_metric(metrics["turn_d12_min_m"])} m, '
            f'minimum d23={format_metric(metrics["turn_d23_min_m"])} m, '
            f'max |e12|={format_metric(metrics["turn_e12_max_abs_m"])} m, '
            f'max |e23|={format_metric(metrics["turn_e23_max_abs_m"])} m; '
            f'R2 path-gap RMSE='
            f'{format_metric(metrics["turn_r2_path_gap_rmse_m"])} m, '
            f'R3 path-gap RMSE='
            f'{format_metric(metrics["turn_r3_path_gap_rmse_m"])} m.'
        ),
        (
            'These section-specific metrics describe the recorded turn and '
            'straight samples; no guidebook pass/fail threshold was applied.'
        ),
    ])
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def analyze(frame, output_dir, tolerance, dwell, turn_threshold):
    times = frame['time_s'].to_numpy(dtype=float)
    leader = frame['leader_tracking_error_m'].to_numpy(dtype=float)
    e12 = frame['e12_euclidean_m'].to_numpy(dtype=float)
    e23 = frame['e23_euclidean_m'].to_numpy(dtype=float)
    d12 = frame['d12_euclidean_m'].to_numpy(dtype=float)
    d23 = frame['d23_euclidean_m'].to_numpy(dtype=float)
    r2_gap = frame['r2_path_gap_m'].to_numpy(dtype=float)
    r3_gap = frame['r3_path_gap_m'].to_numpy(dtype=float)
    r2_gap_error = frame['r2_path_gap_error_m'].to_numpy(dtype=float)
    r3_gap_error = frame['r3_path_gap_error_m'].to_numpy(dtype=float)

    propagation_sources = {
        'R2': (
            'r2_path_gap_error_m'
            if path_gap_available(r2_gap_error)
            else 'e12_euclidean_m'
        ),
        'R3': (
            'r3_path_gap_error_m'
            if path_gap_available(r3_gap_error)
            else 'e23_euclidean_m'
        ),
    }
    follower_errors = {
        'R2': frame[propagation_sources['R2']].to_numpy(dtype=float),
        'R3': frame[propagation_sources['R3']].to_numpy(dtype=float),
    }

    yaw = frame['r1_yaw_rad'].to_numpy(dtype=float)
    yaw_rate = np.full(yaw.shape, np.nan, dtype=float)
    valid_yaw = np.isfinite(yaw)
    if valid_yaw.sum() >= 2:
        valid_times = times[valid_yaw]
        if np.any(np.diff(valid_times) <= 0.0):
            raise ValueError(
                'Finite r1_yaw_rad samples must have increasing time_s.'
            )
        yaw_rate[valid_yaw] = np.gradient(
            np.unwrap(yaw[valid_yaw]),
            valid_times,
        )

    turn_mask = np.isfinite(yaw_rate) & (
        np.abs(yaw_rate) > turn_threshold
    )
    straight_mask = np.isfinite(yaw_rate) & ~turn_mask
    turn_error_sources = {
        'R1-R2 spacing error': 'e12_euclidean_m',
        'R2-R3 spacing error': 'e23_euclidean_m',
    }

    plot_trajectories(frame, output_dir)
    plot_leader_error(frame, output_dir)
    plot_euclidean_distances(frame, output_dir)
    plot_path_gaps(frame, output_dir)
    plot_error_propagation(frame, output_dir, propagation_sources)
    plot_turn_stability(
        frame,
        output_dir,
        yaw_rate,
        turn_mask,
        turn_error_sources,
    )

    turn_values = lambda column: frame.loc[turn_mask, column].to_numpy(
        dtype=float
    )
    straight_values = lambda column: frame.loc[
        straight_mask,
        column,
    ].to_numpy(dtype=float)

    metrics = {
        'leader_tracking_mean_m': mean_or_nan(leader),
        'leader_tracking_max_m': max_or_nan(leader),
        'leader_tracking_rmse_m': rmse(leader),
        'leader_tracking_p95_m': percentile_95(leader),
        'e12_mean_signed_m': mean_or_nan(e12),
        'e12_mean_abs_m': mean_absolute_error(e12),
        'e12_max_abs_m': max_or_nan(np.abs(e12)),
        'e12_rmse_m': rmse(e12),
        'd12_min_m': min_or_nan(d12),
        'd12_max_m': max_or_nan(d12),
        'e12_convergence_time_s': convergence_time(
            times, e12, tolerance, dwell
        ),
        'e23_mean_signed_m': mean_or_nan(e23),
        'e23_mean_abs_m': mean_absolute_error(e23),
        'e23_max_abs_m': max_or_nan(np.abs(e23)),
        'e23_rmse_m': rmse(e23),
        'd23_min_m': min_or_nan(d23),
        'd23_max_m': max_or_nan(d23),
        'e23_convergence_time_s': convergence_time(
            times, e23, tolerance, dwell
        ),
        'r2_path_gap_mean_m': mean_or_nan(r2_gap),
        'r2_path_gap_mean_abs_error_m': mean_absolute_error(r2_gap_error),
        'r2_path_gap_max_abs_error_m': max_or_nan(np.abs(r2_gap_error)),
        'r2_path_gap_rmse_m': rmse(r2_gap_error),
        'r2_path_gap_convergence_time_s': (
            convergence_time(times, r2_gap_error, tolerance, dwell)
            if path_gap_available(r2_gap_error)
            else float('nan')
        ),
        'r3_path_gap_mean_m': mean_or_nan(r3_gap),
        'r3_path_gap_mean_abs_error_m': mean_absolute_error(r3_gap_error),
        'r3_path_gap_max_abs_error_m': max_or_nan(np.abs(r3_gap_error)),
        'r3_path_gap_rmse_m': rmse(r3_gap_error),
        'r3_path_gap_convergence_time_s': (
            convergence_time(times, r3_gap_error, tolerance, dwell)
            if path_gap_available(r3_gap_error)
            else float('nan')
        ),
        'leader_tracking_peak_m': max_or_nan(leader),
        'leader_tracking_peak_time_s': peak_and_time(times, leader)[1],
        'r2_propagation_peak_m': max_or_nan(np.abs(follower_errors['R2'])),
        'r2_propagation_peak_time_s': peak_and_time(
            times, np.abs(follower_errors['R2'])
        )[1],
        'r3_propagation_peak_m': max_or_nan(np.abs(follower_errors['R3'])),
        'r3_propagation_peak_time_s': peak_and_time(
            times, np.abs(follower_errors['R3'])
        )[1],
        'leader_r2_error_correlation': correlation(
            leader, np.abs(follower_errors['R2'])
        ),
        'leader_r3_error_correlation': correlation(
            leader, np.abs(follower_errors['R3'])
        ),
        'turn_leader_tracking_rmse_m': rmse(
            frame.loc[turn_mask, 'leader_tracking_error_m']
        ),
        'turn_e12_rmse_m': rmse(turn_values('e12_euclidean_m')),
        'turn_e23_rmse_m': rmse(turn_values('e23_euclidean_m')),
        'turn_r2_path_gap_rmse_m': rmse(turn_values('r2_path_gap_error_m')),
        'turn_r3_path_gap_rmse_m': rmse(turn_values('r3_path_gap_error_m')),
        'turn_d12_min_m': min_or_nan(turn_values('d12_euclidean_m')),
        'turn_d23_min_m': min_or_nan(turn_values('d23_euclidean_m')),
        'turn_e12_max_abs_m': max_or_nan(
            np.abs(turn_values('e12_euclidean_m'))
        ),
        'turn_e23_max_abs_m': max_or_nan(
            np.abs(turn_values('e23_euclidean_m'))
        ),
        'straight_leader_tracking_rmse_m': rmse(
            frame.loc[straight_mask, 'leader_tracking_error_m']
        ),
        'straight_e12_rmse_m': rmse(straight_values('e12_euclidean_m')),
        'straight_e23_rmse_m': rmse(straight_values('e23_euclidean_m')),
        'straight_r2_path_gap_rmse_m': rmse(
            straight_values('r2_path_gap_error_m')
        ),
        'straight_r3_path_gap_rmse_m': rmse(
            straight_values('r3_path_gap_error_m')
        ),
    }
    metrics['turn_rate_threshold_rad_s'] = turn_threshold
    metrics['turn_sample_count'] = int(turn_mask.sum())
    metrics['straight_sample_count'] = int(straight_mask.sum())
    metrics['convergence_tolerance_m'] = tolerance
    metrics['convergence_dwell_s'] = dwell

    write_metrics_csv(metrics, output_dir / 'metrics_summary.csv')
    write_summary(
        frame,
        metrics,
        output_dir / 'part6_summary.txt',
        tolerance,
        dwell,
        turn_threshold,
        int(turn_mask.sum()),
        int(straight_mask.sum()),
        propagation_sources,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze a three-boat Part 6 trajectory logger CSV.'
    )
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument(
        '--convergence-tolerance',
        type=float,
        default=DEFAULT_CONVERGENCE_TOLERANCE_M,
    )
    parser.add_argument(
        '--convergence-dwell',
        type=float,
        default=DEFAULT_CONVERGENCE_DWELL_S,
    )
    parser.add_argument(
        '--turn-rate-threshold',
        type=float,
        default=DEFAULT_TURN_RATE_THRESHOLD_RAD_S,
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.convergence_tolerance)
        or args.convergence_tolerance < 0.0
    ):
        parser.error('--convergence-tolerance must be finite and >= 0.')
    if (
        not math.isfinite(args.convergence_dwell)
        or args.convergence_dwell < 0.0
    ):
        parser.error('--convergence-dwell must be finite and >= 0.')
    if (
        not math.isfinite(args.turn_rate_threshold)
        or args.turn_rate_threshold < 0.0
    ):
        parser.error('--turn-rate-threshold must be finite and >= 0.')
    return args


def main():
    args = parse_args()
    try:
        frame = validate_and_load(args.input_csv)
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else args.input_csv.resolve().parent / 'part6_analysis'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = analyze(
            frame,
            output_dir,
            args.convergence_tolerance,
            args.convergence_dwell,
            args.turn_rate_threshold,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f'Analysis failed: {error}') from error

    duration = float(frame['time_s'].iloc[-1] - frame['time_s'].iloc[0])
    generated_files = [
        '01_trajectories.png',
        '02_leader_tracking_error.png',
        '03_euclidean_distances.png',
        '04_euclidean_distance_errors.png',
        '05_path_gap.png',
        '06_error_propagation.png',
        '07_turn_stability.png',
        'metrics_summary.csv',
        'part6_summary.txt',
    ]

    print(f'Samples: {len(frame)}')
    print(f'Total experiment duration: {duration:.3f} s')
    print(f'Output directory: {output_dir}')
    print(
        'Leader RMSE: '
        f'{format_metric(metrics["leader_tracking_rmse_m"], " m")}'
    )
    print(f'e12 RMSE: {format_metric(metrics["e12_rmse_m"], " m")}')
    print(f'e23 RMSE: {format_metric(metrics["e23_rmse_m"], " m")}')
    print(f'Minimum d12: {format_metric(metrics["d12_min_m"], " m")}')
    print(f'Minimum d23: {format_metric(metrics["d23_min_m"], " m")}')
    print(
        'Convergence times: '
        f'e12={format_metric(metrics["e12_convergence_time_s"], " s")}, '
        f'e23={format_metric(metrics["e23_convergence_time_s"], " s")}, '
        f'R2 path gap='
        f'{format_metric(metrics["r2_path_gap_convergence_time_s"], " s")}, '
        f'R3 path gap='
        f'{format_metric(metrics["r3_path_gap_convergence_time_s"], " s")}'
    )
    print('Generated files:')
    for filename in generated_files:
        print(f'  {output_dir / filename}')


if __name__ == '__main__':
    main()
