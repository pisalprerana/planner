"""Shortest-path Dubins curves and arc-length sampling."""

import math

import numpy as np


_TWO_PI = 2.0 * math.pi


def _mod2pi(angle):
    return angle % _TWO_PI


def _candidate_paths(alpha, beta, distance):
    sin_alpha = math.sin(alpha)
    sin_beta = math.sin(beta)
    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_alpha_beta = math.cos(alpha - beta)
    candidates = []

    p_squared = (
        2.0 + distance**2 - 2.0 * cos_alpha_beta
        + 2.0 * distance * (sin_alpha - sin_beta)
    )
    if p_squared >= -1e-12:
        p = math.sqrt(max(0.0, p_squared))
        angle = math.atan2(
            cos_beta - cos_alpha,
            distance + sin_alpha - sin_beta
        )
        candidates.append(("LSL", _mod2pi(-alpha + angle), p,
                           _mod2pi(beta - angle)))

    p_squared = (
        2.0 + distance**2 - 2.0 * cos_alpha_beta
        + 2.0 * distance * (-sin_alpha + sin_beta)
    )
    if p_squared >= -1e-12:
        p = math.sqrt(max(0.0, p_squared))
        angle = math.atan2(
            cos_alpha - cos_beta,
            distance - sin_alpha + sin_beta
        )
        candidates.append(("RSR", _mod2pi(alpha - angle), p,
                           _mod2pi(-beta + angle)))

    p_squared = (
        -2.0 + distance**2 + 2.0 * cos_alpha_beta
        + 2.0 * distance * (sin_alpha + sin_beta)
    )
    if p_squared >= -1e-12:
        p = math.sqrt(max(0.0, p_squared))
        angle = math.atan2(
            -cos_alpha - cos_beta,
            distance + sin_alpha + sin_beta
        ) - math.atan2(-2.0, math.sqrt(max(0.0, p_squared)))
        candidates.append(("LSR", _mod2pi(-alpha + angle), p,
                           _mod2pi(-beta + angle)))

    p_squared = (
        -2.0 + distance**2 + 2.0 * cos_alpha_beta
        - 2.0 * distance * (sin_alpha + sin_beta)
    )
    if p_squared >= -1e-12:
        p = math.sqrt(max(0.0, p_squared))
        angle = math.atan2(
            cos_alpha + cos_beta,
            distance - sin_alpha - sin_beta
        ) - math.atan2(2.0, math.sqrt(max(0.0, p_squared)))
        candidates.append(("RSL", _mod2pi(alpha - angle), p,
                           _mod2pi(beta - angle)))

    value = (
        6.0 - distance**2 + 2.0 * cos_alpha_beta
        + 2.0 * distance * (sin_alpha - sin_beta)
    ) / 8.0
    if abs(value) <= 1.0 + 1e-12:
        value = min(1.0, max(-1.0, value))
        p = _mod2pi(_TWO_PI - math.acos(value))
        angle = math.atan2(
            cos_alpha - cos_beta,
            distance - sin_alpha + sin_beta
        )
        t = _mod2pi(alpha - angle + p / 2.0)
        q = _mod2pi(alpha - beta - t + p)
        candidates.append(("RLR", t, p, q))

    value = (
        6.0 - distance**2 + 2.0 * cos_alpha_beta
        + 2.0 * distance * (-sin_alpha + sin_beta)
    ) / 8.0
    if abs(value) <= 1.0 + 1e-12:
        value = min(1.0, max(-1.0, value))
        p = _mod2pi(_TWO_PI - math.acos(value))
        angle = math.atan2(
            cos_alpha - cos_beta,
            distance + sin_alpha - sin_beta
        )
        t = _mod2pi(-alpha - angle + p / 2.0)
        q = _mod2pi(beta - alpha - t + p)
        candidates.append(("LRL", t, p, q))

    return candidates


def _advance(state, mode, distance, turning_radius):
    x, y, yaw = state

    if mode == "S":
        return np.array([
            x + distance * math.cos(yaw),
            y + distance * math.sin(yaw),
            yaw
        ])

    angle = distance / turning_radius
    if mode == "L":
        next_yaw = yaw + angle
        return np.array([
            x + turning_radius * (math.sin(next_yaw) - math.sin(yaw)),
            y + turning_radius * (math.cos(yaw) - math.cos(next_yaw)),
            next_yaw
        ])

    next_yaw = yaw - angle
    return np.array([
        x + turning_radius * (math.sin(yaw) - math.sin(next_yaw)),
        y + turning_radius * (math.cos(next_yaw) - math.cos(yaw)),
        next_yaw
    ])


def generate_dubins_path(
    start,
    goal,
    turning_radius,
    straight_step,
    curve_step=None
):
    """Return the shortest forward-only Dubins path sampled as ``[x, y, yaw]``.

    Yaw is in radians and remains unwrapped so it changes continuously along
    the sampled curve. The final yaw may differ from ``goal[2]`` by a multiple
    of 2*pi, while representing the same orientation.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start and goal must each contain x, y, and yaw.")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError("start and goal values must be finite.")
    if not math.isfinite(turning_radius) or turning_radius <= 0.0:
        raise ValueError("turning_radius must be a positive finite value.")
    if curve_step is None:
        curve_step = straight_step
    if not math.isfinite(straight_step) or straight_step <= 0.0:
        raise ValueError("straight_step must be a positive finite value.")
    if not math.isfinite(curve_step) or curve_step <= 0.0:
        raise ValueError("curve_step must be a positive finite value.")

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    straight_distance = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    alpha = _mod2pi(start[2] - theta)
    beta = _mod2pi(goal[2] - theta)
    normalized_distance = straight_distance / turning_radius

    candidates = _candidate_paths(alpha, beta, normalized_distance)
    if not candidates:
        raise RuntimeError("No valid Dubins path was found.")

    path_type, t, p, q = min(
        candidates,
        key=lambda candidate: sum(candidate[1:])
    )

    state = start.copy()
    samples = [state.copy()]
    for mode, normalized_length in zip(path_type, (t, p, q)):
        segment_length = normalized_length * turning_radius
        if segment_length <= 1e-12:
            continue

        step_size = straight_step if mode == "S" else curve_step
        sample_count = max(1, math.ceil(segment_length / step_size))
        increment = segment_length / sample_count
        for _ in range(sample_count):
            state = _advance(state, mode, increment, turning_radius)
            samples.append(state.copy())

    samples[-1][0] = goal[0]
    samples[-1][1] = goal[1]
    return np.asarray(samples, dtype=float)