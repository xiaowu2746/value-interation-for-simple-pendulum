"""Pendulum dynamics and grid helpers for value iteration."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin


@dataclass(frozen=True)
class PendulumParams:
    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    damping: float = 0.05
    torque_limit: float = 3.0
    dt: float = 0.05


@dataclass(frozen=True)
class GridSpec:
    theta_min: float = -pi
    theta_max: float = pi
    omega_min: float = -8.0
    omega_max: float = 8.0
    n_theta: int = 61
    n_omega: int = 61


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi). The upright target is theta = 0."""
    return (theta + pi) % (2.0 * pi) - pi


def angle_error(theta: float, target: float = 0.0) -> float:
    return wrap_angle(theta - target)


def make_linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def make_theta_grid(spec: GridSpec) -> list[float]:
    # Exclude pi because -pi and pi represent the same angle.
    step = (spec.theta_max - spec.theta_min) / spec.n_theta
    return [spec.theta_min + i * step for i in range(spec.n_theta)]


def make_omega_grid(spec: GridSpec) -> list[float]:
    return make_linspace(spec.omega_min, spec.omega_max, spec.n_omega)


def nearest_index(value: float, grid: list[float]) -> int:
    if value <= grid[0]:
        return 0
    if value >= grid[-1]:
        return len(grid) - 1
    step = (grid[-1] - grid[0]) / (len(grid) - 1)
    idx = int(round((value - grid[0]) / step))
    return max(0, min(len(grid) - 1, idx))


def nearest_theta_index(theta: float, theta_grid: list[float]) -> int:
    theta = wrap_angle(theta)
    step = 2.0 * pi / len(theta_grid)
    idx = int(round((theta - theta_grid[0]) / step)) % len(theta_grid)
    return idx


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def derivatives(theta: float, omega: float, torque: float, params: PendulumParams) -> tuple[float, float]:
    """Return [theta_dot, omega_dot].

    theta = 0 is the upright equilibrium. With zero input this equilibrium is
    unstable, so the gravity term has positive sign near theta = 0.
    """
    inertia = params.mass * params.length * params.length
    theta_dot = omega
    omega_dot = (
        params.gravity / params.length * sin(theta)
        + torque / inertia
        - params.damping * omega
    )
    return theta_dot, omega_dot


def rk4_step(theta: float, omega: float, torque: float, params: PendulumParams) -> tuple[float, float]:
    dt = params.dt

    k1_t, k1_o = derivatives(theta, omega, torque, params)
    k2_t, k2_o = derivatives(theta + 0.5 * dt * k1_t, omega + 0.5 * dt * k1_o, torque, params)
    k3_t, k3_o = derivatives(theta + 0.5 * dt * k2_t, omega + 0.5 * dt * k2_o, torque, params)
    k4_t, k4_o = derivatives(theta + dt * k3_t, omega + dt * k3_o, torque, params)

    next_theta = theta + dt / 6.0 * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
    next_omega = omega + dt / 6.0 * (k1_o + 2.0 * k2_o + 2.0 * k3_o + k4_o)
    return wrap_angle(next_theta), next_omega


def grid_state(
    theta: float,
    omega: float,
    theta_grid: list[float],
    omega_grid: list[float],
) -> tuple[int, int]:
    return nearest_theta_index(theta, theta_grid), nearest_index(omega, omega_grid)
