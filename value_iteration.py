"""Vectorized value iteration for the torque-limited pendulum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from costs import CostParams, stage_cost_grid
from pendulum import GridSpec, PendulumParams


@dataclass(frozen=True)
class ValueIterationConfig:
    discount: float = 0.985
    tolerance: float = 1e-3
    max_iterations: int = 700
    actions: tuple[float, ...] = (-5.0, 0.0, 5.0)
    cost_name: str = "minimum_time"


@dataclass
class ValueIterationResult:
    theta_grid: np.ndarray
    omega_grid: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    policy: np.ndarray
    deltas: list[float]
    iterations: int
    cost_name: str


def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def rk4_step_grid(theta, omega, torque: float, params: PendulumParams):
    dt = params.dt

    def deriv(th, om):
        theta_dot = om
        omega_dot = (
            params.gravity / params.length * np.sin(th)
            + torque / (params.mass * params.length * params.length)
            - params.damping * om
        )
        return theta_dot, omega_dot

    k1_t, k1_o = deriv(theta, omega)
    k2_t, k2_o = deriv(theta + 0.5 * dt * k1_t, omega + 0.5 * dt * k1_o)
    k3_t, k3_o = deriv(theta + 0.5 * dt * k2_t, omega + 0.5 * dt * k2_o)
    k4_t, k4_o = deriv(theta + dt * k3_t, omega + dt * k3_o)

    next_theta = theta + dt / 6.0 * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)
    next_omega = omega + dt / 6.0 * (k1_o + 2.0 * k2_o + 2.0 * k3_o + k4_o)
    return wrap_angle(next_theta), next_omega


def make_grids(grid: GridSpec):
    theta_step = (grid.theta_max - grid.theta_min) / grid.n_theta
    theta_grid = grid.theta_min + theta_step * np.arange(grid.n_theta)
    omega_grid = np.linspace(grid.omega_min, grid.omega_max, grid.n_omega)
    theta_mesh, omega_mesh = np.meshgrid(theta_grid, omega_grid, indexing="ij")
    return theta_grid, omega_grid, theta_mesh, omega_mesh


def nearest_indices(theta, omega, theta_grid, omega_grid):
    theta_step = 2.0 * np.pi / len(theta_grid)
    omega_step = (omega_grid[-1] - omega_grid[0]) / (len(omega_grid) - 1)
    theta_idx = np.rint((wrap_angle(theta) - theta_grid[0]) / theta_step).astype(np.int64) % len(theta_grid)
    omega_idx = np.rint((np.clip(omega, omega_grid[0], omega_grid[-1]) - omega_grid[0]) / omega_step).astype(np.int64)
    omega_idx = np.clip(omega_idx, 0, len(omega_grid) - 1)
    return theta_idx, omega_idx


def build_model(theta_mesh, omega_mesh, theta_grid, omega_grid, actions, dynamics, cost_params, cost_name):
    next_i = []
    next_j = []
    costs = []
    for torque in actions:
        next_theta, next_omega = rk4_step_grid(theta_mesh, omega_mesh, float(torque), dynamics)
        i, j = nearest_indices(next_theta, next_omega, theta_grid, omega_grid)
        next_i.append(i)
        next_j.append(j)
        costs.append(stage_cost_grid(theta_mesh, omega_mesh, float(torque), cost_params, cost_name))

    return np.stack(next_i), np.stack(next_j), np.stack(costs)


def run_value_iteration(
    grid: GridSpec,
    dynamics: PendulumParams,
    cost_params: CostParams,
    config: ValueIterationConfig,
) -> ValueIterationResult:
    theta_grid, omega_grid, theta_mesh, omega_mesh = make_grids(grid)
    actions = np.clip(np.asarray(config.actions, dtype=float), -dynamics.torque_limit, dynamics.torque_limit)
    next_i, next_j, costs = build_model(
        theta_mesh,
        omega_mesh,
        theta_grid,
        omega_grid,
        actions,
        dynamics,
        cost_params,
        config.cost_name,
    )

    values = np.zeros(theta_mesh.shape, dtype=float)
    policy_index = np.zeros(theta_mesh.shape, dtype=np.int64)
    deltas: list[float] = []

    for iteration in range(1, config.max_iterations + 1):
        candidates = costs + config.discount * values[next_i, next_j]
        new_policy_index = np.argmin(candidates, axis=0)
        new_values = np.take_along_axis(candidates, new_policy_index[None, :, :], axis=0)[0]

        delta = float(np.max(np.abs(new_values - values)))
        values = new_values
        policy_index = new_policy_index
        deltas.append(delta)
        if delta < config.tolerance:
            break

    return ValueIterationResult(
        theta_grid=theta_grid,
        omega_grid=omega_grid,
        actions=actions,
        values=values,
        policy=actions[policy_index],
        deltas=deltas,
        iterations=len(deltas),
        cost_name=config.cost_name,
    )
