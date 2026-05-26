"""Running costs and terminal tests for pendulum swing-up."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendulum import angle_error


@dataclass(frozen=True)
class CostParams:
    theta_tolerance: float = 0.12
    omega_tolerance: float = 0.35
    q_theta: float = 8.0
    q_omega: float = 0.25
    r_torque: float = 0.02


def is_target(theta: float, omega: float, params: CostParams) -> bool:
    return abs(angle_error(theta)) <= params.theta_tolerance and abs(omega) <= params.omega_tolerance


def minimum_time_cost(theta: float, omega: float, torque: float, params: CostParams) -> float:
    if is_target(theta, omega, params):
        return 0.0
    return 1.0


def quadratic_cost(theta: float, omega: float, torque: float, params: CostParams) -> float:
    if is_target(theta, omega, params):
        return 0.0
    err = angle_error(theta)
    return params.q_theta * err * err + params.q_omega * omega * omega + params.r_torque * torque * torque


def target_mask(theta, omega, params: CostParams):
    return (np.abs(theta) <= params.theta_tolerance) & (np.abs(omega) <= params.omega_tolerance)


def stage_cost_grid(theta, omega, torque: float, params: CostParams, name: str):
    if name == "minimum_time":
        return np.where(target_mask(theta, omega, params), 0.0, 1.0)
    if name == "quadratic":
        cost = params.q_theta * theta**2 + params.q_omega * omega**2 + params.r_torque * torque**2
        return np.where(target_mask(theta, omega, params), 0.0, cost)
    raise ValueError(f"unknown cost function: {name}")


def get_cost_function(name: str):
    if name == "minimum_time":
        return minimum_time_cost
    if name == "quadratic":
        return quadratic_cost
    raise ValueError(f"unknown cost function: {name}")
