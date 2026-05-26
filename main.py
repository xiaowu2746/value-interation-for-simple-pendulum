"""Run value iteration experiments for the torque-limited pendulum."""

from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path

from costs import CostParams
from pendulum import GridSpec, PendulumParams, angle_error, clamp, grid_state, rk4_step
from plotting import (
    ensure_dir,
    save_animation_gif,
    save_convergence_plot,
    save_policy,
    save_state_plot,
    save_torque_plot,
    save_value_function,
    write_csv,
)
from value_iteration import ValueIterationConfig, ValueIterationResult, run_value_iteration


def policy_action(theta: float, omega: float, result: ValueIterationResult) -> float:
    i, j = grid_state(theta, omega, result.theta_grid, result.omega_grid)
    return result.policy[i][j]


def simulate_policy(
    result: ValueIterationResult,
    dynamics: PendulumParams,
    theta0: float,
    omega0: float,
    duration: float,
) -> list[dict[str, float]]:
    steps = int(duration / dynamics.dt)
    theta, omega = theta0, omega0
    trajectory: list[dict[str, float]] = []

    for k in range(steps + 1):
        torque = policy_action(theta, omega, result)
        trajectory.append(
            {
                "time": k * dynamics.dt,
                "theta": theta,
                "theta_error": angle_error(theta),
                "omega": omega,
                "torque": torque,
            }
        )
        theta, omega = rk4_step(theta, omega, torque, dynamics)
        omega = clamp(omega, result.omega_grid[0], result.omega_grid[-1])

    return trajectory


def export_result(
    result: ValueIterationResult,
    trajectory: list[dict[str, float]],
    out_dir: Path,
) -> dict[str, float | str | int]:
    prefix = result.cost_name
    write_csv(out_dir / f"{prefix}_trajectory.csv", trajectory)
    save_value_function(
        out_dir / f"{prefix}_value_function.png",
        result.theta_grid,
        result.omega_grid,
        result.values,
        f"{prefix}: value function",
    )
    save_policy(
        out_dir / f"{prefix}_policy.png",
        result.theta_grid,
        result.omega_grid,
        result.policy,
        f"{prefix}: policy",
    )
    save_state_plot(out_dir / f"{prefix}_states.png", trajectory, f"{prefix}: closed-loop states")
    save_torque_plot(out_dir / f"{prefix}_torque.png", trajectory, f"{prefix}: control input")
    save_convergence_plot(out_dir / f"{prefix}_convergence.png", result.deltas, f"{prefix}: convergence")
    save_animation_gif(out_dir / f"{prefix}_animation.gif", trajectory, f"{prefix}: pendulum swing-up")

    final = trajectory[-1]
    total_abs_torque = sum(abs(row["torque"]) for row in trajectory) * (trajectory[1]["time"] - trajectory[0]["time"])
    settling_time = None
    for row in trajectory:
        if abs(row["theta_error"]) < 0.15 and abs(row["omega"]) < 0.45:
            settling_time = row["time"]
            break

    return {
        "cost": result.cost_name,
        "iterations": result.iterations,
        "final_delta": result.deltas[-1],
        "final_theta_error": final["theta_error"],
        "final_omega": final["omega"],
        "integral_abs_torque": total_abs_torque,
        "first_entered_target_neighborhood_s": settling_time if settling_time is not None else "not reached",
    }


def write_summary(path: Path, summaries: list[dict[str, float | str | int]]) -> None:
    lines = ["# Value Iteration Experiment Summary", ""]
    for item in summaries:
        lines.append(f"## {item['cost']}")
        lines.append("")
        lines.append(f"- Iterations: {item['iterations']}")
        lines.append(f"- Final Bellman update: {float(item['final_delta']):.6f}")
        lines.append(f"- Final theta error: {float(item['final_theta_error']):.4f} rad")
        lines.append(f"- Final theta_dot: {float(item['final_omega']):.4f} rad/s")
        lines.append(f"- Integral absolute torque: {float(item['integral_abs_torque']):.4f} Nms")
        lines.append(f"- First entered target neighborhood: {item['first_entered_target_neighborhood_s']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-theta", type=int, default=81, help="number of angle grid points")
    parser.add_argument("--n-omega", type=int, default=81, help="number of angular velocity grid points")
    parser.add_argument("--omega-max", type=float, default=8.0, help="absolute angular velocity grid bound")
    parser.add_argument("--u-max", type=float, default=5.0, help="torque limit")
    parser.add_argument("--dt", type=float, default=0.05, help="time step")
    parser.add_argument("--duration", type=float, default=12.0, help="closed-loop simulation duration")
    parser.add_argument("--max-iterations", type=int, default=700, help="maximum VI iterations")
    parser.add_argument("--out", type=str, default="results", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out)

    dynamics = PendulumParams(dt=args.dt, torque_limit=args.u_max)
    grid = GridSpec(
        omega_min=-args.omega_max,
        omega_max=args.omega_max,
        n_theta=args.n_theta,
        n_omega=args.n_omega,
    )
    cost_params = CostParams()
    actions = (-args.u_max, 0.0, args.u_max)

    experiment_summaries = []
    for cost_name in ("minimum_time", "quadratic"):
        config = ValueIterationConfig(
            actions=actions,
            cost_name=cost_name,
            max_iterations=args.max_iterations,
        )
        print(f"Running value iteration for {cost_name}...")
        result = run_value_iteration(grid, dynamics, cost_params, config)
        trajectory = simulate_policy(
            result,
            dynamics,
            theta0=pi - 0.05,
            omega0=0.0,
            duration=args.duration,
        )
        experiment_summaries.append(export_result(result, trajectory, out_dir))
        print(
            f"  iterations={result.iterations}, "
            f"final_delta={result.deltas[-1]:.6f}, "
            f"final_error={trajectory[-1]['theta_error']:.4f}"
        )

    write_summary(out_dir / "summary.md", experiment_summaries)
    (out_dir / "summary.json").write_text(
        json.dumps(experiment_summaries, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Results written to {out_dir}")


if __name__ == "__main__":
    main()
