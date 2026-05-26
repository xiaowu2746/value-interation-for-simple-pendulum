"""Matplotlib exporters for plots, CSV files, and pendulum GIF animations."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_value_function(path: str | Path, theta_grid, omega_grid, values, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[omega_grid[0], omega_grid[-1], theta_grid[0], theta_grid[-1]],
        cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_ylabel(r"$\theta$ (rad)")
    fig.colorbar(image, ax=ax, label="cost-to-go")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_policy(path: str | Path, theta_grid, omega_grid, policy, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    image = ax.imshow(
        policy,
        origin="lower",
        aspect="auto",
        extent=[omega_grid[0], omega_grid[-1], theta_grid[0], theta_grid[-1]],
        cmap="coolwarm",
        vmin=float(np.min(policy)),
        vmax=float(np.max(policy)),
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_ylabel(r"$\theta$ (rad)")
    fig.colorbar(image, ax=ax, label="torque (Nm)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_state_plot(path: str | Path, trajectory: list[dict[str, float]], title: str) -> None:
    t = [row["time"] for row in trajectory]
    theta_error = [row["theta_error"] for row in trajectory]
    omega = [row["omega"] for row in trajectory]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    ax.plot(t, theta_error, label=r"$\theta$ error")
    ax.plot(t, omega, label=r"$\dot{\theta}$")
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("state")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_torque_plot(path: str | Path, trajectory: list[dict[str, float]], title: str) -> None:
    t = [row["time"] for row in trajectory]
    torque = [row["torque"] for row in trajectory]

    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    ax.step(t, torque, where="post")
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("torque (Nm)")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_convergence_plot(path: str | Path, deltas: list[float], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.semilogy(range(1, len(deltas) + 1), deltas)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("max Bellman update")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_animation_gif(
    path: str | Path,
    trajectory: list[dict[str, float]],
    title: str,
    fps: int = 24,
    stride: int = 2,
) -> None:
    sampled = trajectory[::stride]
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    rod, = ax.plot([], [], color="#1f77b4", linewidth=4)
    bob, = ax.plot([], [], "o", color="#d62728", markersize=16)
    pivot, = ax.plot([0.0], [0.0], "ko", markersize=5)
    text = ax.text(0.03, 0.95, "", transform=ax.transAxes, va="top")

    def update(frame):
        row = sampled[frame]
        theta = row["theta"]
        x = np.sin(theta)
        y = np.cos(theta)
        rod.set_data([0.0, x], [0.0, y])
        bob.set_data([x], [y])
        text.set_text(
            f"t = {row['time']:.2f} s\n"
            f"theta err = {row['theta_error']:.2f} rad\n"
            f"u = {row['torque']:.1f} Nm"
        )
        return rod, bob, pivot, text

    animation = FuncAnimation(fig, update, frames=len(sampled), interval=1000 / fps, blit=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
