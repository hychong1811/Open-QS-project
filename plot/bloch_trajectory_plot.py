"""Bloch-sphere plotting utilities for one-qubit trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ArrayLike = Union[Sequence[float], np.ndarray]

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass
class BlochTrajectory:
    times: np.ndarray
    vectors: np.ndarray
    label: str = "trajectory"

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float).reshape(-1)
        self.vectors = np.asarray(self.vectors, dtype=float)
        if self.vectors.ndim != 2 or self.vectors.shape[1] != 3:
            raise ValueError("vectors must have shape (T, 3).")
        if self.vectors.shape[0] != self.times.shape[0]:
            raise ValueError("times and vectors must have the same leading dimension.")


def density_matrix_to_bloch_vector(rho: np.ndarray, *, normalise_trace: bool = True) -> np.ndarray:
    """Convert a 2x2 density matrix to a Bloch vector.

    For subnormalised states, `normalise_trace=True` first divides by the trace.
    """

    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must be a 2x2 matrix.")

    trace_val = np.trace(rho)
    if normalise_trace:
        if abs(trace_val) < 1e-14:
            raise ValueError("Cannot form Bloch vector from a zero-trace matrix.")
        rho = rho / trace_val

    return np.array([
        np.real(np.trace(rho @ SIGMA_X)),
        np.real(np.trace(rho @ SIGMA_Y)),
        np.real(np.trace(rho @ SIGMA_Z)),
    ], dtype=float)


def density_matrices_to_bloch_vectors(
    density_matrices: np.ndarray,
    *,
    normalise_trace: bool = True,
) -> np.ndarray:
    """Convert a stack of density matrices with shape (T,2,2) to Bloch vectors."""

    rhos = np.asarray(density_matrices, dtype=complex)
    if rhos.ndim != 3 or rhos.shape[1:] != (2, 2):
        raise ValueError("density_matrices must have shape (T, 2, 2).")
    return np.array([
        density_matrix_to_bloch_vector(rho, normalise_trace=normalise_trace) for rho in rhos
    ], dtype=float)


def make_bloch_trajectory_from_density_matrices(
    times: ArrayLike,
    density_matrices: np.ndarray,
    *,
    normalise_trace: bool = True,
    label: str = "trajectory",
) -> BlochTrajectory:
    vectors = density_matrices_to_bloch_vectors(
        density_matrices,
        normalise_trace=normalise_trace,
    )
    return BlochTrajectory(times=np.asarray(times, dtype=float), vectors=vectors, label=label)


def _draw_bloch_sphere(ax: plt.Axes, *, sphere_alpha: float = 0.09, wire_alpha: float = 0.16) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=sphere_alpha, linewidth=0.0, color="lightgray", shade=False)
    ax.plot_wireframe(x, y, z, rstride=6, cstride=6, alpha=wire_alpha, color="gray")

    ax.plot([-1, 1], [0, 0], [0, 0], color="black", alpha=0.35, linewidth=1.0)
    ax.plot([0, 0], [-1, 1], [0, 0], color="black", alpha=0.35, linewidth=1.0)
    ax.plot([0, 0], [0, 0], [-1, 1], color="black", alpha=0.35, linewidth=1.0)

    ax.text(1.08, 0, 0, "x")
    ax.text(0, 1.08, 0, "y")
    ax.text(0, 0, 1.08, "z")

    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_zlim([-1.05, 1.05])
    ax.set_box_aspect((1, 1, 1))


def plot_bloch_trajectory_comparison(
    exact: BlochTrajectory,
    obtained: Optional[BlochTrajectory] = None,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_sphere: bool = True,
    show_start_end: bool = True,
    obtained_marker: str = "o",
    obtained_alpha: float = 0.9,
    view_elev: float = 22.0,
    view_azim: float = -58.0,
    cmap: str = "viridis",
    point_size: float = 28.0,
    obtained_line: bool = False,
    obtained_line_alpha: float = 0.25,
    show_colorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Bloch trajectories with:
    - exact: black dotted curve, no gradient points
    - obtained: time-gradient points
    """

    if ax is None:
        fig = plt.figure(figsize=(7.0, 6.4))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if show_sphere:
        _draw_bloch_sphere(ax)

    # -------- Resolve time arrays --------
    exact_times = getattr(exact, "times", None)
    if exact_times is None:
        exact_times = np.arange(exact.vectors.shape[0], dtype=float)
    else:
        exact_times = np.asarray(exact_times, dtype=float)

    if exact.vectors.shape[0] != exact_times.shape[0]:
        raise ValueError("exact.vectors and exact.times must have matching length.")

    obtained_times = None
    if obtained is not None:
        obtained_times = getattr(obtained, "times", None)
        if obtained_times is None:
            obtained_times = np.arange(obtained.vectors.shape[0], dtype=float)
        else:
            obtained_times = np.asarray(obtained_times, dtype=float)

        if obtained.vectors.shape[0] != obtained_times.shape[0]:
            raise ValueError("obtained.vectors and obtained.times must have matching length.")

    # -------- Shared normalization for obtained gradient --------
    if obtained_times is not None:
        tmin = np.min(obtained_times)
        tmax = np.max(obtained_times)
        if np.isclose(tmin, tmax):
            tmax = tmin + 1.0
        norm = Normalize(vmin=tmin, vmax=tmax)
    else:
        norm = None

    # -------- Exact: black dotted line only --------
    ax.plot(
        exact.vectors[:, 0],
        exact.vectors[:, 1],
        exact.vectors[:, 2],
        linestyle=":",
        linewidth=2.0,
        color="black",
        label=f"Exact {exact.label}",
    )

    # -------- Obtained: gradient points --------
    if obtained is not None:
        if obtained_line:
            ax.plot(
                obtained.vectors[:, 0],
                obtained.vectors[:, 1],
                obtained.vectors[:, 2],
                linewidth=1.2,
                alpha=obtained_line_alpha,
                color="gray",
                linestyle="-",
            )

        ax.scatter(
            obtained.vectors[:, 0],
            obtained.vectors[:, 1],
            obtained.vectors[:, 2],
            c=obtained_times,
            cmap=cmap,
            norm=norm,
            s=point_size,
            marker=obtained_marker,
            alpha=obtained_alpha,
            depthshade=False,
            label=f"Obtained {obtained.label}",
        )

    # -------- Start / end markers from obtained if available, else exact --------
    if show_start_end:
        ref_vectors = obtained.vectors if obtained is not None else exact.vectors
        ref_times = obtained_times if obtained is not None else exact_times

        if ref_vectors.shape[0] > 0:
            start = ref_vectors[0]
            end = ref_vectors[-1]

            if obtained is not None:
                start_color = plt.get_cmap(cmap)(norm(ref_times[0]))
                end_color = plt.get_cmap(cmap)(norm(ref_times[-1]))
            else:
                start_color = "black"
                end_color = "black"

            ax.scatter(
                [start[0]], [start[1]], [start[2]],
                color=[start_color],
                s=55,
                edgecolors="black",
                linewidths=0.6,
                label="Start",
            )
            ax.scatter(
                [end[0]], [end[1]], [end[2]],
                color=[end_color],
                s=75,
                marker="^",
                edgecolors="black",
                linewidths=0.6,
                label="End",
            )

    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if title is not None:
        ax.set_title(title)

    if show_colorbar and obtained is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.75)
        cbar.set_label("Time")

    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax