"""General-purpose coherence plotting utilities.

This module is intentionally generic so it can be reused for different
non-unitary evolution examples (e.g. ZZ dephasing and amplitude damping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

ArrayLike = Union[Sequence[float], Sequence[complex], np.ndarray]


@dataclass
class CoherenceSeries:
    """Container for coherence data evaluated on a time grid.

    Attributes
    ----------
    times:
        1D array of times.
    values:
        1D complex array of coherence values, typically rho[i, j](t).
    label:
        Legend label for the series.
    """

    times: np.ndarray
    values: np.ndarray
    label: str = "coherence"

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float).reshape(-1)
        self.values = np.asarray(self.values, dtype=complex).reshape(-1)
        if self.times.shape[0] != self.values.shape[0]:
            raise ValueError("times and values must have the same length.")


def extract_matrix_element_from_density_matrices(
    density_matrices: ArrayLike,
    i: int = 0,
    j: int = 1,
) -> np.ndarray:
    """Extract rho[i, j](t) from a time-ordered stack of density matrices.

    Parameters
    ----------
    density_matrices:
        Array with shape (T, d, d) or an iterable of d x d matrices.
    i, j:
        Matrix element to extract.
    """

    rhos = np.asarray(density_matrices, dtype=complex)
    if rhos.ndim != 3:
        raise ValueError("density_matrices must have shape (T, d, d).")
    return rhos[:, i, j]


def make_coherence_series_from_density_matrices(
    times: ArrayLike,
    density_matrices: ArrayLike,
    i: int = 0,
    j: int = 1,
    label: str = "coherence",
) -> CoherenceSeries:
    """Build a CoherenceSeries from density matrices."""

    values = extract_matrix_element_from_density_matrices(density_matrices, i=i, j=j)
    return CoherenceSeries(times=np.asarray(times, dtype=float), values=values, label=label)


def _coherence_component(values: np.ndarray, component: str) -> np.ndarray:
    component = component.lower()
    if component == "abs":
        return np.abs(values)
    if component == "real":
        return np.real(values)
    if component == "imag":
        return np.imag(values)
    if component == "phase":
        return np.angle(values)
    if component == "complex":
        return values
    raise ValueError("component must be one of {'abs', 'real', 'imag', 'phase', 'complex'}. ")


def plot_coherence_comparison(
    exact: CoherenceSeries,
    obtained: Optional[CoherenceSeries] = None,
    *,
    component: str = "abs",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "t",
    ylabel: Optional[str] = None,
    color: Optional[str] = None,
    exact_linestyle: str = "-",
    obtained_marker: str = "o",
    obtained_markevery: Optional[int] = None,
    obtained_alpha: float = 0.9,
    grid: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot exact coherence and obtained data using the same color.

    For `component='complex'`, two subplots are created for the real and
    imaginary parts.
    """

    if component.lower() == "complex":
        return plot_complex_coherence_comparison(
            exact,
            obtained,
            ax=None,
            title=title,
            xlabel=xlabel,
            color=color,
            obtained_marker=obtained_marker,
            obtained_markevery=obtained_markevery,
            obtained_alpha=obtained_alpha,
            grid=grid,
        )

    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
    else:
        fig = ax.figure

    exact_y = _coherence_component(exact.values, component)
    line, = ax.plot(
        exact.times,
        exact_y,
        linestyle=exact_linestyle,
        linewidth=2.0,
        color=color,
        label=f"Exact {exact.label}",
    )
    line_color = line.get_color()

    if obtained is not None:
        obtained_y = _coherence_component(obtained.values, component)
        ax.plot(
            obtained.times,
            obtained_y,
            linestyle="None",
            marker=obtained_marker,
            markersize=5,
            markevery=obtained_markevery,
            color=line_color,
            alpha=obtained_alpha,
            label=f"Obtained {obtained.label}",
        )

    if ylabel is None:
        ylabel = {
            "abs": r"$|\rho_{01}(t)|$",
            "real": r"$\Re\,\rho_{01}(t)$",
            "imag": r"$\Im\,\rho_{01}(t)$",
            "phase": r"$\arg\,\rho_{01}(t)$",
        }.get(component.lower(), "coherence")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_complex_coherence_comparison(
    exact: CoherenceSeries,
    obtained: Optional[CoherenceSeries] = None,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "t",
    colors: Optional[Sequence[str]] = None,
    obtained_marker: str = "o",
    obtained_markevery: Optional[int] = None,
    obtained_alpha: float = 0.9,
    grid: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot real and imaginary parts of the coherence on the same panel.

    Color convention:
    - colors=None:
        matplotlib default colors are used
    - colors=[c_real, c_imag]:
        exact and obtained real parts use c_real,
        exact and obtained imaginary parts use c_imag
    - colors=[c_exact_real, c_exact_imag, c_obt_real, c_obt_imag]:
        use fully explicit colors
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
    else:
        fig = ax.figure

    # -------- Resolve colors --------
    if colors is None:
        c_exact_real = None
        c_exact_imag = None
        c_obt_real = None
        c_obt_imag = None
    else:
        colors = list(colors)
        if len(colors) == 1:
            c_exact_real = colors[0]
            c_exact_imag = colors[0]
            c_obt_real = colors[0]
            c_obt_imag = colors[0]
        elif len(colors) == 2:
            c_exact_real = colors[0]
            c_exact_imag = colors[1]
            c_obt_real = colors[0]
            c_obt_imag = colors[1]
        elif len(colors) == 4:
            c_exact_real = colors[0]
            c_exact_imag = colors[1]
            c_obt_real = colors[2]
            c_obt_imag = colors[3]
        else:
            raise ValueError(
                "colors must have length 1, 2, or 4."
            )

    # -------- Exact curves --------
    line_real, = ax.plot(
        exact.times,
        np.real(exact.values),
        linewidth=2.0,
        color=c_exact_real,
        label=rf"Exact Re({exact.label})",
    )

    line_imag, = ax.plot(
        exact.times,
        np.imag(exact.values),
        linewidth=2.0,
        linestyle="--",
        color=c_exact_imag,
        label=rf"Exact Im({exact.label})",
    )

    # If colors were not explicitly given, use the exact-line colors for obtained data
    if c_obt_real is None:
        c_obt_real = line_real.get_color()
    if c_obt_imag is None:
        c_obt_imag = line_imag.get_color()

    # -------- Obtained data --------
    if obtained is not None:
        ax.plot(
            obtained.times,
            np.real(obtained.values),
            linestyle="None",
            marker=obtained_marker,
            markersize=5,
            markevery=obtained_markevery,
            alpha=obtained_alpha,
            color=c_obt_real,
            label=rf"Obtained Re({obtained.label})",
        )
        ax.plot(
            obtained.times,
            np.imag(obtained.values),
            linestyle="None",
            marker=obtained_marker,
            markersize=5,
            markevery=obtained_markevery,
            alpha=obtained_alpha,
            color=c_obt_imag,
            label=rf"Obtained Im({obtained.label})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\rho_{01}(t)$")
    if grid:
        ax.grid(True, alpha=0.3)
    ax.legend()
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax




def extract_population_from_density_matrices(
    density_matrices: ArrayLike,
    level: int = 0,
) -> np.ndarray:
    """Extract rho[level, level](t) from a time-ordered stack of density matrices."""

    rhos = np.asarray(density_matrices, dtype=complex)
    if rhos.ndim != 3:
        raise ValueError("density_matrices must have shape (T, d, d).")
    return np.real_if_close(rhos[:, level, level])


def plot_populations_and_coherence_comparison(
    times: ArrayLike,
    exact_density_matrices: ArrayLike,
    obtained_density_matrices: Optional[ArrayLike] = None,
    *,
    coherence_component: str = "real",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "t",
    ylabel: str = "value",
    colors: Optional[Sequence[str]] = None,
    obtained_marker: str = "o",
    obtained_markevery: Optional[int] = None,
    obtained_alpha: float = 0.9,
    grid: bool = True,
    ground_label: str = r"Ground population $\rho_{00}(t)$",
    excited_label: str = r"Excited population $\rho_{11}(t)$",
    coherence_label: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ground/excited populations and one coherence component on one axis.

    Parameters
    ----------
    times:
        1D time grid.
    exact_density_matrices:
        Array of exact density matrices with shape (T, 2, 2).
    obtained_density_matrices:
        Optional obtained/reconstructed density matrices with shape (T, 2, 2).
    coherence_component:
        Which component of rho[0, 1](t) to plot. Supported values are
        {'abs', 'real', 'imag', 'phase'}.
    colors:
        Either None or a length-3 sequence giving the colors for
        [ground, excited, coherence]. Exact and obtained data for the same
        quantity use the same color.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    exact_rhos = np.asarray(exact_density_matrices, dtype=complex)
    if exact_rhos.ndim != 3:
        raise ValueError("exact_density_matrices must have shape (T, d, d).")
    if exact_rhos.shape[0] != times.shape[0]:
        raise ValueError("times and exact_density_matrices must have the same length.")

    if obtained_density_matrices is not None:
        obtained_rhos = np.asarray(obtained_density_matrices, dtype=complex)
        if obtained_rhos.ndim != 3:
            raise ValueError("obtained_density_matrices must have shape (T, d, d).")
        if obtained_rhos.shape[0] != times.shape[0]:
            raise ValueError("times and obtained_density_matrices must have the same length.")
    else:
        obtained_rhos = None

    component = coherence_component.lower()
    if component == "complex":
        raise ValueError(
            "coherence_component='complex' is not supported here; choose one of "
            "{'abs', 'real', 'imag', 'phase'}."
        )

    if coherence_label is None:
        coherence_label = {
            "abs": r"Coherence $|\rho_{01}(t)|$",
            "real": r"Coherence $\Re\,\rho_{01}(t)$",
            "imag": r"Coherence $\Im\,\rho_{01}(t)$",
            "phase": r"Coherence $\arg\,\rho_{01}(t)$",
        }.get(component, "Coherence")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
    else:
        fig = ax.figure

    if colors is not None:
        colors = list(colors)
        if len(colors) != 3:
            raise ValueError("colors must be None or a length-3 sequence: [ground, excited, coherence].")
        ground_color, excited_color, coherence_color = colors
    else:
        ground_color = excited_color = coherence_color = None

    exact_ground = extract_population_from_density_matrices(exact_rhos, level=0)
    exact_excited = extract_population_from_density_matrices(exact_rhos, level=1)
    exact_coherence = _coherence_component(extract_matrix_element_from_density_matrices(exact_rhos, 0, 1), component)

    line_ground, = ax.plot(
        times,
        exact_ground,
        linewidth=2.0,
        color=ground_color,
        label=f"Exact {ground_label}",
    )
    line_excited, = ax.plot(
        times,
        exact_excited,
        linewidth=2.0,
        color=excited_color,
        label=f"Exact {excited_label}",
    )
    line_coherence, = ax.plot(
        times,
        exact_coherence,
        linewidth=2.0,
        color=coherence_color,
        label=f"Exact {coherence_label}",
    )

    if obtained_rhos is not None:
        obtained_ground = extract_population_from_density_matrices(obtained_rhos, level=0)
        obtained_excited = extract_population_from_density_matrices(obtained_rhos, level=1)
        obtained_coherence = _coherence_component(
            extract_matrix_element_from_density_matrices(obtained_rhos, 0, 1),
            component,
        )

        ax.plot(
            times,
            obtained_ground,
            linestyle="None",
            marker=obtained_marker,
            markersize=5,
            markevery=obtained_markevery,
            alpha=obtained_alpha,
            color=line_ground.get_color(),
            label=f"Obtained {ground_label}",
        )
        ax.plot(
            times,
            obtained_excited,
            linestyle="None",
            marker=obtained_marker,
            markersize=5,
            markevery=obtained_markevery,
            alpha=obtained_alpha,
            color=line_excited.get_color(),
            label=f"Obtained {excited_label}",
        )
        ax.plot(
            times,
            obtained_coherence,
            linestyle="None",
            marker=obtained_marker,
            markersize=5,
            markevery=obtained_markevery,
            alpha=obtained_alpha,
            color=line_coherence.get_color(),
            label=f"Obtained {coherence_label}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)
    fig.tight_layout()
    return fig, ax