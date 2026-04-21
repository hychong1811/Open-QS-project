
"""
Utilities for plotting noise-resilience sweep results.

This module is intentionally general so it can be reused by:
- subnorm noise-resilience notebooks
- timeevo/ZZ noise-resilience notebooks
- timeevo/ampdamp noise-resilience notebooks

The plotting helpers accept either plain arrays or pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ArrayLike = Sequence[float] | np.ndarray


@dataclass
class SweepPlotColumns:
    """Column names for a standard noise-sweep summary DataFrame."""

    sweep_value: str = "sweep_value"
    fidelity_mean: str = "fidelity_mean"
    fidelity_std: str = "fidelity_std"
    success_mean: str = "success_probability_mean"
    success_std: str = "success_probability_std"
    label: Optional[str] = None
    method_column="method"


def default_sweep_xlabel(
    noise_kind: Optional[str] = None,
    parameter_name: Optional[str] = None,
) -> str:
    """Return a sensible x-axis label for a noise sweep."""
    if parameter_name is not None:
        return str(parameter_name)

    if noise_kind is None:
        return "Sweep parameter"

    noise_kind = str(noise_kind).strip().lower()
    mapping = {
        "amplitude_damping": "Amplitude-damping strength",
        "phase_damping": "Phase-damping strength",
        "depolarizing": "Depolarizing probability",
        "readout": "Readout assignment error",
        "t1": r"$T_1$",
        "t2": r"$T_2$",
    }
    return mapping.get(noise_kind, f"{noise_kind} sweep parameter")


def _to_numpy_1d(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def _apply_grid(ax: plt.Axes, grid: bool) -> None:
    if grid:
        ax.grid(True, alpha=0.3)


def plot_metric_vs_sweep(
    sweep_values: ArrayLike,
    metric_values: ArrayLike,
    *,
    metric_std: Optional[ArrayLike] = None,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Sweep parameter",
    ylabel: str = "Metric",
    color: Optional[str] = None,
    marker: str = "o",
    linewidth: float = 2.0,
    alpha: float = 0.95,
    grid: bool = True,
    show_errorband: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot one metric against a swept parameter."""
    x = _to_numpy_1d(sweep_values, "sweep_values")
    y = _to_numpy_1d(metric_values, "metric_values")

    if x.shape[0] != y.shape[0]:
        raise ValueError("sweep_values and metric_values must have matching length.")

    yerr = None
    if metric_std is not None:
        yerr = _to_numpy_1d(metric_std, "metric_std")
        if yerr.shape[0] != x.shape[0]:
            raise ValueError("metric_std must have the same length as sweep_values.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
    else:
        fig = ax.figure

    line, = ax.plot(
        x,
        y,
        marker=marker,
        linewidth=linewidth,
        alpha=alpha,
        color=color,
        label=label,
    )
    line_color = line.get_color()

    if yerr is not None and show_errorband:
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=line_color,
            alpha=0.18,
            linewidth=0.0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    _apply_grid(ax, grid)
    if label is not None:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
    fig.tight_layout()
    return fig, ax


def plot_fidelity_and_success_vs_sweep(
    sweep_values: ArrayLike,
    fidelity_values: ArrayLike,
    success_values: ArrayLike,
    *,
    fidelity_std: Optional[ArrayLike] = None,
    success_std: Optional[ArrayLike] = None,
    axes: Optional[Sequence[plt.Axes]] = None,
    title: Optional[str] = None,
    xlabel: str = "Sweep parameter",
    fidelity_label: str = "Fidelity",
    success_label: str = "Success probability",
    colors: Optional[Sequence[str]] = None,
    markers: Sequence[str] = ("o", "s"),
    grid: bool = True,
    sharex: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot fidelity and success probability against the same sweep parameter."""
    x = _to_numpy_1d(sweep_values, "sweep_values")
    fidelity = _to_numpy_1d(fidelity_values, "fidelity_values")
    success = _to_numpy_1d(success_values, "success_values")

    if x.shape[0] != fidelity.shape[0] or x.shape[0] != success.shape[0]:
        raise ValueError("All input arrays must have matching length.")

    if axes is None:
        fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=sharex)
    else:
        axs = np.asarray(axes).ravel()
        if axs.size != 2:
            raise ValueError("axes must contain exactly two matplotlib axes.")
        fig = axs[0].figure

    if colors is None:
        fidelity_color = None
        success_color = None
    else:
        if len(colors) != 2:
            raise ValueError("colors must contain exactly two entries.")
        fidelity_color, success_color = colors

    plot_metric_vs_sweep(
        x,
        fidelity,
        metric_std=fidelity_std,
        ax=axs[0],
        xlabel=xlabel,
        ylabel="Fidelity",
        label=fidelity_label,
        color=fidelity_color,
        marker=markers[0],
        grid=grid,
    )
    plot_metric_vs_sweep(
        x,
        success,
        metric_std=success_std,
        ax=axs[1],
        xlabel=xlabel,
        ylabel="Success probability",
        label=success_label,
        color=success_color,
        marker=markers[1],
        grid=grid,
    )

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    return fig, axs


def plot_results_dataframe(
    dataframe,
    *,
    columns: SweepPlotColumns = SweepPlotColumns(),
    axes: Optional[Sequence[plt.Axes]] = None,
    title: Optional[str] = None,
    xlabel: str = "Sweep parameter",
    colors: Optional[Sequence[str]] = None,
    markers: Sequence[str] = ("o", "s"),
    grid: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot a standard results DataFrame with fidelity and success-probability columns."""
    if pd is None:
        raise ImportError("pandas is required for plot_results_dataframe.")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    required = [columns.sweep_value, columns.fidelity_mean, columns.success_mean]
    missing = [col for col in required if col not in dataframe.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    fidelity_std = dataframe[columns.fidelity_std].to_numpy() if columns.fidelity_std in dataframe.columns else None
    success_std = dataframe[columns.success_std].to_numpy() if columns.success_std in dataframe.columns else None

    figure_title = title
    if figure_title is None and columns.label is not None and columns.label in dataframe.columns:
        unique_labels = dataframe[columns.label].astype(str).unique()
        if len(unique_labels) == 1:
            figure_title = unique_labels[0]

    return plot_fidelity_and_success_vs_sweep(
        dataframe[columns.sweep_value].to_numpy(),
        dataframe[columns.fidelity_mean].to_numpy(),
        dataframe[columns.success_mean].to_numpy(),
        fidelity_std=fidelity_std,
        success_std=success_std,
        axes=axes,
        title=figure_title,
        xlabel=xlabel,
        colors=colors,
        markers=markers,
        grid=grid,
    )


def plot_grouped_results_dataframe(
    dataframe,
    *,
    group_column: str,
    columns: SweepPlotColumns = SweepPlotColumns(),
    axes: Optional[Sequence[plt.Axes]] = None,
    title: Optional[str] = None,
    xlabel: str = "Sweep parameter",
    colors: Optional[Sequence[str]] = None,
    markers: Sequence[str] = ("o", "s"),
    grid: bool = True,
    legend_title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot grouped sweep results, overlaying multiple curves on the same two axes."""
    if pd is None:
        raise ImportError("pandas is required for plot_grouped_results_dataframe.")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if group_column not in dataframe.columns:
        raise ValueError(f"group_column '{group_column}' not found in DataFrame.")

    required = [columns.sweep_value, columns.fidelity_mean, columns.success_mean]
    missing = [col for col in required if col not in dataframe.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if axes is None:
        fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=True)
    else:
        axs = np.asarray(axes).ravel()
        if axs.size != 2:
            raise ValueError("axes must contain exactly two matplotlib axes.")
        fig = axs[0].figure

    groups = list(dataframe[group_column].astype(str).unique())
    color_cycle = None if colors is None else list(colors)

    for idx, group in enumerate(groups):
        subdf = dataframe[dataframe[group_column].astype(str) == group].sort_values(columns.sweep_value)
        color = None if color_cycle is None else color_cycle[idx % len(color_cycle)]

        fidelity_std = subdf[columns.fidelity_std].to_numpy() if columns.fidelity_std in subdf.columns else None
        success_std = subdf[columns.success_std].to_numpy() if columns.success_std in subdf.columns else None

        plot_metric_vs_sweep(
            subdf[columns.sweep_value].to_numpy(),
            subdf[columns.fidelity_mean].to_numpy(),
            metric_std=fidelity_std,
            ax=axs[0],
            xlabel=xlabel,
            ylabel="Fidelity",
            label=group,
            color=color,
            marker=markers[0],
            grid=grid,
        )
        plot_metric_vs_sweep(
            subdf[columns.sweep_value].to_numpy(),
            subdf[columns.success_mean].to_numpy(),
            metric_std=success_std,
            ax=axs[1],
            xlabel=xlabel,
            ylabel="Success probability",
            label=group,
            color=color,
            marker=markers[1],
            grid=grid,
        )

    if legend_title is not None:
        axs[0].legend(title=legend_title)
        axs[1].legend(title=legend_title)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    return fig, axs

def _plot_metric_vs_sweep_group_safe(
    ax: plt.Axes,
    sweep_values: ArrayLike,
    metric_values: ArrayLike,
    *,
    metric_std: Optional[ArrayLike] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: str = "o",
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 0.95,
    grid: bool = True,
    show_errorband: bool = True,
) -> None:
    """
    Plot one grouped curve while safely ignoring NaN values.

    This is useful when one method does not define a metric
    at all sweep points, for example success probability for
    a unitary-only baseline.
    """
    x = _to_numpy_1d(sweep_values, "sweep_values")
    y = _to_numpy_1d(metric_values, "metric_values")

    if x.shape[0] != y.shape[0]:
        raise ValueError("sweep_values and metric_values must have matching length.")

    valid = np.isfinite(x) & np.isfinite(y)

    yerr = None
    if metric_std is not None:
        yerr = _to_numpy_1d(metric_std, "metric_std")
        if yerr.shape[0] != x.shape[0]:
            raise ValueError("metric_std must have the same length as sweep_values.")
        valid = valid & np.isfinite(yerr)

    if not np.any(valid):
        return

    x_plot = x[valid]
    y_plot = y[valid]

    line, = ax.plot(
        x_plot,
        y_plot,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        color=color,
        label=label,
    )
    line_color = line.get_color()

    if yerr is not None and show_errorband:
        yerr_plot = yerr[valid]
        ax.fill_between(
            x_plot,
            y_plot - yerr_plot,
            y_plot + yerr_plot,
            color=line_color,
            alpha=0.18,
            linewidth=0.0,
        )

    _apply_grid(ax, grid)

def plot_method_comparison_dataframe(
    dataframe,
    *,
    method_column: str = "method",
    columns: SweepPlotColumns = SweepPlotColumns(),
    axes: Optional[Sequence[plt.Axes]] = None,
    title: Optional[str] = None,
    xlabel: str = "Sweep parameter",
    colors: Optional[Sequence[str]] = None,
    markers: Optional[Sequence[str]] = None,
    line_styles: Optional[dict[str, str]] = None,
    grid: bool = True,
    legend_title: Optional[str] = "Method",
    show_errorband: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot fidelity and success probability versus sweep parameter,
    overlaying multiple methods on the same figure.

    Expected DataFrame columns by default:
    - method
    - sweep_value
    - fidelity_mean
    - fidelity_std        (optional)
    - success_probability_mean
    - success_probability_std  (optional)

    Notes
    -----
    - Methods with NaN success probability values are skipped only on the
      success-probability panel, but still appear on the fidelity panel.
    - This is useful for comparing:
        * paper SVD dilation
        * Sz.-Nagy dilation
        * unitary-only rescaled baseline
    """
    if pd is None:
        raise ImportError("pandas is required for plot_method_comparison_dataframe.")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    required = [method_column, columns.sweep_value, columns.fidelity_mean]
    missing = [col for col in required if col not in dataframe.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if columns.success_mean not in dataframe.columns:
        raise ValueError(
            f"DataFrame must contain success metric column '{columns.success_mean}'."
        )

    if axes is None:
        fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.6), sharex=True)
    else:
        axs = np.asarray(axes).ravel()
        if axs.size != 2:
            raise ValueError("axes must contain exactly two matplotlib axes.")
        fig = axs[0].figure

    methods = list(dataframe[method_column].astype(str).unique())

    if colors is None:
        color_cycle = [None] * len(methods)
    else:
        color_cycle = list(colors)
        if len(color_cycle) < len(methods):
            color_cycle = [color_cycle[i % len(color_cycle)] for i in range(len(methods))]

    if markers is None:
        base_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        marker_cycle = [base_markers[i % len(base_markers)] for i in range(len(methods))]
    else:
        marker_cycle = list(markers)
        if len(marker_cycle) < len(methods):
            marker_cycle = [marker_cycle[i % len(marker_cycle)] for i in range(len(methods))]

    # Default line-style convention for method comparison
    default_line_styles = {
        "paper_svd": "-",
        "paper": "-",
        "svd": "-",
        "sz_nagy": "--",
        "sz-nagy": "--",
        "sznagy": "--",
        "unitary_rescaled": "-.",
        "unitary_only": "-.",
        "unitary-only": "-.",
        "unitary": "-.",
    }

    if line_styles is None:
        line_styles = {}

    for idx, method in enumerate(methods):
        subdf = dataframe[dataframe[method_column].astype(str) == method].sort_values(columns.sweep_value)

        fidelity_std = (
            subdf[columns.fidelity_std].to_numpy()
            if columns.fidelity_std in subdf.columns else None
        )
        success_std = (
            subdf[columns.success_std].to_numpy()
            if columns.success_std in subdf.columns else None
        )

        color = color_cycle[idx]
        marker = marker_cycle[idx]

        method_key = str(method).strip().lower()
        linestyle = line_styles.get(method, line_styles.get(method_key, default_line_styles.get(method_key, "-")))

        _plot_metric_vs_sweep_group_safe(
            axs[0],
            subdf[columns.sweep_value].to_numpy(),
            subdf[columns.fidelity_mean].to_numpy(),
            metric_std=fidelity_std,
            label=method,
            color=color,
            marker=marker,
            linestyle=linestyle,
            grid=grid,
            show_errorband=show_errorband,
        )

        _plot_metric_vs_sweep_group_safe(
            axs[1],
            subdf[columns.sweep_value].to_numpy(),
            subdf[columns.success_mean].to_numpy(),
            metric_std=success_std,
            label=method,
            color=color,
            marker=marker,
            linestyle=linestyle,
            grid=grid,
            show_errorband=show_errorband,
        )

    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("Fidelity")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("Success probability")

    if legend_title is not None:
        axs[0].legend(title=legend_title)
        axs[1].legend(title=legend_title)
    else:
        axs[0].legend()
        axs[1].legend()

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    return fig, axs