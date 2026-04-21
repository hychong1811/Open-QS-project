from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from noise.noise_simulation import SimpleNISQNoiseParameters


@dataclass(frozen=True)
class NoiseSweepSpecification:
    """Specification for a one-parameter noise-resilience sweep.

    Parameters
    ----------
    noise_kind:
        One of:
        - "amplitude_damping"
        - "phase_damping"
        - "depolarizing"
        - "readout"
    sweep_values:
        Values to sweep.
    parameterization:
        Either "probability" or "rate".
        For amplitude/phase damping, "rate" means a decay rate gamma used via
            p = 1 - exp(-gamma * duration)
        while "probability" means the per-step channel probability directly.
        For depolarizing/readout, only "probability" is currently supported.
    duration:
        Effective duration of the noisy step. Required when parameterization is
        "rate" for amplitude or phase damping.
    isolate_noise:
        If True, zero out all other noise mechanisms before setting the swept one.
    apply_to:
        For amplitude/phase damping and readout, choose where the swept noise is
        applied: "system", "ancilla", or "both".
    depolarizing_mode:
        For depolarizing sweeps, choose "1q", "2q", or "both".
    symmetric_readout:
        If True, use the same readout error probability for 0->1 and 1->0.
    """

    noise_kind: str
    sweep_values: Sequence[float]
    parameterization: str = "probability"
    duration: Optional[float] = None
    isolate_noise: bool = True
    apply_to: str = "both"
    depolarizing_mode: str = "both"
    symmetric_readout: bool = True


@dataclass(frozen=True)
class NoiseSweepResult:
    raw_results: pd.DataFrame
    summary_results: pd.DataFrame


RunnerType = Callable[[SimpleNISQNoiseParameters, float, int, np.random.Generator], Mapping[str, Any]]


def linear_sweep(start: float, stop: float, num: int) -> np.ndarray:
    """Return a uniformly spaced sweep grid."""
    if num <= 0:
        raise ValueError("num must be positive.")
    return np.linspace(float(start), float(stop), int(num))


def log_sweep(start: float, stop: float, num: int) -> np.ndarray:
    """Return a logarithmically spaced sweep grid."""
    if start <= 0 or stop <= 0:
        raise ValueError("log_sweep requires positive endpoints.")
    if num <= 0:
        raise ValueError("num must be positive.")
    return np.geomspace(float(start), float(stop), int(num))


def probability_from_decay_rate(decay_rate: float, duration: float) -> float:
    """Convert a decay rate gamma into a per-step channel probability.

    Uses p = 1 - exp(-gamma * duration).
    """
    if decay_rate < 0:
        raise ValueError("decay_rate must be nonnegative.")
    if duration < 0:
        raise ValueError("duration must be nonnegative.")
    return float(1.0 - np.exp(-decay_rate * duration))


def equivalent_relaxation_time_from_probability(probability: float, duration: float) -> float:
    """Convert a per-step amplitude-damping probability into an equivalent T1.

    Solves p = 1 - exp(-duration / T1).
    """
    p = float(probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("probability must satisfy 0 <= p <= 1.")
    if duration < 0:
        raise ValueError("duration must be nonnegative.")
    if p == 0.0:
        return np.inf
    eps = 1e-15
    p_eff = min(p, 1.0 - eps)
    return float(-duration / np.log(1.0 - p_eff))


def equivalent_dephasing_time_from_probability(probability: float, duration: float) -> float:
    """Convert a per-step phase-damping probability into an equivalent Tphi.

    Solves p = 1 - exp(-duration / Tphi).
    """
    return equivalent_relaxation_time_from_probability(probability, duration)


def _validate_probability(probability: float, *, name: str) -> float:
    p = float(probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1.")
    return p


def _normalize_apply_to(apply_to: str) -> str:
    value = str(apply_to).strip().lower()
    if value not in {"system", "ancilla", "both"}:
        raise ValueError("apply_to must be one of {'system', 'ancilla', 'both'}." )
    return value


def _normalize_depolarizing_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in {"1q", "2q", "both"}:
        raise ValueError("depolarizing_mode must be one of {'1q', '2q', 'both'}." )
    return value


def _zero_noise_fields(params: SimpleNISQNoiseParameters) -> SimpleNISQNoiseParameters:
    """Return a copy with all noise strengths zeroed, while preserving gate times."""
    return replace(
        params,
        t1_system=np.inf,
        t2_system=np.inf,
        t1_ancilla=np.inf,
        t2_ancilla=np.inf,
        p1q_depolarizing=0.0,
        p2q_depolarizing=0.0,
        readout_p0_to_1_system=0.0,
        readout_p1_to_0_system=0.0,
        readout_p0_to_1_ancilla=0.0,
        readout_p1_to_0_ancilla=0.0,
    )


def build_noise_parameters_for_sweep(
    base_params: SimpleNISQNoiseParameters,
    *,
    noise_kind: str,
    sweep_value: float,
    parameterization: str = "probability",
    duration: Optional[float] = None,
    isolate_noise: bool = True,
    apply_to: str = "both",
    depolarizing_mode: str = "both",
    symmetric_readout: bool = True,
) -> SimpleNISQNoiseParameters:
    """Create a noise-parameter object for a single sweep point.

    This helper is designed to be compatible with `noise_simulation.py` while
    exposing the more natural sweep variables:
    - amplitude-damping probability or decay rate,
    - phase-damping probability or decay rate,
    - depolarizing probability,
    - readout assignment error.

    Notes
    -----
    `noise_simulation.py` internally parameterizes decoherence through T1/T2.
    This function converts the swept amplitude/phase damping strengths into the
    corresponding equivalent T1/Tphi values over the supplied `duration`.
    """
    kind = str(noise_kind).strip().lower()
    par = str(parameterization).strip().lower()
    apply_to = _normalize_apply_to(apply_to)
    depolarizing_mode = _normalize_depolarizing_mode(depolarizing_mode)

    params = _zero_noise_fields(base_params) if isolate_noise else replace(base_params)

    if kind in {"amplitude_damping", "phase_damping"} and par == "rate":
        if duration is None:
            raise ValueError("duration must be provided when parameterization='rate'.")
        probability = probability_from_decay_rate(float(sweep_value), float(duration))
    elif kind in {"amplitude_damping", "phase_damping", "depolarizing", "readout"} and par == "probability":
        probability = _validate_probability(float(sweep_value), name="sweep_value")
    else:
        raise ValueError(
            "Unsupported combination of noise_kind and parameterization. "
            "Use probability for all kinds, or rate for amplitude/phase damping."
        )

    if kind == "amplitude_damping":
        if duration is None:
            duration = float(base_params.two_qubit_gate_time)
        t1_equiv = equivalent_relaxation_time_from_probability(probability, float(duration))
        updates: dict[str, Any] = {}
        if apply_to in {"system", "both"}:
            updates["t1_system"] = t1_equiv
            if isolate_noise:
                updates["t2_system"] = np.inf
        if apply_to in {"ancilla", "both"}:
            updates["t1_ancilla"] = t1_equiv
            if isolate_noise:
                updates["t2_ancilla"] = np.inf
        return replace(params, **updates)

    if kind == "phase_damping":
        if duration is None:
            duration = float(base_params.two_qubit_gate_time)
        tphi_equiv = equivalent_dephasing_time_from_probability(probability, float(duration))
        updates = {}
        if apply_to in {"system", "both"}:
            if isolate_noise:
                updates["t1_system"] = np.inf
            updates["t2_system"] = tphi_equiv
        if apply_to in {"ancilla", "both"}:
            if isolate_noise:
                updates["t1_ancilla"] = np.inf
            updates["t2_ancilla"] = tphi_equiv
        return replace(params, **updates)

    if kind == "depolarizing":
        updates = {}
        if depolarizing_mode in {"1q", "both"}:
            updates["p1q_depolarizing"] = probability
        if depolarizing_mode in {"2q", "both"}:
            updates["p2q_depolarizing"] = probability
        return replace(params, **updates)

    if kind == "readout":
        updates = {}
        if apply_to in {"system", "both"}:
            updates["readout_p0_to_1_system"] = probability
            updates["readout_p1_to_0_system"] = probability if symmetric_readout else probability
        if apply_to in {"ancilla", "both"}:
            updates["readout_p0_to_1_ancilla"] = probability
            updates["readout_p1_to_0_ancilla"] = probability if symmetric_readout else probability
        return replace(params, **updates)

    raise ValueError("noise_kind must be one of {'amplitude_damping', 'phase_damping', 'depolarizing', 'readout'}." )


def run_noise_resilience_sweep(
    runner: RunnerType,
    *,
    base_params: SimpleNISQNoiseParameters,
    specification: NoiseSweepSpecification,
    repeats: int = 1,
    rng_seed: Optional[int] = None,
    runner_kwargs: Optional[Mapping[str, Any]] = None,
) -> NoiseSweepResult:
    """Execute a generic one-parameter noise-resilience sweep.

    Parameters
    ----------
    runner:
        Callback with signature
            runner(noise_params, sweep_value, repeat_index, rng, **runner_kwargs)
        returning a mapping containing at least whatever metrics you want to
        summarize later, for example
            {'fidelity': ..., 'success_probability': ...}
    base_params:
        Base noise parameters whose gate times and untouched fields are used as
        defaults.
    specification:
        Sweep specification.
    repeats:
        Number of repeated executions per sweep point.
    rng_seed:
        Seed for reproducible repeated runs.
    runner_kwargs:
        Optional extra keyword arguments forwarded to `runner`.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    runner_kwargs = dict(runner_kwargs or {})
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, Any]] = []

    for sweep_index, sweep_value in enumerate(np.asarray(specification.sweep_values, dtype=float).reshape(-1)):
        noise_params = build_noise_parameters_for_sweep(
            base_params,
            noise_kind=specification.noise_kind,
            sweep_value=float(sweep_value),
            parameterization=specification.parameterization,
            duration=specification.duration,
            isolate_noise=specification.isolate_noise,
            apply_to=specification.apply_to,
            depolarizing_mode=specification.depolarizing_mode,
            symmetric_readout=specification.symmetric_readout,
        )

        for repeat_index in range(repeats):
            result = dict(runner(noise_params, float(sweep_value), repeat_index, rng, **runner_kwargs))
            row = {
                "noise_kind": specification.noise_kind,
                "parameterization": specification.parameterization,
                "sweep_index": int(sweep_index),
                "sweep_value": float(sweep_value),
                "repeat_index": int(repeat_index),
            }
            row.update(result)
            rows.append(row)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_noise_resilience_results(raw_df)
    return NoiseSweepResult(raw_results=raw_df, summary_results=summary_df)


def summarize_noise_resilience_results(
    raw_results: pd.DataFrame,
    *,
    metric_columns: Optional[Sequence[str]] = None,
    group_columns: Sequence[str] = ("noise_kind", "parameterization", "sweep_value"),
) -> pd.DataFrame:
    """Aggregate raw repeated-run results into mean/std summary columns.

    The output naming convention is designed to match the plotting helper:
        metric -> metric_mean, metric_std

    Example
    -------
    If `metric_columns=['fidelity', 'success_probability']`, the summary
    DataFrame will contain
        fidelity_mean, fidelity_std,
        success_probability_mean, success_probability_std.
    """
    if not isinstance(raw_results, pd.DataFrame):
        raise TypeError("raw_results must be a pandas DataFrame.")
    if raw_results.empty:
        raise ValueError("raw_results must not be empty.")

    if metric_columns is None:
        excluded = set(group_columns) | {"repeat_index", "sweep_index"}
        metric_columns = [
            column
            for column in raw_results.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(raw_results[column])
        ]

    metric_columns = list(metric_columns)
    if len(metric_columns) == 0:
        raise ValueError("No metric columns were found to summarize.")

    agg_spec: dict[str, list[str]] = {column: ["mean", "std"] for column in metric_columns}
    summary = raw_results.groupby(list(group_columns), dropna=False).agg(agg_spec)
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary = summary.reset_index()

    for column in metric_columns:
        std_col = f"{column}_std"
        if std_col in summary.columns:
            summary[std_col] = summary[std_col].fillna(0.0)
    return summary.sort_values(list(group_columns)).reset_index(drop=True)


def noise_parameters_to_dict(params: SimpleNISQNoiseParameters) -> dict[str, Any]:
    """Return a plain dict representation of a noise-parameter object."""
    return asdict(params)
