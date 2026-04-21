from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

try:
    from qutip import Qobj
except Exception:  # pragma: no cover - optional dependency
    Qobj = None


DensityLike = Union[np.ndarray, "Qobj"]


@dataclass(frozen=True)
class SuccessProbabilityResult:
    """Container for a single success-probability estimate."""

    success_probability: float
    total_probability: float
    success_weight: Optional[float] = None
    label: Optional[str] = None


@dataclass(frozen=True)
class SuccessProbabilitySummary:
    """Mean/std summary for many success probabilities."""

    sample_size: int
    mean_success_probability: float
    std_success_probability: float
    min_success_probability: float
    max_success_probability: float



def _as_density_matrix(x: DensityLike) -> np.ndarray:
    if Qobj is not None and isinstance(x, Qobj):
        if x.isket:
            vec = np.asarray(x.full(), dtype=complex).reshape(-1)
            return np.outer(vec, vec.conj())
        arr = np.asarray(x.full(), dtype=complex)
    else:
        arr = np.asarray(x, dtype=complex)
        if arr.ndim == 1:
            vec = arr.reshape(-1)
            return np.outer(vec, vec.conj())

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input must be a state vector or a square density matrix.")
    return arr



def _as_square_operator(op: np.ndarray) -> np.ndarray:
    arr = np.asarray(op, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Operator must be a square matrix.")
    return arr



def _nonnegative_real_trace(mat: np.ndarray, tol: float = 1e-12) -> float:
    val = float(np.real(np.trace(mat)))
    if val < -tol:
        raise ValueError("Encountered a negative trace beyond tolerance.")
    return max(0.0, val)



def success_probability_for_operator(
    density_matrix: DensityLike,
    operator: np.ndarray,
    *,
    label: Optional[str] = None,
    tol: float = 1e-12,
) -> SuccessProbabilityResult:
    """Return p_succ = Tr(K rho K^dagger) for one nonunitary operator K.

    This is the natural success probability for a single ancilla-postselected
    run of the dilation algorithm.
    """
    rho = _as_density_matrix(density_matrix)
    k = _as_square_operator(operator)
    if rho.shape[0] != k.shape[0]:
        raise ValueError("density_matrix and operator dimensions must agree.")

    out = k @ rho @ k.conj().T
    p_succ = _nonnegative_real_trace(out, tol=tol)
    total = _nonnegative_real_trace(rho, tol=tol)
    if total <= tol:
        raise ValueError("Input density matrix has zero trace.")
    return SuccessProbabilityResult(
        success_probability=float(p_succ / total),
        total_probability=total,
        success_weight=None,
        label=label,
    )



def output_trace_for_operator(
    density_matrix: DensityLike,
    operator: np.ndarray,
    *,
    tol: float = 1e-12,
) -> float:
    """Return Tr(K rho K^dagger) without normalizing by Tr(rho)."""
    rho = _as_density_matrix(density_matrix)
    k = _as_square_operator(operator)
    if rho.shape[0] != k.shape[0]:
        raise ValueError("density_matrix and operator dimensions must agree.")
    return _nonnegative_real_trace(k @ rho @ k.conj().T, tol=tol)



def success_probability_from_binary_probabilities(
    probabilities: Sequence[float],
    *,
    success_index: int = 0,
    tol: float = 1e-12,
    label: Optional[str] = None,
) -> SuccessProbabilityResult:
    """Estimate success probability from a binary probability vector.

    Example:
        probabilities = [P(success), P(failure)]
    """
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.size == 0:
        raise ValueError("probabilities must be nonempty.")
    if not (0 <= success_index < probs.size):
        raise ValueError("success_index out of range.")
    if np.any(probs < -tol):
        raise ValueError("probabilities must be nonnegative within tolerance.")

    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= tol:
        raise ValueError("Total probability is zero.")
    return SuccessProbabilityResult(
        success_probability=float(probs[success_index] / total),
        total_probability=total,
        success_weight=None,
        label=label,
    )



def success_probability_from_counts(
    counts: Mapping[object, int],
    *,
    success_key: object = 0,
    label: Optional[str] = None,
) -> SuccessProbabilityResult:
    """Estimate success probability from sampled counts.

    This is useful for ancilla-postselection counts, e.g. counts={0: n0, 1: n1}
    or counts={"0": n0, "1": n1}.
    """
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        raise ValueError("Counts must contain at least one shot.")
    success = int(counts.get(success_key, 0))
    if success < 0:
        raise ValueError("Success count must be nonnegative.")
    return SuccessProbabilityResult(
        success_probability=float(success / total),
        total_probability=float(total),
        success_weight=None,
        label=label,
    )



def effective_success_probability(
    success_probabilities: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
) -> SuccessProbabilityResult:
    """Return the weighted effective retained-shot fraction.

    For a workflow that executes multiple subruns r with weights w_r and each
    subrun has success probability p_r, this returns

        p_eff = sum_r w_r p_r / sum_r w_r.

    For a trace-preserving channel implemented branch-by-branch, this is often
    the operationally relevant retained-shot fraction.
    """
    ps = np.asarray(success_probabilities, dtype=float).reshape(-1)
    if ps.size == 0:
        raise ValueError("success_probabilities must be nonempty.")
    if np.any(ps < 0.0) or np.any(ps > 1.0):
        raise ValueError("success_probabilities must lie in [0, 1].")

    if weights is None:
        w = np.ones_like(ps, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != ps.shape:
            raise ValueError("weights must have the same length as success_probabilities.")
        if np.any(w < 0.0):
            raise ValueError("weights must be nonnegative.")

    total_weight = float(np.sum(w))
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive number.")

    return SuccessProbabilityResult(
        success_probability=float(np.sum(w * ps) / total_weight),
        total_probability=total_weight,
        success_weight=total_weight,
        label=label,
    )



def success_probabilities_for_operator_list(
    density_matrix: DensityLike,
    operators: Sequence[np.ndarray],
    *,
    labels: Optional[Sequence[str]] = None,
    tol: float = 1e-12,
) -> list[SuccessProbabilityResult]:
    """Compute success probabilities for a list of operators on one state."""
    if labels is not None and len(labels) != len(operators):
        raise ValueError("labels must match the number of operators.")
    results: list[SuccessProbabilityResult] = []
    for idx, op in enumerate(operators):
        label = None if labels is None else labels[idx]
        results.append(success_probability_for_operator(density_matrix, op, label=label, tol=tol))
    return results



def summarize_success_probability_results(
    results: Iterable[SuccessProbabilityResult],
    *,
    sample_size: Optional[int] = None,
) -> SuccessProbabilitySummary:
    """Return mean/std/min/max summary over many success probabilities."""
    results = list(results)
    if len(results) == 0:
        raise ValueError("results must contain at least one item.")

    vals = np.array([r.success_probability for r in results], dtype=float)
    n = len(results) if sample_size is None else int(sample_size)
    return SuccessProbabilitySummary(
        sample_size=n,
        mean_success_probability=float(np.mean(vals)),
        std_success_probability=float(np.std(vals, ddof=0)),
        min_success_probability=float(np.min(vals)),
        max_success_probability=float(np.max(vals)),
    )

def method_comparison_success_probability(
    method: str,
    *,
    raw_success_probability: Optional[float] = None,
    unitary_baseline_value: float = 1.0,
    unitary_baseline_as_nan: bool = False,
) -> float:
    """
    Return a standardized success-probability value for cross-method comparison.

    Parameters
    ----------
    method:
        Method label, for example:
        - "paper_svd"
        - "sz_nagy"
        - "unitary_rescaled"
    raw_success_probability:
        The postselection success probability computed from the run.
        This is used for methods that genuinely involve postselection.
    unitary_baseline_value:
        Value assigned to the unitary-only baseline when it is treated as
        having deterministic acceptance. Default is 1.0.
    unitary_baseline_as_nan:
        If True, return np.nan instead of a flat value for the unitary-only
        baseline. This is useful when you want to mark the success probability
        as "not applicable" rather than "equal to one".

    Returns
    -------
    float
        Standardized success-probability value for plotting and aggregation.
    """
    method = str(method).strip().lower()

    if method in {"paper_svd", "svd", "paper", "sz_nagy", "sz-nagy", "sznagy"}:
        if raw_success_probability is None:
            raise ValueError(
                f"raw_success_probability must be provided for method '{method}'."
            )
        return float(raw_success_probability)

    if method in {"unitary_rescaled", "unitary-only", "unitary_only", "unitary"}:
        if unitary_baseline_as_nan:
            return float(np.nan)
        return float(unitary_baseline_value)

    raise ValueError(f"Unknown method label: {method}")