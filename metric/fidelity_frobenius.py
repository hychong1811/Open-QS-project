from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy as np
import scipy.linalg as la

try:
    from qutip import Qobj
except Exception:  # pragma: no cover - optional dependency
    Qobj = None

from .tomography import normalize_density_matrix


DensityLike = Union[np.ndarray, "Qobj"]


@dataclass(frozen=True)
class FidelityFrobeniusResult:
    """Container for the paper's two main comparison metrics.

    The paper computes
    1. fidelity between normalized exact and simulated density matrices, and
    2. Frobenius distance between the unnormalized density matrices.
    """

    exact_density_matrix: np.ndarray
    simulated_density_matrix: np.ndarray
    exact_density_matrix_normalized: np.ndarray
    simulated_density_matrix_normalized: np.ndarray
    fidelity: float
    frobenius_distance: float


@dataclass(frozen=True)
class BatchMetricSummary:
    """Mean and standard deviation summary over many generated states."""

    sample_size: int
    mean_fidelity: float
    std_fidelity: float
    mean_frobenius_distance: float
    std_frobenius_distance: float



def _as_complex_density_matrix(x: DensityLike) -> np.ndarray:
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



def _hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)



def matrix_sqrt_psd(rho: DensityLike, tol: float = 1e-12) -> np.ndarray:
    """Positive-semidefinite square root using an eigenvalue decomposition."""
    arr = _hermitize(_as_complex_density_matrix(rho))
    evals, evecs = la.eigh(arr)
    evals = np.clip(np.real_if_close(evals), 0.0, None)
    sqrt_evals = np.sqrt(np.maximum(evals, 0.0))
    return evecs @ np.diag(sqrt_evals) @ evecs.conj().T



def fidelity_uhlmann_squared(
    rho_a: DensityLike,
    rho_b: DensityLike,
    *,
    normalize_inputs: bool = True,
    tol: float = 1e-12,
) -> float:
    """Return the standard density-matrix fidelity.

    This implements Eq. (13) of Schlimgen et al.:

        F(rho_a, rho_b) = [Tr sqrt( sqrt(rho_a) rho_b sqrt(rho_a) )]^2.

    By default, inputs are normalized first, which matches the paper's metric
    convention for subnormalized-state tomography.
    """
    a = _as_complex_density_matrix(rho_a)
    b = _as_complex_density_matrix(rho_b)

    if normalize_inputs:
        a = normalize_density_matrix(a, tol=tol)
        b = normalize_density_matrix(b, tol=tol)

    a = _hermitize(a)
    b = _hermitize(b)

    sqrt_a = matrix_sqrt_psd(a, tol=tol)
    middle = sqrt_a @ b @ sqrt_a
    sqrt_middle = matrix_sqrt_psd(middle, tol=tol)
    fid = np.trace(sqrt_middle)
    return float(np.real(fid * np.conj(fid)))



def frobenius_distance(rho_a: DensityLike, rho_b: DensityLike) -> float:
    """Return ||rho_a - rho_b||_F for unnormalized density matrices."""
    a = _as_complex_density_matrix(rho_a)
    b = _as_complex_density_matrix(rho_b)
    return float(la.norm(a - b, ord="fro"))



def evaluate_fidelity_frobenius(
    exact_density_matrix: DensityLike,
    simulated_density_matrix: DensityLike,
    *,
    tol: float = 1e-12,
) -> FidelityFrobeniusResult:
    """Evaluate the paper's fidelity and Frobenius metrics together."""
    exact = _as_complex_density_matrix(exact_density_matrix)
    simulated = _as_complex_density_matrix(simulated_density_matrix)

    exact_n = normalize_density_matrix(exact, tol=tol)
    simulated_n = normalize_density_matrix(simulated, tol=tol)

    return FidelityFrobeniusResult(
        exact_density_matrix=exact,
        simulated_density_matrix=simulated,
        exact_density_matrix_normalized=exact_n,
        simulated_density_matrix_normalized=simulated_n,
        fidelity=fidelity_uhlmann_squared(exact_n, simulated_n, normalize_inputs=False, tol=tol),
        frobenius_distance=frobenius_distance(exact, simulated),
    )



def batch_evaluate_fidelity_frobenius(
    exact_density_matrices: Sequence[DensityLike],
    simulated_density_matrices: Sequence[DensityLike],
    *,
    tol: float = 1e-12,
) -> list[FidelityFrobeniusResult]:
    """Evaluate fidelity and Frobenius distance pairwise over two sequences."""
    if len(exact_density_matrices) != len(simulated_density_matrices):
        raise ValueError("Input sequences must have the same length.")
    return [
        evaluate_fidelity_frobenius(exact, simulated, tol=tol)
        for exact, simulated in zip(exact_density_matrices, simulated_density_matrices)
    ]



def summarize_metric_results(
    results: Iterable[FidelityFrobeniusResult],
    *,
    sample_size: int,
) -> BatchMetricSummary:
    """Average fidelity and Frobenius distance over many prepared states."""
    results = list(results)
    if len(results) == 0:
        raise ValueError("results must contain at least one metric result.")

    fidelities = np.array([r.fidelity for r in results], dtype=float)
    distances = np.array([r.frobenius_distance for r in results], dtype=float)

    return BatchMetricSummary(
        sample_size=int(sample_size),
        mean_fidelity=float(np.mean(fidelities)),
        std_fidelity=float(np.std(fidelities, ddof=0)),
        mean_frobenius_distance=float(np.mean(distances)),
        std_frobenius_distance=float(np.std(distances, ddof=0)),
    )
