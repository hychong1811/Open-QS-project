from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Union

import numpy as np
import scipy.linalg as la

try:
    import qutip as qt
    from qutip import Qobj
except Exception:  # pragma: no cover - optional dependency
    qt = None
    Qobj = None


BasisLabel = Literal["X", "Y", "Z"]
StateLike = Union[np.ndarray, "Qobj"]
DensityLike = Union[np.ndarray, "Qobj"]


@dataclass(frozen=True)
class SingleQubitTomographyResult:
    """Container for one-qubit tomography data.

    Attributes
    ----------
    input_density_matrix:
        The exact input density matrix used to generate measurement outcomes.
    ideal_expectations:
        Exact Bloch-component expectations Tr(rho sigma_i).
    measured_expectations:
        Expectations estimated from measurement counts. If shots=None, these
        equal the ideal expectations.
    basis_probabilities:
        Exact measurement probabilities in each tomography basis.
    counts:
        Sampled counts in each basis. If shots=None, this is None.
    shots:
        Number of shots per basis. If None, exact probabilities are used.
    reconstructed_density_matrix:
        Linear-inversion density matrix from measured expectations.
    physical_density_matrix:
        Reconstructed matrix projected onto the physical one-qubit state space
        by PSD projection and trace renormalization.
    """

    input_density_matrix: np.ndarray
    ideal_expectations: Dict[str, float]
    measured_expectations: Dict[str, float]
    basis_probabilities: Dict[str, Dict[str, float]]
    counts: Optional[Dict[str, Dict[str, int]]]
    shots: Optional[int]
    reconstructed_density_matrix: np.ndarray
    physical_density_matrix: np.ndarray



def _as_complex_array(x: DensityLike) -> np.ndarray:
    if Qobj is not None and isinstance(x, Qobj):
        return np.asarray(x.full(), dtype=complex)
    return np.asarray(x, dtype=complex)



def _as_density_matrix(state_or_rho: DensityLike) -> np.ndarray:
    """Convert a ket or density operator into a density matrix."""
    if Qobj is not None and isinstance(state_or_rho, Qobj):
        if state_or_rho.isket:
            vec = np.asarray(state_or_rho.full(), dtype=complex).reshape(-1)
            return np.outer(vec, vec.conj())
        arr = np.asarray(state_or_rho.full(), dtype=complex)
    else:
        arr = np.asarray(state_or_rho, dtype=complex)
        if arr.ndim == 1:
            vec = arr.reshape(-1)
            return np.outer(vec, vec.conj())

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input must be a state vector or a square density matrix.")
    return arr



def _validate_one_qubit_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> None:
    if rho.shape != (2, 2):
        raise ValueError("This module currently supports one-qubit tomography only.")
    if not np.allclose(rho, rho.conj().T, atol=tol):
        raise ValueError("Density matrix must be Hermitian within tolerance.")
    tr = np.trace(rho)
    if np.real(tr) < -tol:
        raise ValueError("Density matrix trace must be nonnegative.")



def pauli_operators() -> Dict[str, np.ndarray]:
    """Return the one-qubit Pauli operators and identity."""
    return {
        "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }



def normalize_density_matrix(rho: DensityLike, tol: float = 1e-12) -> np.ndarray:
    """Normalize a density matrix to trace 1.

    This is useful for subnormalized states, where the fidelity in the paper is
    computed on normalized density matrices while the Frobenius distance is
    computed on the unnormalized states.
    """
    arr = _as_density_matrix(rho)
    tr = np.trace(arr)
    if abs(tr) <= tol:
        raise ValueError("Cannot normalize a zero-trace density matrix.")
    return arr / tr



def bloch_vector_from_density_matrix(rho: DensityLike) -> np.ndarray:
    """Return the Bloch vector (⟨X⟩, ⟨Y⟩, ⟨Z⟩) for a one-qubit state."""
    arr = _as_density_matrix(rho)
    _validate_one_qubit_density_matrix(arr)
    paulis = pauli_operators()
    return np.array(
        [
            np.real(np.trace(arr @ paulis["X"])),
            np.real(np.trace(arr @ paulis["Y"])),
            np.real(np.trace(arr @ paulis["Z"])),
        ],
        dtype=float,
    )



def density_matrix_from_bloch_vector(bloch: np.ndarray, trace: float = 1.0) -> np.ndarray:
    """Construct rho = (trace / 2) [I + r·sigma] for a one-qubit state."""
    r = np.asarray(bloch, dtype=float).reshape(3)
    paulis = pauli_operators()
    rho = 0.5 * trace * (
        paulis["I"] + r[0] * paulis["X"] + r[1] * paulis["Y"] + r[2] * paulis["Z"]
    )
    return np.asarray(rho, dtype=complex)



def expectation_value(rho: DensityLike, operator: DensityLike) -> complex:
    """Compute Tr(rho O)."""
    rho_arr = _as_density_matrix(rho)
    op = _as_complex_array(operator)
    return np.trace(rho_arr @ op)



def basis_projectors(basis: BasisLabel) -> Dict[str, np.ndarray]:
    """Projectors for the one-qubit X, Y, or Z measurement basis."""
    basis = basis.upper()
    if basis == "Z":
        ket0 = np.array([1.0, 0.0], dtype=complex)
        ket1 = np.array([0.0, 1.0], dtype=complex)
        return {
            "0": np.outer(ket0, ket0.conj()),
            "1": np.outer(ket1, ket1.conj()),
        }
    if basis == "X":
        ket_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
        ket_minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2.0)
        return {
            "0": np.outer(ket_plus, ket_plus.conj()),
            "1": np.outer(ket_minus, ket_minus.conj()),
        }
    if basis == "Y":
        ket_plus_i = np.array([1.0, 1j], dtype=complex) / np.sqrt(2.0)
        ket_minus_i = np.array([1.0, -1j], dtype=complex) / np.sqrt(2.0)
        return {
            "0": np.outer(ket_plus_i, ket_plus_i.conj()),
            "1": np.outer(ket_minus_i, ket_minus_i.conj()),
        }
    raise ValueError("basis must be one of 'X', 'Y', 'Z'.")



def basis_probabilities(rho: DensityLike, basis: BasisLabel) -> Dict[str, float]:
    """Exact probabilities of outcomes '0' and '1' in a tomography basis."""
    arr = _as_density_matrix(rho)
    _validate_one_qubit_density_matrix(arr)
    projectors = basis_projectors(basis)
    probs = {
        outcome: float(np.real(np.trace(arr @ proj)))
        for outcome, proj in projectors.items()
    }
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Total probability is nonpositive; check the input state.")
    for key in probs:
        probs[key] = max(0.0, probs[key])
    total = sum(probs.values())
    for key in probs:
        probs[key] /= total
    return probs



def sample_basis_counts(
    rho: DensityLike,
    basis: BasisLabel,
    shots: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, int]:
    """Sample projective tomography counts in a chosen basis."""
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")
    if rng is None:
        rng = np.random.default_rng()
    probs = basis_probabilities(rho, basis)
    sampled = rng.multinomial(shots, [probs["0"], probs["1"]])
    return {"0": int(sampled[0]), "1": int(sampled[1])}



def expectation_from_counts(counts: Mapping[str, int]) -> float:
    """Estimate a Pauli expectation from counts in its eigenbasis.

    The tomography conventions here map outcome '0' to the +1 eigenvalue and
    outcome '1' to the -1 eigenvalue, so

        <sigma> = (n_0 - n_1) / (n_0 + n_1).
    """
    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))
    total = n0 + n1
    if total <= 0:
        raise ValueError("Counts must contain at least one shot.")
    return float((n0 - n1) / total)



def linear_inversion_density_matrix(
    expectations: Mapping[str, float],
    *,
    trace: float = 1.0,
) -> np.ndarray:
    """Reconstruct a one-qubit density matrix by linear inversion.

    For single-qubit tomography,

        rho = (trace / 2) [I + <X>X + <Y>Y + <Z>Z].
    """
    ex = float(expectations["X"])
    ey = float(expectations["Y"])
    ez = float(expectations["Z"])
    return density_matrix_from_bloch_vector(np.array([ex, ey, ez]), trace=trace)



def project_to_physical_density_matrix(rho: DensityLike, tol: float = 1e-12) -> np.ndarray:
    """Project a Hermitian matrix to the nearest physical density matrix.

    This is done by diagonalizing, clipping negative eigenvalues to zero, and
    renormalizing the trace to 1. For one-qubit tomography with finite shots,
    linear inversion can otherwise yield a slightly unphysical estimate.
    """
    arr = _as_density_matrix(rho)
    herm = 0.5 * (arr + arr.conj().T)
    evals, evecs = la.eigh(herm)
    evals = np.clip(np.real_if_close(evals), 0.0, None)
    if np.sum(evals) <= tol:
        raise ValueError("Cannot project a zero matrix to a physical density matrix.")
    evals = evals / np.sum(evals)
    return evecs @ np.diag(evals) @ evecs.conj().T



def perform_single_qubit_tomography(
    state_or_rho: DensityLike,
    *,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_physical: bool = True,
) -> SingleQubitTomographyResult:
    """Perform one-qubit X/Y/Z tomography by linear inversion.

    Parameters
    ----------
    state_or_rho:
        One-qubit state vector or density matrix. Subnormalized one-qubit states
        are accepted; the linear inversion preserves the input trace.
    shots:
        Number of shots per tomography basis. If None, exact expectations are
        used instead of multinomial sampling.
    seed:
        Random seed for multinomial sampling.
    enforce_physical:
        If True, project the normalized reconstructed density matrix to the
        nearest physical density matrix.
    """
    rho = _as_density_matrix(state_or_rho)
    _validate_one_qubit_density_matrix(rho)

    trace_val = float(np.real(np.trace(rho)))
    if trace_val <= 0:
        raise ValueError("Input density matrix must have positive trace.")

    rho_normalized = normalize_density_matrix(rho)
    paulis = pauli_operators()
    ideal_expectations = {
        label: float(np.real(np.trace(rho_normalized @ paulis[label])))
        for label in ("X", "Y", "Z")
    }
    probabilities = {
        label: basis_probabilities(rho_normalized, label)
        for label in ("X", "Y", "Z")
    }

    counts: Optional[Dict[str, Dict[str, int]]] = None
    if shots is None:
        measured_expectations = dict(ideal_expectations)
    else:
        if shots <= 0:
            raise ValueError("shots must be a positive integer when provided.")
        rng = np.random.default_rng(seed)
        counts = {
            label: sample_basis_counts(rho_normalized, label, shots, rng)
            for label in ("X", "Y", "Z")
        }
        measured_expectations = {
            label: expectation_from_counts(counts[label])
            for label in ("X", "Y", "Z")
        }

    reconstructed_normalized = linear_inversion_density_matrix(measured_expectations, trace=1.0)
    if enforce_physical:
        physical_normalized = project_to_physical_density_matrix(reconstructed_normalized)
    else:
        physical_normalized = reconstructed_normalized

    reconstructed_subnormalized = trace_val * reconstructed_normalized
    physical_subnormalized = trace_val * physical_normalized

    return SingleQubitTomographyResult(
        input_density_matrix=rho,
        ideal_expectations=ideal_expectations,
        measured_expectations=measured_expectations,
        basis_probabilities=probabilities,
        counts=counts,
        shots=shots,
        reconstructed_density_matrix=reconstructed_subnormalized,
        physical_density_matrix=physical_subnormalized,
    )



def batch_single_qubit_tomography(
    states_or_rhos: Any,
    *,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_physical: bool = True,
) -> list[SingleQubitTomographyResult]:
    """Run single-qubit tomography over an iterable of states or density matrices."""
    rng = np.random.default_rng(seed)
    results: list[SingleQubitTomographyResult] = []
    for item in states_or_rhos:
        sub_seed = None if shots is None else int(rng.integers(0, 2**32 - 1))
        results.append(
            perform_single_qubit_tomography(
                item,
                shots=shots,
                seed=sub_seed,
                enforce_physical=enforce_physical,
            )
        )
    return results
