from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

try:  # Prefer package-style imports when project root is on sys.path
    from project.algo.dilation_algo import (  # type: ignore
        PostselectionResult,
        build_state_preparation_dilation,
        density_matrix_from_state,
        prepare_subnormalized_state_from_uniform_superposition,
    )
except Exception:  # pragma: no cover - fallback for direct folder imports
    from algo.dilation_algo import (  # type: ignore
        PostselectionResult,
        build_state_preparation_dilation,
        density_matrix_from_state,
        prepare_subnormalized_state_from_uniform_superposition,
    )


ArrayLike = np.ndarray


@dataclass(frozen=True)
class SubnormalizedStateSample:
    """Container for a paper-style subnormalised state-preparation sample.

    Attributes
    ----------
    normalized_two_qubit_state:
        The exact normalized two-qubit pure state
            |psi_2> = a00|00> + a01|01> + a10|10> + a11|11>.
    target_subnormalized_state:
        The one-qubit state extracted from the first two amplitudes,
            |psi_1> = a00|0> + a01|1>.
    exact_density_matrix:
        The unnormalised one-qubit density matrix |psi_1><psi_1|.
    normalized_density_matrix:
        The corresponding normalized one-qubit density matrix.
    state_norm:
        Euclidean norm ||psi_1||.
    state_trace:
        Trace of the unnormalised density matrix, equal to ||psi_1||^2.
    diagonal_operator:
        The nonunitary diagonal operator Sigma = diag(a00, a01).
    dilation_result:
        Optional ideal one-ancilla simulation result for the preparation circuit.
    recovered_density_from_success_branch:
        Optional density matrix recovered from the ancilla-|0> branch via the
        uniform-superposition scaling relation.
    recovery_error_norm:
        Frobenius norm of the recovery error.
    """

    normalized_two_qubit_state: ArrayLike
    target_subnormalized_state: ArrayLike
    exact_density_matrix: ArrayLike
    normalized_density_matrix: ArrayLike
    state_norm: float
    state_trace: float
    diagonal_operator: ArrayLike
    dilation_result: Optional[PostselectionResult]
    recovered_density_from_success_branch: Optional[ArrayLike]
    recovery_error_norm: Optional[float]


@dataclass(frozen=True)
class SubnormalizedStateEnsembleSummary:
    """Summary statistics for a collection of random subnormalised states."""

    num_states: int
    mean_norm: float
    std_norm: float
    mean_trace: float
    std_trace: float
    min_norm: float
    max_norm: float



def _as_complex_vector(state: Sequence[complex], *, expected_dim: Optional[int] = None) -> ArrayLike:
    arr = np.asarray(state, dtype=complex).reshape(-1)
    if expected_dim is not None and arr.size != expected_dim:
        raise ValueError(f"Expected a vector of length {expected_dim}, got {arr.size}.")
    return arr



def normalize_state_vector(state: Sequence[complex], tol: float = 1e-12) -> ArrayLike:
    """Return the normalized copy of a complex state vector."""
    vec = _as_complex_vector(state)
    norm = np.linalg.norm(vec)
    if norm <= tol:
        raise ValueError("Cannot normalize a zero vector.")
    return vec / norm



def random_normalized_two_qubit_state(
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> ArrayLike:
    """Generate a Haar-like random normalized two-qubit pure state.

    The implementation uses i.i.d. complex Gaussian amplitudes followed by
    normalization, which yields the standard unit-sphere distribution for pure
    states.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    vec = rng.normal(size=4) + 1j * rng.normal(size=4)
    return normalize_state_vector(vec)



def extract_subnormalized_one_qubit_state(
    normalized_two_qubit_state: Sequence[complex],
) -> ArrayLike:
    """Extract |psi_1> = a00|0> + a01|1> from a normalized two-qubit state.

    This matches the paper's state-preparation construction, where the target
    one-qubit subnormalised state is taken from the first two amplitudes of the
    normalized two-qubit state ordered as
        [a00, a01, a10, a11].
    """
    psi2 = _as_complex_vector(normalized_two_qubit_state, expected_dim=4)
    norm = np.linalg.norm(psi2)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError("Input two-qubit state must be normalized.")
    return psi2[:2].copy()



def exact_subnormalized_density_matrix(
    subnormalized_state: Sequence[complex],
) -> ArrayLike:
    """Return the unnormalised density matrix |psi><psi|."""
    psi = _as_complex_vector(subnormalized_state, expected_dim=2)
    return density_matrix_from_state(psi)



def normalized_density_matrix_from_subnormalized_state(
    subnormalized_state: Sequence[complex],
    tol: float = 1e-12,
) -> ArrayLike:
    """Normalize |psi><psi| to trace 1."""
    rho = exact_subnormalized_density_matrix(subnormalized_state)
    tr = np.trace(rho)
    if abs(tr) <= tol:
        raise ValueError("Cannot normalize a zero-trace density matrix.")
    return rho / tr



def state_preparation_diagonal_operator(
    subnormalized_state: Sequence[complex],
) -> ArrayLike:
    """Return Sigma = diag(c0, c1) for |psi> = c0|0> + c1|1>."""
    coeffs = _as_complex_vector(subnormalized_state, expected_dim=2)
    return np.diag(coeffs)



def recover_target_density_from_success_branch(
    success_branch_state: Sequence[complex],
    *,
    num_system_qubits: int,
) -> ArrayLike:
    """Recover the target subnormalised density from the ancilla-|0> branch.

    In the state-preparation circuit, the ancilla-|0> branch equals
        Sigma |~_k>,
    where |~_k> is the k-qubit uniform superposition with amplitudes
        1 / sqrt(2^k).
    Therefore, if the desired target state is
        |phi> = sum_j c_j |j>,
    then
        |phi> = sqrt(2^k) * |success_branch>,
    and the corresponding density matrix satisfies
        |phi><phi| = 2^k * |success_branch><success_branch|.
    """
    branch = _as_complex_vector(success_branch_state)
    expected_dim = 2 ** num_system_qubits
    if branch.size != expected_dim:
        raise ValueError(
            f"Success branch dimension {branch.size} is incompatible with "
            f"num_system_qubits={num_system_qubits}."
        )
    return (2 ** num_system_qubits) * density_matrix_from_state(branch)



def recover_target_density_from_postselection(
    normalized_postselected_state: Sequence[complex],
    *,
    success_probability: float,
    num_system_qubits: int,
) -> ArrayLike:
    """Recover the target subnormalised density from postselected data.

    If tomography is performed on the normalized successful branch, the target
    unnormalised density matrix is
        2^k * p_success * |psi_cond><psi_cond|,
    where k is the number of system qubits and p_success is the probability of
    measuring ancilla |0>.
    """
    if success_probability < 0:
        raise ValueError("success_probability must be nonnegative.")
    psi = _as_complex_vector(normalized_postselected_state)
    expected_dim = 2 ** num_system_qubits
    if psi.size != expected_dim:
        raise ValueError(
            f"Postselected state dimension {psi.size} is incompatible with "
            f"num_system_qubits={num_system_qubits}."
        )
    return (2 ** num_system_qubits) * float(success_probability) * density_matrix_from_state(psi)



def build_subnormalized_state_sample(
    normalized_two_qubit_state: Sequence[complex],
    *,
    simulate_dilation: bool = True,
    auto_scale: bool = False,
    tol: float = 1e-12,
) -> SubnormalizedStateSample:
    """Construct all exact data for one paper-style state-preparation instance."""
    psi2 = _as_complex_vector(normalized_two_qubit_state, expected_dim=4)
    psi2 = normalize_state_vector(psi2, tol=tol)
    psi1 = extract_subnormalized_one_qubit_state(psi2)

    rho_exact = exact_subnormalized_density_matrix(psi1)
    trace_val = float(np.real(np.trace(rho_exact)))
    norm_val = float(np.linalg.norm(psi1))
    rho_normalized = normalized_density_matrix_from_subnormalized_state(psi1, tol=tol)
    sigma = state_preparation_diagonal_operator(psi1)

    dilation_result: Optional[PostselectionResult] = None
    recovered_density: Optional[ArrayLike] = None
    recovery_error_norm: Optional[float] = None

    if simulate_dilation:
        dilation_result = prepare_subnormalized_state_from_uniform_superposition(
            psi1,
            auto_scale=auto_scale,
            tol=tol,
        )
        recovered_density = recover_target_density_from_success_branch(
            dilation_result.ancilla_zero_branch,
            num_system_qubits=1,
        )
        recovery_error_norm = float(np.linalg.norm(recovered_density - rho_exact, ord="fro"))

    return SubnormalizedStateSample(
        normalized_two_qubit_state=psi2,
        target_subnormalized_state=psi1,
        exact_density_matrix=rho_exact,
        normalized_density_matrix=rho_normalized,
        state_norm=norm_val,
        state_trace=trace_val,
        diagonal_operator=sigma,
        dilation_result=dilation_result,
        recovered_density_from_success_branch=recovered_density,
        recovery_error_norm=recovery_error_norm,
    )



def random_subnormalized_state_sample(
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    simulate_dilation: bool = True,
    auto_scale: bool = False,
    tol: float = 1e-12,
) -> SubnormalizedStateSample:
    """Generate one random paper-style subnormalised state-preparation sample."""
    psi2 = random_normalized_two_qubit_state(seed=seed, rng=rng)
    return build_subnormalized_state_sample(
        psi2,
        simulate_dilation=simulate_dilation,
        auto_scale=auto_scale,
        tol=tol,
    )



def generate_random_subnormalized_state_ensemble(
    num_states: int,
    *,
    seed: Optional[int] = None,
    simulate_dilation: bool = True,
    auto_scale: bool = False,
    tol: float = 1e-12,
) -> list[SubnormalizedStateSample]:
    """Generate an ensemble of random subnormalised one-qubit states.

    Each sample is produced from a random normalized two-qubit state by taking
    the first two amplitudes, matching the paper's construction.
    """
    if num_states <= 0:
        raise ValueError("num_states must be a positive integer.")

    rng = np.random.default_rng(seed)
    samples: list[SubnormalizedStateSample] = []
    for _ in range(num_states):
        psi2 = random_normalized_two_qubit_state(rng=rng)
        samples.append(
            build_subnormalized_state_sample(
                psi2,
                simulate_dilation=simulate_dilation,
                auto_scale=auto_scale,
                tol=tol,
            )
        )
    return samples



def summarize_subnormalized_state_ensemble(
    samples: Iterable[SubnormalizedStateSample],
) -> SubnormalizedStateEnsembleSummary:
    """Return norm/trace statistics for a sample ensemble."""
    samples = list(samples)
    if len(samples) == 0:
        raise ValueError("samples must contain at least one element.")

    norms = np.array([s.state_norm for s in samples], dtype=float)
    traces = np.array([s.state_trace for s in samples], dtype=float)

    return SubnormalizedStateEnsembleSummary(
        num_states=len(samples),
        mean_norm=float(np.mean(norms)),
        std_norm=float(np.std(norms, ddof=0)),
        mean_trace=float(np.mean(traces)),
        std_trace=float(np.std(traces, ddof=0)),
        min_norm=float(np.min(norms)),
        max_norm=float(np.max(norms)),
    )



def paper_style_state_preparation_data(
    num_states: int = 98,
    *,
    seed: Optional[int] = None,
    simulate_dilation: bool = True,
    auto_scale: bool = False,
    tol: float = 1e-12,
) -> tuple[list[SubnormalizedStateSample], SubnormalizedStateEnsembleSummary]:
    """Convenience wrapper for the paper-style random-state experiment."""
    samples = generate_random_subnormalized_state_ensemble(
        num_states,
        seed=seed,
        simulate_dilation=simulate_dilation,
        auto_scale=auto_scale,
        tol=tol,
    )
    return samples, summarize_subnormalized_state_ensemble(samples)



def build_diagonal_dilation_for_sample(
    sample: SubnormalizedStateSample,
    *,
    auto_scale: bool = False,
    tol: float = 1e-12,
):
    """Return the diagonal one-ancilla dilation associated with a sample."""
    return build_state_preparation_dilation(
        sample.target_subnormalized_state,
        auto_scale=auto_scale,
        tol=tol,
    )
