from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import scipy.linalg as la

try:
    import qutip as qt
    from qutip import Qobj
except Exception:  # pragma: no cover - optional dependency
    qt = None
    Qobj = None


ArrayLike = np.ndarray
StateLike = Union[np.ndarray, "Qobj"]


@dataclass(frozen=True)
class SzNagyDilation:
    """Direct one-ancilla Sz.-Nagy dilation for a square contraction K.

    For a contraction K with ||K||_2 <= 1, define the defect operators

        D_K      = sqrt(I - K^\dagger K),
        D_Kstar  = sqrt(I - K K^\dagger).

    The corresponding Julia operator is the unitary

        U = [[K,       D_Kstar],
             [D_K,    -K^\dagger]].

    Acting on |0> \otimes |psi> produces the ancilla-|0> branch K|psi> and the
    ancilla-|1> branch D_K|psi>.
    """

    original_operator: ArrayLike
    scaled_operator: ArrayLike
    scale_factor: float
    defect_right: ArrayLike
    defect_left: ArrayLike
    unitary: ArrayLike

    @property
    def operator(self) -> ArrayLike:
        return self.scaled_operator

    @property
    def dimension(self) -> int:
        return int(self.scaled_operator.shape[0])

    def reconstructed_block_operator(self) -> ArrayLike:
        return self.unitary[: self.dimension, : self.dimension]


@dataclass(frozen=True)
class StatePostselectionResult:
    """Result of applying the Sz.-Nagy dilation to a pure state."""

    input_state: ArrayLike
    ancilla_system_output: ArrayLike
    ancilla_zero_branch: ArrayLike
    ancilla_one_branch: ArrayLike
    p_success: float
    p_failure: float
    normalized_success_state: Optional[ArrayLike]
    target_state: ArrayLike
    state_error_norm: float
    scale_factor: float


@dataclass(frozen=True)
class DensityPostselectionResult:
    """Result of applying the Sz.-Nagy dilation to a density matrix."""

    input_density_matrix: ArrayLike
    ancilla_system_output: ArrayLike
    ancilla_zero_block: ArrayLike
    ancilla_one_block: ArrayLike
    p_success: float
    p_failure: float
    normalized_success_density: Optional[ArrayLike]
    target_density: ArrayLike
    density_error_frobenius: float
    scale_factor: float



def _as_complex_square_matrix(operator: ArrayLike) -> ArrayLike:
    arr = np.asarray(operator, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Operator must be a square matrix.")
    return arr



def _as_complex_state_vector(state: StateLike) -> ArrayLike:
    if Qobj is not None and isinstance(state, Qobj):
        if not state.isket:
            raise ValueError("Qobj input must be a ket state.")
        vec = np.asarray(state.full(), dtype=complex).reshape(-1)
    else:
        vec = np.asarray(state, dtype=complex).reshape(-1)
    return vec



def _maybe_to_qobj(vector: ArrayLike, dims: Optional[Sequence[Sequence[int]]]) -> StateLike:
    if dims is None:
        return vector
    if qt is None:
        raise ImportError("qutip is not installed, so Qobj output cannot be created.")
    return qt.Qobj(vector.reshape((-1, 1)), dims=dims)



def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0



def operator_norm_2(operator: ArrayLike) -> float:
    op = _as_complex_square_matrix(operator)
    return float(la.svdvals(op)[0])



def is_contraction(operator: ArrayLike, tol: float = 1e-12) -> bool:
    return operator_norm_2(operator) <= 1.0 + tol



def scale_to_contraction(operator: ArrayLike, tol: float = 1e-12) -> tuple[ArrayLike, float]:
    op = _as_complex_square_matrix(operator)
    smax = operator_norm_2(op)
    if smax <= 1.0 + tol:
        return op.copy(), 1.0
    return op / smax, smax



def _positive_semidefinite_sqrt(matrix: ArrayLike, tol: float = 1e-12) -> ArrayLike:
    """Return the Hermitian square root of a PSD matrix using eigendecomposition."""
    mat = _as_complex_square_matrix(matrix)
    mat = 0.5 * (mat + mat.conj().T)
    evals, evecs = la.eigh(mat)
    evals = np.where(evals < tol, 0.0, evals)
    if np.any(evals < -tol):
        raise ValueError("Matrix is not positive semidefinite within tolerance.")
    root = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
    return 0.5 * (root + root.conj().T)



def defect_operator_right(operator: ArrayLike, tol: float = 1e-12) -> ArrayLike:
    """Return D_K = sqrt(I - K^\dagger K)."""
    op = _as_complex_square_matrix(operator)
    n = op.shape[0]
    return _positive_semidefinite_sqrt(np.eye(n, dtype=complex) - op.conj().T @ op, tol=tol)



def defect_operator_left(operator: ArrayLike, tol: float = 1e-12) -> ArrayLike:
    """Return D_{K*} = sqrt(I - K K^\dagger)."""
    op = _as_complex_square_matrix(operator)
    n = op.shape[0]
    return _positive_semidefinite_sqrt(np.eye(n, dtype=complex) - op @ op.conj().T, tol=tol)



def build_sz_nagy_dilation(
    operator: ArrayLike,
    *,
    auto_scale: bool = True,
    tol: float = 1e-12,
) -> SzNagyDilation:
    """Build the direct Sz.-Nagy one-ancilla dilation for a square operator.

    Parameters
    ----------
    operator:
        Square complex matrix K.
    auto_scale:
        If True and ||K||_2 > 1, rescale by the spectral norm so the effective
        operator is a contraction.
    tol:
        Numerical tolerance.
    """
    op = _as_complex_square_matrix(operator)

    if auto_scale:
        scaled, scale_factor = scale_to_contraction(op, tol=tol)
    else:
        scaled = op.copy()
        scale_factor = 1.0
        if not is_contraction(scaled, tol=tol):
            raise ValueError("Operator is not a contraction. Set auto_scale=True or rescale manually.")

    defect_r = defect_operator_right(scaled, tol=tol)
    defect_l = defect_operator_left(scaled, tol=tol)
    unitary = np.block([
        [scaled, defect_l],
        [defect_r, -scaled.conj().T],
    ])

    ident = np.eye(unitary.shape[0], dtype=complex)
    if not np.allclose(unitary.conj().T @ unitary, ident, atol=1e-10):
        raise ValueError("Constructed Sz.-Nagy dilation is not unitary within tolerance.")

    return SzNagyDilation(
        original_operator=op,
        scaled_operator=scaled,
        scale_factor=float(scale_factor),
        defect_right=defect_r,
        defect_left=defect_l,
        unitary=unitary,
    )



def ancilla_zero_state(num_system_levels: int) -> ArrayLike:
    anc = np.array([1.0, 0.0], dtype=complex)
    sys0 = np.zeros(num_system_levels, dtype=complex)
    sys0[0] = 1.0
    return np.kron(anc, sys0)



def apply_sz_nagy_dilation_to_state(
    dilation: SzNagyDilation,
    state: StateLike,
) -> StatePostselectionResult:
    """Apply the dilation to |0>\otimes|psi> and postselect ancilla |0>."""
    psi = _as_complex_state_vector(state)
    n = dilation.dimension
    if psi.shape[0] != n:
        raise ValueError("State dimension does not match dilation operator dimension.")

    input_embedded = np.concatenate([psi, np.zeros_like(psi)])
    output = dilation.unitary @ input_embedded

    zero_branch = output[:n]
    one_branch = output[n:]
    p_success = float(np.vdot(zero_branch, zero_branch).real)
    p_failure = float(np.vdot(one_branch, one_branch).real)

    normalized = None
    if p_success > 0.0:
        normalized = zero_branch / np.sqrt(p_success)

    target = dilation.operator @ psi
    err = float(np.linalg.norm(zero_branch - target))

    return StatePostselectionResult(
        input_state=psi,
        ancilla_system_output=output,
        ancilla_zero_branch=zero_branch,
        ancilla_one_branch=one_branch,
        p_success=p_success,
        p_failure=p_failure,
        normalized_success_state=normalized,
        target_state=target,
        state_error_norm=err,
        scale_factor=dilation.scale_factor,
    )



def apply_sz_nagy_dilation_to_density_matrix(
    dilation: SzNagyDilation,
    density_matrix: ArrayLike,
) -> DensityPostselectionResult:
    """Apply the dilation to |0><0|\otimes rho and extract the ancilla blocks."""
    rho = _as_complex_square_matrix(density_matrix)
    n = dilation.dimension
    if rho.shape != (n, n):
        raise ValueError("Density matrix dimension does not match dilation operator dimension.")

    rho = 0.5 * (rho + rho.conj().T)
    if not np.allclose(np.trace(rho).imag, 0.0, atol=1e-10):
        raise ValueError("Density matrix trace must be real within tolerance.")

    anc0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    embedded = np.kron(anc0, rho)
    output = dilation.unitary @ embedded @ dilation.unitary.conj().T

    zero_block = output[:n, :n]
    one_block = output[n:, n:]
    p_success = float(np.trace(zero_block).real)
    p_failure = float(np.trace(one_block).real)

    normalized = None
    if p_success > 0.0:
        normalized = zero_block / p_success

    target = dilation.operator @ rho @ dilation.operator.conj().T
    err = float(la.norm(zero_block - target, ord="fro"))

    return DensityPostselectionResult(
        input_density_matrix=rho,
        ancilla_system_output=output,
        ancilla_zero_block=zero_block,
        ancilla_one_block=one_block,
        p_success=p_success,
        p_failure=p_failure,
        normalized_success_density=normalized,
        target_density=target,
        density_error_frobenius=err,
        scale_factor=dilation.scale_factor,
    )



def normalized_density_matrix_from_state(state: StateLike) -> ArrayLike:
    psi = _as_complex_state_vector(state)
    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise ValueError("State vector cannot be the zero vector.")
    psi = psi / norm
    return np.outer(psi, psi.conj())



def partial_trace_ancilla_two_block(output_density: ArrayLike) -> ArrayLike:
    """Trace out the ancilla for a 2N x 2N density matrix written in 2x2 blocks."""
    rho = _as_complex_square_matrix(output_density)
    if rho.shape[0] % 2 != 0:
        raise ValueError("Output density matrix dimension must be even.")
    n = rho.shape[0] // 2
    return rho[:n, :n] + rho[n:, n:]



def ancilla_success_projector(num_system_levels: int) -> ArrayLike:
    """Projector |0><0| \otimes I_system."""
    return np.block([
        [np.eye(num_system_levels, dtype=complex), np.zeros((num_system_levels, num_system_levels), dtype=complex)],
        [np.zeros((num_system_levels, num_system_levels), dtype=complex), np.zeros((num_system_levels, num_system_levels), dtype=complex)],
    ])
