from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import scipy.linalg as la

from .svd_utils import SVDDecomposition, compute_operator_svd

try:
    import qutip as qt
    from qutip import Qobj
except Exception:  # pragma: no cover - optional dependency
    qt = None
    Qobj = None


ArrayLike = np.ndarray
StateLike = Union[np.ndarray, "Qobj"]


@dataclass(frozen=True)
class DiagonalDilation:
    """One-ancilla diagonal dilation data.

    For a diagonal operator Sigma = diag(sigma_j), this stores the two unit-
    modulus diagonal blocks Sigma_plus and Sigma_minus such that

        (Sigma_plus + Sigma_minus) / 2 = Sigma,

    and the dilated unitary is

        U_Sigma = Sigma_plus ⊕ Sigma_minus.
    """

    diagonal: ArrayLike
    sigma_plus: ArrayLike
    sigma_minus: ArrayLike
    unitary_dilation: ArrayLike
    scale_factor: float

    @property
    def Sigma(self) -> ArrayLike:
        return np.diag(self.diagonal)

    @property
    def Sigma_plus(self) -> ArrayLike:
        return np.diag(self.sigma_plus)

    @property
    def Sigma_minus(self) -> ArrayLike:
        return np.diag(self.sigma_minus)


@dataclass(frozen=True)
class SVDOneAncillaDilation:
    """Full data for the SVD-based one-ancilla dilation algorithm."""

    original_operator: ArrayLike
    scaled_operator: ArrayLike
    scale_factor: float
    U: ArrayLike
    singular_values: ArrayLike
    Vh: ArrayLike
    diagonal_dilation: DiagonalDilation
    full_unitary: ArrayLike

    @property
    def Sigma(self) -> ArrayLike:
        return np.diag(self.singular_values)

    def reconstructed_scaled_operator(self) -> ArrayLike:
        return self.U @ self.Sigma @ self.Vh


@dataclass(frozen=True)
class PostselectionResult:
    """Result of simulating the dilation algorithm on a state vector."""

    input_state: ArrayLike
    ancilla_system_output: ArrayLike
    ancilla_zero_branch: ArrayLike
    ancilla_one_branch: ArrayLike
    p_success: float
    p_failure: float
    normalized_success_state: Optional[ArrayLike]
    target_scaled_state: ArrayLike
    state_error_norm: float
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



def hadamard() -> ArrayLike:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)



def projector(bit: int) -> ArrayLike:
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1.")
    p = np.zeros((2, 2), dtype=complex)
    p[bit, bit] = 1.0
    return p



def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0



def uniform_superposition(num_qubits: int) -> ArrayLike:
    dim = 2 ** num_qubits
    return np.ones(dim, dtype=complex) / np.sqrt(dim)



def unit_modulus_pair_from_sigma(sigma: complex, tol: float = 1e-12) -> tuple[complex, complex]:
    """Construct sigma_+, sigma_- for a scalar sigma with |sigma| <= 1.

    This implements Eq. (2) from Schlimgen et al. in a numerically stable form:

        sigma_± = exp(i arg sigma) * (|sigma| ± i sqrt(1 - |sigma|^2)).

    so that
        (sigma_+ + sigma_-) / 2 = sigma,
        |sigma_+| = |sigma_-| = 1.

    For sigma = 0 we choose (+i, -i).
    """
    mag = abs(sigma)
    if mag > 1.0 + tol:
        raise ValueError("Each diagonal entry must satisfy |sigma| <= 1.")

    mag = min(mag, 1.0)
    if mag < tol:
        return 1j, -1j

    phase = sigma / mag
    imag_part = np.sqrt(max(0.0, 1.0 - mag * mag))
    sigma_plus = phase * (mag + 1j * imag_part)
    sigma_minus = phase * (mag - 1j * imag_part)
    return sigma_plus, sigma_minus



def build_diagonal_dilation(
    diagonal_entries: Sequence[complex],
    *,
    auto_scale: bool = False,
    tol: float = 1e-12,
) -> DiagonalDilation:
    """Build the one-ancilla unitary dilation of a diagonal operator.

    Parameters
    ----------
    diagonal_entries:
        Entries of Sigma = diag(sigma_j).
    auto_scale:
        If True and max_j |sigma_j| > 1, divide all entries by max_j |sigma_j|.
    tol:
        Numerical tolerance.
    """
    diagonal = np.asarray(diagonal_entries, dtype=complex).reshape(-1)
    if diagonal.size == 0:
        raise ValueError("diagonal_entries cannot be empty.")

    max_abs = float(np.max(np.abs(diagonal)))
    scale_factor = 1.0
    if max_abs > 1.0 + tol:
        if not auto_scale:
            raise ValueError(
                "Diagonal entries exceed unit magnitude. Set auto_scale=True or rescale manually."
            )
        scale_factor = max_abs
        diagonal = diagonal / scale_factor

    sigma_plus = np.empty_like(diagonal)
    sigma_minus = np.empty_like(diagonal)
    for idx, sigma in enumerate(diagonal):
        sp, sm = unit_modulus_pair_from_sigma(sigma, tol=tol)
        sigma_plus[idx] = sp
        sigma_minus[idx] = sm

    unitary_dilation = la.block_diag(np.diag(sigma_plus), np.diag(sigma_minus))

    if not np.allclose(
        unitary_dilation.conj().T @ unitary_dilation,
        np.eye(unitary_dilation.shape[0], dtype=complex),
        atol=1e-10,
    ):
        raise RuntimeError("Constructed dilation is not unitary within tolerance.")

    return DiagonalDilation(
        diagonal=diagonal,
        sigma_plus=sigma_plus,
        sigma_minus=sigma_minus,
        unitary_dilation=unitary_dilation,
        scale_factor=float(scale_factor),
    )



def build_state_preparation_dilation(
    target_amplitudes: Sequence[complex],
    *,
    auto_scale: bool = False,
    tol: float = 1e-12,
) -> DiagonalDilation:
    """Build the diagonal dilation used for probabilistic state preparation.

    Following Sec. II A of the paper, if the target state is

        |phi> = sum_j c_j |j>,

    then set Sigma = diag(c_j). Acting on a uniform superposition and then
    postselecting ancilla |0> yields a state proportional to |phi>.
    """
    coeffs = np.asarray(target_amplitudes, dtype=complex).reshape(-1)
    return build_diagonal_dilation(coeffs, auto_scale=auto_scale, tol=tol)



def build_one_ancilla_svd_dilation(
    operator: ArrayLike,
    *,
    auto_scale: bool = True,
    tol: float = 1e-12,
) -> SVDOneAncillaDilation:
    """Build the full one-ancilla unitary implementing the SVD dilation algorithm.

    If M = U Sigma V^dagger, the algorithm applies, in order,

        H_anc -> (I_anc ⊗ V^dagger) -> U_Sigma -> (I_anc ⊗ U) -> H_anc,

    where U_Sigma = Sigma_plus ⊕ Sigma_minus is the diagonal one-ancilla
    dilation of the singular-value matrix.
    """
    op = _as_complex_square_matrix(operator)
    svd: SVDDecomposition = compute_operator_svd(op, auto_scale=auto_scale, tol=tol)

    diagonal_dilation = build_diagonal_dilation(
        svd.singular_values,
        auto_scale=False,
        tol=tol,
    )

    dim = op.shape[0]
    H_anc = np.kron(hadamard(), np.eye(dim, dtype=complex))
    U_block = np.kron(np.eye(2, dtype=complex), svd.U)
    Vh_block = np.kron(np.eye(2, dtype=complex), svd.Vh)

    full_unitary = H_anc @ U_block @ diagonal_dilation.unitary_dilation @ Vh_block @ H_anc

    if not np.allclose(
        full_unitary.conj().T @ full_unitary,
        np.eye(full_unitary.shape[0], dtype=complex),
        atol=1e-10,
    ):
        raise RuntimeError("Full one-ancilla dilation unitary failed the unitarity check.")

    return SVDOneAncillaDilation(
        original_operator=svd.operator,
        scaled_operator=svd.scaled_operator,
        scale_factor=svd.scale_factor,
        U=svd.U,
        singular_values=svd.singular_values,
        Vh=svd.Vh,
        diagonal_dilation=diagonal_dilation,
        full_unitary=full_unitary,
    )



def apply_one_ancilla_dilation_to_state(
    operator: ArrayLike,
    state: StateLike,
    *,
    auto_scale: bool = True,
    tol: float = 1e-12,
    return_qobj: bool = False,
) -> PostselectionResult:
    """Simulate the one-ancilla SVD dilation on a pure state.

    Parameters
    ----------
    operator:
        Square operator M.
    state:
        Ket vector |psi> as a numpy array or qutip.Qobj.
    auto_scale:
        Rescale M by its largest singular value if needed so the implemented
        operation is a contraction.
    return_qobj:
        If True and the input was a qutip ket, output branches are also returned
        as qutip kets through the dataclass fields.
    """
    op = _as_complex_square_matrix(operator)
    psi = _as_complex_state_vector(state)

    dim = op.shape[0]
    if psi.size != dim:
        raise ValueError("State dimension does not match operator dimension.")

    dilation = build_one_ancilla_svd_dilation(op, auto_scale=auto_scale, tol=tol)

    ancilla_zero = np.array([1.0, 0.0], dtype=complex)
    input_full = np.kron(ancilla_zero, psi)
    output_full = dilation.full_unitary @ input_full

    zero_branch = output_full[:dim]
    one_branch = output_full[dim:]

    p_success = float(np.vdot(zero_branch, zero_branch).real)
    p_failure = float(np.vdot(one_branch, one_branch).real)

    target_scaled_state = dilation.scaled_operator @ psi
    state_error_norm = float(np.linalg.norm(zero_branch - target_scaled_state))

    normalized_success_state = None
    if p_success > tol:
        normalized_success_state = zero_branch / np.sqrt(p_success)

    # Keep dataclass storage as numpy arrays for simplicity and portability.
    return PostselectionResult(
        input_state=psi,
        ancilla_system_output=output_full,
        ancilla_zero_branch=zero_branch,
        ancilla_one_branch=one_branch,
        p_success=p_success,
        p_failure=p_failure,
        normalized_success_state=normalized_success_state,
        target_scaled_state=target_scaled_state,
        state_error_norm=state_error_norm,
        scale_factor=dilation.scale_factor,
    )



def prepare_subnormalized_state_from_uniform_superposition(
    target_amplitudes: Sequence[complex],
    *,
    auto_scale: bool = False,
    tol: float = 1e-12,
) -> PostselectionResult:
    """Simulate the state-preparation circuit of Sec. II A.

    The target amplitudes c_j define Sigma = diag(c_j). Starting from the
    system uniform superposition and ancilla |0>, the ancilla-|0> branch equals
    Sigma |~> where |~> is the uniform superposition state.
    """
    coeffs = np.asarray(target_amplitudes, dtype=complex).reshape(-1)
    dim = coeffs.size
    if not is_power_of_two(dim):
        raise ValueError("Number of target amplitudes must be a power of 2.")

    diagonal_dilation = build_state_preparation_dilation(
        coeffs, auto_scale=auto_scale, tol=tol
    )

    sys_state = uniform_superposition(int(np.log2(dim)))
    H_anc = np.kron(hadamard(), np.eye(dim, dtype=complex))
    ancilla_zero = np.array([1.0, 0.0], dtype=complex)
    input_full = np.kron(ancilla_zero, sys_state)
    output_full = H_anc @ diagonal_dilation.unitary_dilation @ H_anc @ input_full

    zero_branch = output_full[:dim]
    one_branch = output_full[dim:]

    p_success = float(np.vdot(zero_branch, zero_branch).real)
    p_failure = float(np.vdot(one_branch, one_branch).real)

    target_state = np.diag(diagonal_dilation.diagonal) @ sys_state
    state_error_norm = float(np.linalg.norm(zero_branch - target_state))

    normalized_success_state = None
    if p_success > tol:
        normalized_success_state = zero_branch / np.sqrt(p_success)

    return PostselectionResult(
        input_state=sys_state,
        ancilla_system_output=output_full,
        ancilla_zero_branch=zero_branch,
        ancilla_one_branch=one_branch,
        p_success=p_success,
        p_failure=p_failure,
        normalized_success_state=normalized_success_state,
        target_scaled_state=target_state,
        state_error_norm=state_error_norm,
        scale_factor=diagonal_dilation.scale_factor,
    )



def density_matrix_from_state(state: StateLike) -> ArrayLike:
    vec = _as_complex_state_vector(state)
    return np.outer(vec, vec.conj())



def to_qobj_ket(vector: ArrayLike, *, dims: Optional[Sequence[Sequence[int]]] = None) -> "Qobj":
    if qt is None:
        raise ImportError("qutip is not installed.")
    vec = np.asarray(vector, dtype=complex).reshape((-1, 1))
    if dims is None:
        n = vec.shape[0]
        dims = [[n], [1]]
    return qt.Qobj(vec, dims=dims)
