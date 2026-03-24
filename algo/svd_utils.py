from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.linalg as la


ArrayLike = np.ndarray


@dataclass(frozen=True)
class SVDDecomposition:
    """Container for an operator SVD.

    Attributes
    ----------
    operator:
        Original operator M.
    U:
        Left singular vectors.
    singular_values:
        Nonnegative singular values.
    Vh:
        Conjugate-transpose of right singular vectors.
    scale_factor:
        Scaling used to turn M into a contraction. The effective
        operator implemented by the dilation algorithm is M / scale_factor.
    scaled_operator:
        M / scale_factor.
    """

    operator: ArrayLike
    U: ArrayLike
    singular_values: ArrayLike
    Vh: ArrayLike
    scale_factor: float
    scaled_operator: ArrayLike

    @property
    def Sigma(self) -> ArrayLike:
        return np.diag(self.singular_values)

    @property
    def V(self) -> ArrayLike:
        return self.Vh.conj().T

    def reconstruct(self) -> ArrayLike:
        return self.U @ self.Sigma @ self.Vh



def _as_complex_array(operator: ArrayLike) -> ArrayLike:
    arr = np.asarray(operator, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Operator must be a square 2D array.")
    return arr



def operator_norm_2(operator: ArrayLike) -> float:
    """Return the spectral norm (largest singular value)."""
    op = _as_complex_array(operator)
    return float(la.svdvals(op)[0])



def is_contraction(operator: ArrayLike, tol: float = 1e-12) -> bool:
    """Return True if ||operator||_2 <= 1 within tolerance."""
    return operator_norm_2(operator) <= 1.0 + tol



def scale_to_contraction(operator: ArrayLike, tol: float = 1e-12) -> Tuple[ArrayLike, float]:
    """Scale an operator by its largest singular value when needed.

    The Schlimgen et al. algorithm requires the diagonal singular-value
    operator to have entries with magnitude <= 1. For a general operator M,
    dividing by the largest singular value ensures this condition.
    """
    op = _as_complex_array(operator)
    smax = operator_norm_2(op)
    if smax <= 1.0 + tol:
        return op.copy(), 1.0
    return op / smax, smax



def compute_operator_svd(
    operator: ArrayLike,
    *,
    auto_scale: bool = True,
    tol: float = 1e-12,
) -> SVDDecomposition:
    """Compute an SVD suitable for the one-ancilla dilation algorithm.

    Parameters
    ----------
    operator:
        Square complex matrix M.
    auto_scale:
        If True, divide M by its largest singular value whenever needed so the
        effective operator is a contraction, as described in the paper.
    tol:
        Numerical tolerance.
    """
    op = _as_complex_array(operator)

    if auto_scale:
        scaled, scale_factor = scale_to_contraction(op, tol=tol)
    else:
        scaled = op.copy()
        scale_factor = 1.0
        if not is_contraction(scaled, tol=tol):
            raise ValueError(
                "Operator is not a contraction. Set auto_scale=True or rescale manually."
            )

    U, svals, Vh = la.svd(scaled, full_matrices=True, lapack_driver="gesvd")

    # Guard against tiny numerical overshoots above 1.
    svals = np.clip(np.real_if_close(svals), 0.0, 1.0)

    return SVDDecomposition(
        operator=op,
        U=U,
        singular_values=np.asarray(svals, dtype=float),
        Vh=Vh,
        scale_factor=float(scale_factor),
        scaled_operator=scaled,
    )



def reconstruct_from_svd(U: ArrayLike, singular_values: ArrayLike, Vh: ArrayLike) -> ArrayLike:
    """Reconstruct U diag(s) Vh."""
    U = np.asarray(U, dtype=complex)
    s = np.asarray(singular_values, dtype=float)
    Vh = np.asarray(Vh, dtype=complex)
    return U @ np.diag(s) @ Vh
