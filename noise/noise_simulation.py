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


DensityLike = Union[np.ndarray, "Qobj"]


@dataclass(frozen=True)
class SimpleNISQNoiseParameters:
    """A compact phenomenological noise model for the one-ancilla workflow.

    This is intentionally simple rather than hardware-exact. It captures the
    dominant effects relevant to the paper's NISQ setting:
      - T1 relaxation (amplitude damping),
      - T2 decoherence via pure dephasing,
      - effective depolarizing gate noise,
      - classical readout assignment error.

    All times are in the same user-chosen units, for example microseconds.
    """

    t1_system: Optional[float] = None
    t2_system: Optional[float] = None
    t1_ancilla: Optional[float] = None
    t2_ancilla: Optional[float] = None
    one_qubit_gate_time: float = 0.0
    two_qubit_gate_time: float = 0.0
    p1q_depolarizing: float = 0.0
    p2q_depolarizing: float = 0.0
    readout_p0_to_1_system: float = 0.0
    readout_p1_to_0_system: float = 0.0
    readout_p0_to_1_ancilla: float = 0.0
    readout_p1_to_0_ancilla: float = 0.0

    def system_assignment_matrix(self) -> np.ndarray:
        return single_qubit_assignment_matrix(
            self.readout_p0_to_1_system,
            self.readout_p1_to_0_system,
        )

    def ancilla_assignment_matrix(self) -> np.ndarray:
        return single_qubit_assignment_matrix(
            self.readout_p0_to_1_ancilla,
            self.readout_p1_to_0_ancilla,
        )


@dataclass(frozen=True)
class NoisyMeasurementResult:
    """Container for noisy binary measurement probabilities and counts."""

    ideal_probabilities: np.ndarray
    observed_probabilities: np.ndarray
    assignment_matrix: np.ndarray
    counts: Optional[dict[str, int]]
    shots: Optional[int]


def _as_density_matrix(rho_or_state: DensityLike) -> np.ndarray:
    if Qobj is not None and isinstance(rho_or_state, Qobj):
        if rho_or_state.isket:
            vec = np.asarray(rho_or_state.full(), dtype=complex).reshape(-1)
            return np.outer(vec, vec.conj())
        arr = np.asarray(rho_or_state.full(), dtype=complex)
    else:
        arr = np.asarray(rho_or_state, dtype=complex)
        if arr.ndim == 1:
            vec = arr.reshape(-1)
            return np.outer(vec, vec.conj())

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input must be a state vector or a square density matrix.")
    return arr


def _as_qobj_if_requested(arr: np.ndarray, as_qobj: bool) -> DensityLike:
    if not as_qobj:
        return arr
    if qt is None:
        raise ImportError("qutip is not installed, so Qobj output cannot be created.")
    return qt.Qobj(arr)


def _hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)


def pauli_matrices() -> dict[str, np.ndarray]:
    return {
        "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }


def kron_n(operators: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for op in operators:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def expand_single_qubit_operator(
    operator: np.ndarray,
    *,
    target: int,
    num_qubits: int,
) -> np.ndarray:
    """Embed a one-qubit operator into an n-qubit Hilbert space.

    Qubit indices are ordered left-to-right in tensor products:
        target = 0  -> operator ⊗ I ⊗ ...
        target = n-1 -> I ⊗ ... ⊗ operator
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    if not (0 <= target < num_qubits):
        raise ValueError("target must satisfy 0 <= target < num_qubits.")

    ops = [pauli_matrices()["I"] for _ in range(num_qubits)]
    ops[target] = np.asarray(operator, dtype=complex)
    return kron_n(ops)


def apply_kraus_channel(
    rho: DensityLike,
    kraus_operators: Sequence[np.ndarray],
    *,
    renormalize: bool = False,
    as_qobj: bool = False,
) -> DensityLike:
    """Apply rho -> sum_k E_k rho E_k^dagger."""
    arr = _as_density_matrix(rho)
    out = np.zeros_like(arr, dtype=complex)
    for ek in kraus_operators:
        e = np.asarray(ek, dtype=complex)
        out += e @ arr @ e.conj().T

    out = _hermitize(out)
    if renormalize:
        tr = np.real(np.trace(out))
        if tr <= 0:
            raise ValueError("Cannot renormalize a matrix with nonpositive trace.")
        out = out / tr
    return _as_qobj_if_requested(out, as_qobj)


def apply_local_kraus_channel(
    rho: DensityLike,
    kraus_operators: Sequence[np.ndarray],
    *,
    target: int,
    num_qubits: int,
    renormalize: bool = False,
    as_qobj: bool = False,
) -> DensityLike:
    """Apply a one-qubit channel to one target qubit of an n-qubit density matrix."""
    full_kraus = [
        expand_single_qubit_operator(np.asarray(ek, dtype=complex), target=target, num_qubits=num_qubits)
        for ek in kraus_operators
    ]
    return apply_kraus_channel(rho, full_kraus, renormalize=renormalize, as_qobj=as_qobj)


def amplitude_damping_probability(duration: float, t1: Optional[float]) -> float:
    """Return gamma = 1 - exp(-t / T1)."""
    if t1 is None or np.isinf(t1):
        return 0.0
    if duration < 0 or t1 <= 0:
        raise ValueError("duration must be >= 0 and T1 must be > 0.")
    return float(1.0 - np.exp(-duration / t1))


def pure_dephasing_time(t1: Optional[float], t2: Optional[float], tol: float = 1e-12) -> float:
    """Infer Tphi from T1 and T2 using 1/T2 = 1/(2T1) + 1/Tphi.

    If T1 is infinite or unspecified, then Tphi = T2.
    If T2 = 2*T1 (within tolerance), then Tphi is taken to be infinite.
    """
    if t2 is None or np.isinf(t2):
        return np.inf
    if t2 <= 0:
        raise ValueError("T2 must be positive.")
    if t1 is None or np.isinf(t1):
        return float(t2)
    if t1 <= 0:
        raise ValueError("T1 must be positive.")

    rate = (1.0 / t2) - (1.0 / (2.0 * t1))
    if rate < -tol:
        raise ValueError("Unphysical inputs: T2 must satisfy T2 <= 2*T1 for Markovian qubit noise.")
    if abs(rate) <= tol:
        return np.inf
    return float(1.0 / rate)


def phase_damping_probability(duration: float, tphi: Optional[float]) -> float:
    """Return lambda = 1 - exp(-t / Tphi) for the pure-dephasing part."""
    if tphi is None or np.isinf(tphi):
        return 0.0
    if duration < 0 or tphi <= 0:
        raise ValueError("duration must be >= 0 and Tphi must be > 0.")
    return float(1.0 - np.exp(-duration / tphi))


def amplitude_phase_damping_probabilities(
    duration: float,
    *,
    t1: Optional[float],
    t2: Optional[float],
    tol: float = 1e-12,
) -> tuple[float, float]:
    """Return (gamma, lambda) for amplitude + pure-dephasing noise.

    The formulas are equivalent to the standard amplitude-plus-phase-damping
    parameterization used for superconducting-qubit channel models:

        gamma  = 1 - exp(-t / T1)
        lambda = 1 - exp(t / T1 - 2 t / T2)

    where lambda is the *pure* dephasing contribution after removing the T1
    contribution from T2.
    """
    gamma = amplitude_damping_probability(duration, t1)

    if t2 is None or np.isinf(t2):
        return gamma, 0.0
    if t1 is None or np.isinf(t1):
        return gamma, phase_damping_probability(duration, t2)

    if duration < 0 or t1 <= 0 or t2 <= 0:
        raise ValueError("duration must be >= 0 and T1, T2 must be > 0.")
    exponent = (duration / t1) - (2.0 * duration / t2)
    if exponent > tol:
        raise ValueError("Unphysical inputs: T2 must satisfy T2 <= 2*T1 for Markovian qubit noise.")
    lam = float(1.0 - np.exp(exponent))
    lam = min(max(lam, 0.0), 1.0)
    return gamma, lam


def amplitude_damping_kraus(
    gamma: float,
    *,
    excited_state_population: float = 0.0,
) -> list[np.ndarray]:
    """Generalized amplitude-damping Kraus operators.

    For excited_state_population = 0 this reduces to ordinary T1 relaxation.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must satisfy 0 <= gamma <= 1.")
    if not (0.0 <= excited_state_population <= 1.0):
        raise ValueError("excited_state_population must satisfy 0 <= p <= 1.")

    a = float(gamma)
    p1 = float(excited_state_population)

    a0 = np.sqrt(1.0 - p1) * np.array(
        [[1.0, 0.0], [0.0, np.sqrt(1.0 - a)]], dtype=complex
    )
    a1 = np.sqrt(1.0 - p1) * np.array(
        [[0.0, np.sqrt(a)], [0.0, 0.0]], dtype=complex
    )
    b0 = np.sqrt(p1) * np.array(
        [[np.sqrt(1.0 - a), 0.0], [0.0, 1.0]], dtype=complex
    )
    b1 = np.sqrt(p1) * np.array(
        [[0.0, 0.0], [np.sqrt(a), 0.0]], dtype=complex
    )
    return [a0, a1, b0, b1]


def phase_damping_kraus(lam: float) -> list[np.ndarray]:
    """Pure phase-damping Kraus operators.

    This channel leaves populations unchanged and damps off-diagonal elements
    by a factor (1 - lambda).
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lambda must satisfy 0 <= lambda <= 1.")
    return [
        np.sqrt(1.0 - lam) * np.eye(2, dtype=complex),
        np.sqrt(lam) * np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
        np.sqrt(lam) * np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
    ]


def single_qubit_depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Single-qubit depolarizing channel.

    rho -> (1-p) rho + p * I/2.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must satisfy 0 <= p <= 1.")
    paulis = pauli_matrices()
    if p == 0.0:
        return [np.eye(2, dtype=complex)]
    return [
        np.sqrt(1.0 - p) * paulis["I"],
        np.sqrt(p / 3.0) * paulis["X"],
        np.sqrt(p / 3.0) * paulis["Y"],
        np.sqrt(p / 3.0) * paulis["Z"],
    ]


def two_qubit_depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Two-qubit depolarizing channel on the full 4x4 space.

    rho -> (1-p) rho + p * I/4.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must satisfy 0 <= p <= 1.")
    if p == 0.0:
        return [np.eye(4, dtype=complex)]

    paulis = pauli_matrices()
    labels = ["I", "X", "Y", "Z"]
    non_identity = []
    for a in labels:
        for b in labels:
            if a == "I" and b == "I":
                continue
            non_identity.append(np.kron(paulis[a], paulis[b]))

    kraus = [np.sqrt(1.0 - p) * np.eye(4, dtype=complex)]
    kraus.extend(np.sqrt(p / 15.0) * op for op in non_identity)
    return kraus


def compose_kraus_channels(
    second: Sequence[np.ndarray],
    first: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Return Kraus operators for applying `first` then `second`."""
    return [np.asarray(e2) @ np.asarray(e1) for e2 in second for e1 in first]


def amplitude_phase_damping_kraus(
    duration: float,
    *,
    t1: Optional[float],
    t2: Optional[float],
    excited_state_population: float = 0.0,
) -> list[np.ndarray]:
    """Combined relaxation + pure dephasing channel via sequential composition."""
    gamma, lam = amplitude_phase_damping_probabilities(duration, t1=t1, t2=t2)
    ad = amplitude_damping_kraus(gamma, excited_state_population=excited_state_population)
    pd = phase_damping_kraus(lam)
    return compose_kraus_channels(pd, ad)


def apply_amplitude_phase_damping(
    rho: DensityLike,
    *,
    target: int,
    num_qubits: int,
    duration: float,
    t1: Optional[float],
    t2: Optional[float],
    excited_state_population: float = 0.0,
    renormalize: bool = False,
    as_qobj: bool = False,
) -> DensityLike:
    """Apply a local T1/T2 channel to one qubit in an n-qubit state."""
    kraus = amplitude_phase_damping_kraus(
        duration,
        t1=t1,
        t2=t2,
        excited_state_population=excited_state_population,
    )
    return apply_local_kraus_channel(
        rho,
        kraus,
        target=target,
        num_qubits=num_qubits,
        renormalize=renormalize,
        as_qobj=as_qobj,
    )


def apply_local_depolarizing(
    rho: DensityLike,
    *,
    target: int,
    num_qubits: int,
    probability: float,
    renormalize: bool = False,
    as_qobj: bool = False,
) -> DensityLike:
    kraus = single_qubit_depolarizing_kraus(probability)
    return apply_local_kraus_channel(
        rho,
        kraus,
        target=target,
        num_qubits=num_qubits,
        renormalize=renormalize,
        as_qobj=as_qobj,
    )


def apply_global_two_qubit_depolarizing(
    rho: DensityLike,
    *,
    probability: float,
    renormalize: bool = False,
    as_qobj: bool = False,
) -> DensityLike:
    """Apply a two-qubit depolarizing channel to a full 4x4 density matrix."""
    arr = _as_density_matrix(rho)
    if arr.shape != (4, 4):
        raise ValueError("apply_global_two_qubit_depolarizing expects a full 2-qubit density matrix.")
    return apply_kraus_channel(
        arr,
        two_qubit_depolarizing_kraus(probability),
        renormalize=renormalize,
        as_qobj=as_qobj,
    )


def computational_basis_probabilities(rho: DensityLike) -> np.ndarray:
    arr = _as_density_matrix(rho)
    probs = np.real(np.diag(arr)).astype(float)
    probs = np.maximum(probs, 0.0)
    total = np.sum(probs)
    if total <= 0:
        raise ValueError("State has nonpositive total probability in the computational basis.")
    return probs / total


def single_qubit_assignment_matrix(p0_to_1: float, p1_to_0: float) -> np.ndarray:
    """Return the 2x2 assignment matrix A with p_obs = A p_true.

    Ordering is [0, 1], so
        A[obs, true] = P(obs | true).
    """
    if not (0.0 <= p0_to_1 <= 1.0 and 0.0 <= p1_to_0 <= 1.0):
        raise ValueError("Readout error probabilities must lie in [0, 1].")
    return np.array(
        [
            [1.0 - p0_to_1, p1_to_0],
            [p0_to_1, 1.0 - p1_to_0],
        ],
        dtype=float,
    )


def apply_assignment_matrix(probabilities: Sequence[float], assignment_matrix: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    amat = np.asarray(assignment_matrix, dtype=float)
    if amat.shape != (probs.size, probs.size):
        raise ValueError("assignment_matrix must be square and match the probability vector length.")
    noisy = amat @ probs
    noisy = np.maximum(noisy, 0.0)
    total = float(np.sum(noisy))
    if total <= 0:
        raise ValueError("Observed probabilities have nonpositive total weight.")
    return noisy / total


def sample_counts(
    probabilities: Sequence[float],
    shots: int,
    *,
    labels: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, int]:
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    probs = np.maximum(probs, 0.0)
    probs = probs / np.sum(probs)
    samples = rng.multinomial(shots, probs)

    if labels is None:
        width = int(np.log2(probs.size)) if probs.size > 1 else 1
        labels = [format(i, f"0{width}b") for i in range(probs.size)]
    labels = list(labels)
    if len(labels) != probs.size:
        raise ValueError("labels must have the same length as probabilities.")
    return {label: int(count) for label, count in zip(labels, samples)}


def noisy_single_qubit_measurement(
    probabilities: Sequence[float],
    *,
    p0_to_1: float,
    p1_to_0: float,
    shots: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> NoisyMeasurementResult:
    """Apply classical readout error to a one-qubit probability vector.

    The input probabilities are assumed to be ordered as [P(0), P(1)].
    """
    ideal = np.asarray(probabilities, dtype=float).reshape(2)
    ideal = ideal / np.sum(ideal)
    amat = single_qubit_assignment_matrix(p0_to_1, p1_to_0)
    observed = apply_assignment_matrix(ideal, amat)
    counts = None
    if shots is not None:
        counts = sample_counts(observed, shots, labels=["0", "1"], rng=rng)
    return NoisyMeasurementResult(
        ideal_probabilities=ideal,
        observed_probabilities=observed,
        assignment_matrix=amat,
        counts=counts,
        shots=shots,
    )


def apply_default_noise_for_one_ancilla_step(
    rho_two_qubit: DensityLike,
    params: SimpleNISQNoiseParameters,
    *,
    apply_one_qubit_decoherence: bool = True,
    apply_two_qubit_depolarizing: bool = True,
    apply_one_qubit_depolarizing: bool = True,
    renormalize: bool = False,
    as_qobj: bool = False,
) -> DensityLike:
    """A convenient default noise layer for the paper's 2-qubit workflow.

    The intended interpretation is:
      1. one-qubit decoherence on ancilla and system during their active/idle time,
      2. optional effective one-qubit depolarizing noise,
      3. optional effective two-qubit depolarizing noise for the entangling step.

    This is a compact phenomenological layer for simulation rather than a full
    pulse-level device model.
    """
    arr = _as_density_matrix(rho_two_qubit)
    if arr.shape != (4, 4):
        raise ValueError("This helper expects a two-qubit density matrix.")

    out: DensityLike = arr

    if apply_one_qubit_decoherence:
        out = apply_amplitude_phase_damping(
            out,
            target=0,
            num_qubits=2,
            duration=params.two_qubit_gate_time,
            t1=params.t1_ancilla,
            t2=params.t2_ancilla,
            renormalize=renormalize,
        )
        out = apply_amplitude_phase_damping(
            out,
            target=1,
            num_qubits=2,
            duration=params.two_qubit_gate_time,
            t1=params.t1_system,
            t2=params.t2_system,
            renormalize=renormalize,
        )

    if apply_one_qubit_depolarizing and params.p1q_depolarizing > 0.0:
        out = apply_local_depolarizing(
            out,
            target=0,
            num_qubits=2,
            probability=params.p1q_depolarizing,
            renormalize=renormalize,
        )
        out = apply_local_depolarizing(
            out,
            target=1,
            num_qubits=2,
            probability=params.p1q_depolarizing,
            renormalize=renormalize,
        )

    if apply_two_qubit_depolarizing and params.p2q_depolarizing > 0.0:
        out = apply_global_two_qubit_depolarizing(
            out,
            probability=params.p2q_depolarizing,
            renormalize=renormalize,
        )

    out_arr = _as_density_matrix(out)
    return _as_qobj_if_requested(out_arr, as_qobj)


def frobenius_distance(rho_a: DensityLike, rho_b: DensityLike) -> float:
    a = _as_density_matrix(rho_a)
    b = _as_density_matrix(rho_b)
    return float(la.norm(a - b, ord="fro"))
