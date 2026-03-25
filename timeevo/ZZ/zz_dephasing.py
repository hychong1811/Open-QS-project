from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from algo.dilation_algo import build_diagonal_dilation, hadamard
from noise.noise_simulation import (
    SimpleNISQNoiseParameters,
    apply_default_noise_for_one_ancilla_step,
    noisy_single_qubit_measurement,
)
from plot.bloch_trajectory_plot import BlochTrajectory, make_bloch_trajectory_from_density_matrices
from plot.coherence_plot import CoherenceSeries, make_coherence_series_from_density_matrices


from metric.tomography import (
    SingleQubitTomographyResult,
    basis_probabilities,
    expectation_from_counts,
    linear_inversion_density_matrix,
    perform_single_qubit_tomography,
    project_to_physical_density_matrix,
)


ArrayLike = Union[Sequence[complex], np.ndarray]


@dataclass(frozen=True)
class AnalyticSVDDilation:
    label: str
    time: float
    kraus_operator: np.ndarray
    U: np.ndarray
    singular_values: np.ndarray
    Vh: np.ndarray
    diagonal_dilation: np.ndarray
    full_unitary: np.ndarray

    @property
    def Sigma(self) -> np.ndarray:
        return np.diag(self.singular_values)

    def reconstruct(self) -> np.ndarray:
        return self.U @ self.Sigma @ self.Vh


@dataclass(frozen=True)
class KrausBranchSimulationResult:
    label: str
    time: float
    exact_subdensity: np.ndarray
    obtained_subdensity: np.ndarray
    success_probability: float
    conditional_state_exact: np.ndarray
    conditional_state_for_tomography: np.ndarray
    tomography_result: SingleQubitTomographyResult
    full_two_qubit_output_density: np.ndarray


@dataclass(frozen=True)
class ZZDephasingTrajectoryResult:
    times: np.ndarray
    exact_density_matrices: np.ndarray
    obtained_density_matrices: np.ndarray
    coherence_exact: CoherenceSeries
    coherence_obtained: CoherenceSeries
    bloch_exact: BlochTrajectory
    bloch_obtained: BlochTrajectory
    branch_results: list[list[KrausBranchSimulationResult]]


def ket_plus() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)


def density_matrix_from_ket(ket: ArrayLike) -> np.ndarray:
    vec = np.asarray(ket, dtype=complex).reshape(-1)
    return np.outer(vec, vec.conj())


def paper_period(theta: float) -> float:
    if theta == 0.0:
        raise ValueError("theta must be nonzero.")
    return np.pi / abs(theta)


def paper_time_grid(theta: float = 0.5, *, periods: float = 1.0, num_points: int = 41) -> np.ndarray:
    return np.linspace(0.0, periods * paper_period(theta), int(num_points), dtype=float)


def zz_dephasing_kraus_operators(
    time: float,
    *,
    theta: float = 0.5,
    lambda0: float = 0.7,
    lambda1: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isclose(lambda0 + lambda1, 1.0, atol=1e-12):
        raise ValueError("lambda0 and lambda1 must satisfy lambda0 + lambda1 = 1.")
    phase = np.exp(1j * theta * float(time))
    phase_conj = np.conjugate(phase)
    k0 = np.sqrt(lambda0) * np.array([[phase, 0.0], [0.0, phase_conj]], dtype=complex)
    k1 = np.sqrt(lambda1) * np.array([[phase_conj, 0.0], [0.0, phase]], dtype=complex)
    return k0, k1


def exact_zz_dephasing_density_matrix(
    time: float,
    *,
    initial_state: Optional[ArrayLike] = None,
    theta: float = 0.5,
    lambda0: float = 0.7,
    lambda1: float = 0.3,
) -> np.ndarray:
    if initial_state is None:
        initial_state = ket_plus()
    rho0 = density_matrix_from_ket(initial_state)
    kraus_ops = zz_dephasing_kraus_operators(time, theta=theta, lambda0=lambda0, lambda1=lambda1)
    return sum(k @ rho0 @ k.conj().T for k in kraus_ops)


def analytic_svd_for_zz_kraus(
    time: float,
    *,
    branch: int,
    theta: float = 0.5,
    lambda0: float = 0.7,
    lambda1: float = 0.3,
) -> AnalyticSVDDilation:
    k0, k1 = zz_dephasing_kraus_operators(time, theta=theta, lambda0=lambda0, lambda1=lambda1)
    if branch == 0:
        weight = float(lambda0)
        phase_diag = np.array([
            np.exp(1j * theta * float(time)),
            np.exp(-1j * theta * float(time)),
        ], dtype=complex)
        kraus = k0
        label = "K0"
    elif branch == 1:
        weight = float(lambda1)
        phase_diag = np.array([
            np.exp(-1j * theta * float(time)),
            np.exp(1j * theta * float(time)),
        ], dtype=complex)
        kraus = k1
        label = "K1"
    else:
        raise ValueError("branch must be 0 or 1.")

    singular_values = np.array([np.sqrt(weight), np.sqrt(weight)], dtype=float)
    U = np.diag(phase_diag)
    Vh = np.eye(2, dtype=complex)

    diagonal = build_diagonal_dilation(singular_values)
    H_anc = np.kron(hadamard(), np.eye(2, dtype=complex))
    U_block = np.kron(np.eye(2, dtype=complex), U)
    Vh_block = np.kron(np.eye(2, dtype=complex), Vh)
    full_unitary = H_anc @ U_block @ diagonal.unitary_dilation @ Vh_block @ H_anc

    return AnalyticSVDDilation(
        label=label,
        time=float(time),
        kraus_operator=kraus,
        U=U,
        singular_values=singular_values,
        Vh=Vh,
        diagonal_dilation=diagonal.unitary_dilation,
        full_unitary=full_unitary,
    )


def ancilla_zero_block(rho_two_qubit: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho_two_qubit, dtype=complex)
    if arr.shape != (4, 4):
        raise ValueError("rho_two_qubit must have shape (4, 4).")
    return arr[:2, :2]


def ancilla_one_block(rho_two_qubit: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho_two_qubit, dtype=complex)
    if arr.shape != (4, 4):
        raise ValueError("rho_two_qubit must have shape (4, 4).")
    return arr[2:, 2:]


def ancilla_success_probability(rho_two_qubit: np.ndarray) -> float:
    return float(np.real(np.trace(ancilla_zero_block(rho_two_qubit))))


def conditional_system_state_from_ancilla_zero(rho_two_qubit: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    block = ancilla_zero_block(rho_two_qubit)
    p0 = float(np.real(np.trace(block)))
    if p0 <= tol:
        raise ValueError("Ancilla-|0> success probability is too small to condition on.")
    return block / p0


def _hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)


def perform_single_qubit_tomography_with_optional_readout(
    rho: np.ndarray,
    *,
    trace_scale: float = 1.0,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    readout_p0_to_1: float = 0.0,
    readout_p1_to_0: float = 0.0,
    enforce_physical: bool = True,
) -> SingleQubitTomographyResult:
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must be a 2x2 one-qubit density matrix.")

    if abs(readout_p0_to_1) < 1e-15 and abs(readout_p1_to_0) < 1e-15:
        tomo = perform_single_qubit_tomography(
            rho,
            shots=shots,
            seed=seed,
            enforce_physical=enforce_physical,
        )
        return SingleQubitTomographyResult(
            input_density_matrix=trace_scale * tomo.input_density_matrix,
            ideal_expectations=tomo.ideal_expectations,
            measured_expectations=tomo.measured_expectations,
            basis_probabilities=tomo.basis_probabilities,
            counts=tomo.counts,
            shots=tomo.shots,
            reconstructed_density_matrix=trace_scale * tomo.reconstructed_density_matrix,
            physical_density_matrix=trace_scale * tomo.physical_density_matrix,
        )

    rng = np.random.default_rng(seed)
    basis_probs = {label: basis_probabilities(rho, label) for label in ("X", "Y", "Z")}
    ideal_expectations = {label: basis_probs[label]["0"] - basis_probs[label]["1"] for label in ("X", "Y", "Z")}

    counts = None
    measured_expectations: dict[str, float] = {}
    if shots is not None:
        counts = {}

    for label in ("X", "Y", "Z"):
        meas = noisy_single_qubit_measurement(
            [basis_probs[label]["0"], basis_probs[label]["1"]],
            p0_to_1=readout_p0_to_1,
            p1_to_0=readout_p1_to_0,
            shots=shots,
            rng=rng,
        )
        if shots is None:
            measured_expectations[label] = float(meas.observed_probabilities[0] - meas.observed_probabilities[1])
        else:
            counts[label] = meas.counts
            measured_expectations[label] = expectation_from_counts(meas.counts)

    reconstructed_normalized = linear_inversion_density_matrix(measured_expectations, trace=1.0)
    if enforce_physical:
        physical_normalized = project_to_physical_density_matrix(reconstructed_normalized)
    else:
        physical_normalized = reconstructed_normalized

    reconstructed_scaled = trace_scale * reconstructed_normalized
    physical_scaled = trace_scale * physical_normalized

    return SingleQubitTomographyResult(
        input_density_matrix=trace_scale * rho,
        ideal_expectations={k: float(v) for k, v in ideal_expectations.items()},
        measured_expectations={k: float(v) for k, v in measured_expectations.items()},
        basis_probabilities=basis_probs,
        counts=counts,
        shots=shots,
        reconstructed_density_matrix=reconstructed_scaled,
        physical_density_matrix=physical_scaled,
    )


def simulate_zz_dephasing_kraus_branch(
    time: float,
    *,
    branch: int,
    initial_state: Optional[ArrayLike] = None,
    theta: float = 0.5,
    lambda0: float = 0.7,
    lambda1: float = 0.3,
    noise_params: Optional[SimpleNISQNoiseParameters] = None,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_physical: bool = True,
) -> KrausBranchSimulationResult:
    if initial_state is None:
        initial_state = ket_plus()
    psi = np.asarray(initial_state, dtype=complex).reshape(2)
    rho0 = density_matrix_from_ket(psi)

    analytic = analytic_svd_for_zz_kraus(time, branch=branch, theta=theta, lambda0=lambda0, lambda1=lambda1)
    input_full_ket = np.kron(np.array([1.0, 0.0], dtype=complex), psi)
    input_full_rho = density_matrix_from_ket(input_full_ket)
    full_output = analytic.full_unitary @ input_full_rho @ analytic.full_unitary.conj().T

    if noise_params is not None:
        full_output = apply_default_noise_for_one_ancilla_step(full_output, noise_params)

    exact_subdensity = analytic.kraus_operator @ rho0 @ analytic.kraus_operator.conj().T
    success_block = ancilla_zero_block(full_output)
    success_probability = float(np.real(np.trace(success_block)))

    if success_probability > 1e-14:
        conditional_for_tomography = _hermitize(success_block / success_probability)
    else:
        conditional_for_tomography = np.eye(2, dtype=complex) / 2.0

    if noise_params is None:
        tomo = perform_single_qubit_tomography(
            conditional_for_tomography,
            shots=shots,
            seed=seed,
            enforce_physical=enforce_physical,
        )
        obtained_subdensity = success_probability * tomo.physical_density_matrix
    else:
        tomo = perform_single_qubit_tomography_with_optional_readout(
            conditional_for_tomography,
            trace_scale=1.0,
            shots=shots,
            seed=seed,
            readout_p0_to_1=noise_params.readout_p0_to_1_system,
            readout_p1_to_0=noise_params.readout_p1_to_0_system,
            enforce_physical=enforce_physical,
        )
        obtained_subdensity = success_probability * tomo.physical_density_matrix

    conditional_exact = exact_subdensity / np.trace(exact_subdensity)

    return KrausBranchSimulationResult(
        label=analytic.label,
        time=float(time),
        exact_subdensity=exact_subdensity,
        obtained_subdensity=obtained_subdensity,
        success_probability=success_probability,
        conditional_state_exact=np.asarray(conditional_exact, dtype=complex),
        conditional_state_for_tomography=np.asarray(conditional_for_tomography, dtype=complex),
        tomography_result=tomo,
        full_two_qubit_output_density=np.asarray(full_output, dtype=complex),
    )


def simulate_zz_dephasing_trajectory(
    times: Sequence[float],
    *,
    initial_state: Optional[ArrayLike] = None,
    theta: float = 0.5,
    lambda0: float = 0.7,
    lambda1: float = 0.3,
    noise_params: Optional[SimpleNISQNoiseParameters] = None,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_physical: bool = True,
    renormalize_obtained_output: bool = True,
) -> ZZDephasingTrajectoryResult:
    if initial_state is None:
        initial_state = ket_plus()

    times = np.asarray(times, dtype=float).reshape(-1)
    exact_rhos: list[np.ndarray] = []
    obtained_rhos: list[np.ndarray] = []
    all_branch_results: list[list[KrausBranchSimulationResult]] = []

    seed_base = 0 if seed is None else int(seed)

    for idx_t, time in enumerate(times):
        exact_total = exact_zz_dephasing_density_matrix(
            time,
            initial_state=initial_state,
            theta=theta,
            lambda0=lambda0,
            lambda1=lambda1,
        )
        branch_results: list[KrausBranchSimulationResult] = []
        obtained_total = np.zeros((2, 2), dtype=complex)

        for branch in (0, 1):
            branch_seed = seed_base + 1000 * idx_t + branch
            result = simulate_zz_dephasing_kraus_branch(
                time,
                branch=branch,
                initial_state=initial_state,
                theta=theta,
                lambda0=lambda0,
                lambda1=lambda1,
                noise_params=noise_params,
                shots=shots,
                seed=branch_seed,
                enforce_physical=enforce_physical,
            )
            branch_results.append(result)
            obtained_total = obtained_total + result.obtained_subdensity

        obtained_total = _hermitize(obtained_total)
        if renormalize_obtained_output:
            tr = float(np.real(np.trace(obtained_total)))
            if tr > 1e-14:
                obtained_total = obtained_total / tr

        exact_rhos.append(np.asarray(exact_total, dtype=complex))
        obtained_rhos.append(np.asarray(obtained_total, dtype=complex))
        all_branch_results.append(branch_results)

    exact_rhos_arr = np.asarray(exact_rhos, dtype=complex)
    obtained_rhos_arr = np.asarray(obtained_rhos, dtype=complex)

    coherence_exact = make_coherence_series_from_density_matrices(times, exact_rhos_arr, label="ZZ dephasing")
    coherence_obtained = make_coherence_series_from_density_matrices(times, obtained_rhos_arr, label="ZZ dephasing")
    bloch_exact = make_bloch_trajectory_from_density_matrices(times, exact_rhos_arr, label="ZZ dephasing")
    bloch_obtained = make_bloch_trajectory_from_density_matrices(times, obtained_rhos_arr, label="ZZ dephasing")

    return ZZDephasingTrajectoryResult(
        times=times,
        exact_density_matrices=exact_rhos_arr,
        obtained_density_matrices=obtained_rhos_arr,
        coherence_exact=coherence_exact,
        coherence_obtained=coherence_obtained,
        bloch_exact=bloch_exact,
        bloch_obtained=bloch_obtained,
        branch_results=all_branch_results,
    )
