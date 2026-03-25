
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
    component_label: str
    branch_label: str
    time: float
    component_weight: float
    exact_subdensity: np.ndarray
    obtained_subdensity: np.ndarray
    success_probability: float
    conditional_state_exact: Optional[np.ndarray]
    conditional_state_for_tomography: np.ndarray
    tomography_result: SingleQubitTomographyResult
    full_two_qubit_output_density: np.ndarray


@dataclass(frozen=True)
class AmpDampTrajectoryResult:
    times: np.ndarray
    exact_density_matrices: np.ndarray
    obtained_density_matrices: np.ndarray
    coherence_exact: CoherenceSeries
    coherence_obtained: CoherenceSeries
    component_branch_results: list[list[KrausBranchSimulationResult]]

    @property
    def exact_populations(self) -> np.ndarray:
        return np.real(np.stack([self.exact_density_matrices[:, 0, 0], self.exact_density_matrices[:, 1, 1]], axis=1))

    @property
    def obtained_populations(self) -> np.ndarray:
        return np.real(np.stack([self.obtained_density_matrices[:, 0, 0], self.obtained_density_matrices[:, 1, 1]], axis=1))


def ket0() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=complex)


def ket1() -> np.ndarray:
    return np.array([0.0, 1.0], dtype=complex)


def ket_plus() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)


def density_matrix_from_ket(ket: ArrayLike) -> np.ndarray:
    vec = np.asarray(ket, dtype=complex).reshape(-1)
    vec = vec / np.linalg.norm(vec)
    return np.outer(vec, vec.conj())


def paper_initial_density_matrix() -> np.ndarray:
    return 0.25 * np.array([[1.0, 1.0], [1.0, 3.0]], dtype=complex)


def paper_initial_state_decomposition() -> list[tuple[float, np.ndarray, str]]:
    return [
        (0.5, ket1(), "|1>"),
        (0.5, ket_plus(), "|+>"),
    ]


def paper_time_grid(*, t_max: float = 25.0, num_points: int = 26) -> np.ndarray:
    return np.linspace(0.0, float(t_max), int(num_points), dtype=float)


def amplitude_damping_kraus_operators(time: float, *, gamma: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    time = float(time)
    gamma = float(gamma)
    if time < 0.0:
        raise ValueError("time must be nonnegative.")
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative.")
    decay = np.exp(-gamma * time)
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(decay)]], dtype=complex)
    k1 = np.array([[0.0, np.sqrt(1.0 - decay)], [0.0, 0.0]], dtype=complex)
    return k0, k1


def exact_amplitude_damping_density_matrix(
    time: float,
    *,
    initial_density_matrix: Optional[np.ndarray] = None,
    gamma: float = 0.15,
) -> np.ndarray:
    rho0 = paper_initial_density_matrix() if initial_density_matrix is None else np.asarray(initial_density_matrix, dtype=complex)
    k0, k1 = amplitude_damping_kraus_operators(time, gamma=gamma)
    return k0 @ rho0 @ k0.conj().T + k1 @ rho0 @ k1.conj().T


def analytic_svd_for_amplitude_damping(time: float, *, branch: int, gamma: float = 0.15) -> AnalyticSVDDilation:
    k0, k1 = amplitude_damping_kraus_operators(time, gamma=gamma)
    decay = np.exp(-float(gamma) * float(time))

    if branch == 0:
        label = "K0"
        kraus = k0
        U = np.eye(2, dtype=complex)
        singular_values = np.array([1.0, np.sqrt(decay)], dtype=float)
        Vh = np.eye(2, dtype=complex)
    elif branch == 1:
        label = "K1"
        kraus = k1
        U = np.eye(2, dtype=complex)
        singular_values = np.array([np.sqrt(1.0 - decay), 0.0], dtype=float)
        Vh = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)  # Pauli X
    else:
        raise ValueError("branch must be 0 or 1.")

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
        raise ValueError("rho must have shape (2, 2).")

    if abs(readout_p0_to_1) < 1e-15 and abs(readout_p1_to_0) < 1e-15:
        tomo = perform_single_qubit_tomography(rho, shots=shots, seed=seed, enforce_physical=enforce_physical)
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

    measured_expectations: dict[str, float] = {}
    counts = {} if shots is not None else None

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

    return SingleQubitTomographyResult(
        input_density_matrix=trace_scale * rho,
        ideal_expectations={k: float(v) for k, v in ideal_expectations.items()},
        measured_expectations={k: float(v) for k, v in measured_expectations.items()},
        basis_probabilities=basis_probs,
        counts=counts,
        shots=shots,
        reconstructed_density_matrix=trace_scale * reconstructed_normalized,
        physical_density_matrix=trace_scale * physical_normalized,
    )


def simulate_ampdamp_branch_on_pure_component(
    time: float,
    *,
    component_weight: float,
    component_ket: ArrayLike,
    component_label: str,
    branch: int,
    gamma: float = 0.15,
    noise_params: Optional[SimpleNISQNoiseParameters] = None,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_physical: bool = True,
) -> KrausBranchSimulationResult:
    psi = np.asarray(component_ket, dtype=complex).reshape(2)
    rho_comp = density_matrix_from_ket(psi)

    analytic = analytic_svd_for_amplitude_damping(time, branch=branch, gamma=gamma)
    input_full_ket = np.kron(np.array([1.0, 0.0], dtype=complex), psi)
    input_full_rho = density_matrix_from_ket(input_full_ket)
    full_output = analytic.full_unitary @ input_full_rho @ analytic.full_unitary.conj().T

    if noise_params is not None:
        full_output = apply_default_noise_for_one_ancilla_step(full_output, noise_params)

    exact_subdensity_unweighted = analytic.kraus_operator @ rho_comp @ analytic.kraus_operator.conj().T
    exact_subdensity = float(component_weight) * exact_subdensity_unweighted

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
        obtained_subdensity = float(component_weight) * success_probability * tomo.physical_density_matrix
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
        obtained_subdensity = float(component_weight) * success_probability * tomo.physical_density_matrix

    exact_trace = float(np.real(np.trace(exact_subdensity_unweighted)))
    conditional_exact = None if exact_trace <= 1e-14 else np.asarray(exact_subdensity_unweighted / exact_trace, dtype=complex)

    return KrausBranchSimulationResult(
        component_label=component_label,
        branch_label=analytic.label,
        time=float(time),
        component_weight=float(component_weight),
        exact_subdensity=np.asarray(exact_subdensity, dtype=complex),
        obtained_subdensity=np.asarray(obtained_subdensity, dtype=complex),
        success_probability=success_probability,
        conditional_state_exact=conditional_exact,
        conditional_state_for_tomography=np.asarray(conditional_for_tomography, dtype=complex),
        tomography_result=tomo,
        full_two_qubit_output_density=np.asarray(full_output, dtype=complex),
    )


def simulate_amplitude_damping_trajectory(
    times: Sequence[float],
    *,
    gamma: float = 0.15,
    component_decomposition: Optional[list[tuple[float, np.ndarray, str]]] = None,
    initial_density_matrix: Optional[np.ndarray] = None,
    noise_params: Optional[SimpleNISQNoiseParameters] = None,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_physical: bool = True,
    renormalize_obtained_output: bool = True,
) -> AmpDampTrajectoryResult:
    times = np.asarray(times, dtype=float).reshape(-1)
    if component_decomposition is None:
        component_decomposition = paper_initial_state_decomposition()

    if initial_density_matrix is None:
        initial_density_matrix = paper_initial_density_matrix()

    exact_rhos = []
    obtained_rhos = []
    branch_results_all: list[list[KrausBranchSimulationResult]] = []
    seed_base = 0 if seed is None else int(seed)

    for idx_t, time in enumerate(times):
        exact_total = exact_amplitude_damping_density_matrix(time, initial_density_matrix=initial_density_matrix, gamma=gamma)
        obtained_total = np.zeros((2, 2), dtype=complex)
        branch_results_t: list[KrausBranchSimulationResult] = []

        for idx_c, (weight, ket, label) in enumerate(component_decomposition):
            for branch in (0, 1):
                this_seed = seed_base + 10000 * idx_t + 100 * idx_c + branch
                result = simulate_ampdamp_branch_on_pure_component(
                    time,
                    component_weight=weight,
                    component_ket=ket,
                    component_label=label,
                    branch=branch,
                    gamma=gamma,
                    noise_params=noise_params,
                    shots=shots,
                    seed=this_seed,
                    enforce_physical=enforce_physical,
                )
                obtained_total = obtained_total + result.obtained_subdensity
                branch_results_t.append(result)

        obtained_total = _hermitize(obtained_total)
        if renormalize_obtained_output:
            tr = float(np.real(np.trace(obtained_total)))
            if tr > 1e-14:
                obtained_total = obtained_total / tr

        exact_rhos.append(np.asarray(exact_total, dtype=complex))
        obtained_rhos.append(np.asarray(obtained_total, dtype=complex))
        branch_results_all.append(branch_results_t)

    exact_rhos_arr = np.asarray(exact_rhos, dtype=complex)
    obtained_rhos_arr = np.asarray(obtained_rhos, dtype=complex)

    return AmpDampTrajectoryResult(
        times=times,
        exact_density_matrices=exact_rhos_arr,
        obtained_density_matrices=obtained_rhos_arr,
        coherence_exact=make_coherence_series_from_density_matrices(times, exact_rhos_arr, label="amplitude damping"),
        coherence_obtained=make_coherence_series_from_density_matrices(times, obtained_rhos_arr, label="amplitude damping"),
        component_branch_results=branch_results_all,
    )


def exact_coherence_formula(time: ArrayLike, *, gamma: float = 0.15) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    return 0.25 * np.exp(-0.5 * float(gamma) * t)


def exact_population_formula(time: ArrayLike, *, gamma: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    decay = np.exp(-float(gamma) * t)
    rho00 = 1.0 - 0.75 * decay
    rho11 = 0.75 * decay
    return rho00, rho11
