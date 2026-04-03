"""
QSIM Noise Channels

Implements realistic quantum noise models using Kraus operator formalism.
Noise is applied to the density matrix after each gate operation.

Supported channels:
- Depolarizing: random Pauli errors (X, Y, or Z applied with probability p)
- Dephasing: phase randomization (Z applied with probability p)
- Amplitude damping: energy relaxation toward |0> (T1 decay)
"""

import numpy as np
from typing import Any


# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def apply_noise_to_state(
    state: np.ndarray,
    qubit: int,
    num_qubits: int,
    noise_config: dict[str, Any],
) -> np.ndarray:
    """Apply noise channels to a single qubit after a gate operation.

    Converts state vector to density matrix, applies Kraus operators,
    then samples a new state vector from the resulting mixed state.

    Args:
        state: Current pure state vector.
        qubit: Index of the qubit to apply noise to.
        num_qubits: Total number of qubits.
        noise_config: Dict with noise parameters:
            - depolarizing: float (error probability 0-1)
            - dephasing: float (error probability 0-1)
            - amplitude_damping: float (damping probability 0-1)

    Returns:
        New state vector after noise application.
    """
    dep_p = noise_config.get("depolarizing", 0.0)
    deph_p = noise_config.get("dephasing", 0.0)
    amp_p = noise_config.get("amplitude_damping", 0.0)

    if dep_p <= 0 and deph_p <= 0 and amp_p <= 0:
        return state

    # Convert to density matrix
    rho = np.outer(state, state.conj())

    if dep_p > 0:
        rho = _depolarizing_channel(rho, qubit, num_qubits, dep_p)

    if deph_p > 0:
        rho = _dephasing_channel(rho, qubit, num_qubits, deph_p)

    if amp_p > 0:
        rho = _amplitude_damping_channel(rho, qubit, num_qubits, amp_p)

    # Sample a pure state from the density matrix
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues / eigenvalues.sum()

    idx = np.random.choice(len(eigenvalues), p=eigenvalues)
    new_state = eigenvectors[:, idx]

    # Fix global phase so largest component is real-positive
    max_idx = np.argmax(np.abs(new_state))
    if np.abs(new_state[max_idx]) > 1e-10:
        new_state = new_state * np.exp(-1j * np.angle(new_state[max_idx]))

    return new_state


def _build_single_qubit_superop(
    kraus_ops: list[np.ndarray], qubit: int, num_qubits: int
) -> list[np.ndarray]:
    """Expand single-qubit Kraus operators to full system size."""
    full_ops = []
    for K in kraus_ops:
        op = np.eye(1, dtype=np.complex128)
        for i in range(num_qubits):
            op = np.kron(op, K if i == qubit else _I)
        full_ops.append(op)
    return full_ops


def _apply_kraus(rho: np.ndarray, kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Apply Kraus operators: rho' = sum_k K_k @ rho @ K_k^dagger."""
    result = np.zeros_like(rho)
    for K in kraus_ops:
        result += K @ rho @ K.conj().T
    return result


def _depolarizing_channel(
    rho: np.ndarray, qubit: int, num_qubits: int, p: float
) -> np.ndarray:
    """Depolarizing noise: with probability p, apply a random Pauli (X, Y, or Z).

    K0 = sqrt(1-p) * I, K1 = sqrt(p/3) * X, K2 = sqrt(p/3) * Y, K3 = sqrt(p/3) * Z
    """
    p = min(max(p, 0), 1)
    kraus_2x2 = [
        np.sqrt(1 - p) * _I,
        np.sqrt(p / 3) * _X,
        np.sqrt(p / 3) * _Y,
        np.sqrt(p / 3) * _Z,
    ]
    return _apply_kraus(rho, _build_single_qubit_superop(kraus_2x2, qubit, num_qubits))


def _dephasing_channel(
    rho: np.ndarray, qubit: int, num_qubits: int, p: float
) -> np.ndarray:
    """Dephasing noise: randomly applies Z with probability p.

    K0 = sqrt(1-p) * I, K1 = sqrt(p) * Z
    Destroys off-diagonal coherence without affecting populations.
    """
    p = min(max(p, 0), 1)
    kraus_2x2 = [
        np.sqrt(1 - p) * _I,
        np.sqrt(p) * _Z,
    ]
    return _apply_kraus(rho, _build_single_qubit_superop(kraus_2x2, qubit, num_qubits))


def _amplitude_damping_channel(
    rho: np.ndarray, qubit: int, num_qubits: int, gamma: float
) -> np.ndarray:
    """Amplitude damping: models T1 energy relaxation toward |0>.

    K0 = [[1, 0], [0, sqrt(1-gamma)]], K1 = [[0, sqrt(gamma)], [0, 0]]
    """
    gamma = min(max(gamma, 0), 1)
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)
    return _apply_kraus(rho, _build_single_qubit_superop([K0, K1], qubit, num_qubits))
