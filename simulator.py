"""
QSIM Quantum Circuit Simulator

Core simulation engine using NumPy for state vector manipulation.
Supports arbitrary multi-qubit gates via einsum-based tensor contraction,
multi-shot measurement, individual qubit measurement, Bloch sphere extraction,
and entanglement detection via von Neumann entropy.
"""

import numpy as np
from typing import Any

from gates import get_gate_matrix
from noise import apply_noise_to_state


def initialize_state(num_qubits: int) -> np.ndarray:
    """Create the |00...0> initial state vector for n qubits.

    Args:
        num_qubits: Number of qubits (1-10).

    Returns:
        Complex state vector of dimension 2^n with all amplitude in |0...0>.
    """
    dim = 2**num_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    return state


def apply_multi_qubit_gate(
    state: np.ndarray,
    gate_matrix: np.ndarray,
    qubit_indices: list[int],
    num_qubits: int,
) -> np.ndarray:
    """Apply an arbitrary k-qubit gate to the state vector using einsum tensor contraction.

    This generalizes to any gate size (1, 2, 3, or more qubits) without
    requiring SWAP networks. The gate matrix is reshaped into a tensor and
    contracted with the state tensor at the specified qubit positions.

    Args:
        state: Current state vector of dimension 2^n.
        gate_matrix: (2^k x 2^k) unitary gate matrix.
        qubit_indices: List of k qubit indices the gate acts on, ordered as
                       the gate expects (e.g., [control, target] for CNOT).
        num_qubits: Total number of qubits in the system.

    Returns:
        New state vector after gate application.
    """
    k = len(qubit_indices)
    gate_tensor = gate_matrix.reshape([2] * k + [2] * k)
    state_tensor = state.reshape([2] * num_qubits)

    # Build einsum index lists
    state_indices = list(range(num_qubits))
    gate_out_indices = list(range(num_qubits, num_qubits + k))
    gate_in_indices = [qubit_indices[i] for i in range(k)]
    gate_indices = gate_out_indices + gate_in_indices

    # Result: same as state but qubit positions replaced with gate output indices
    result_indices = list(state_indices)
    for i, qi in enumerate(qubit_indices):
        result_indices[qi] = num_qubits + i

    new_tensor = np.einsum(gate_tensor, gate_indices, state_tensor, state_indices, result_indices)
    return new_tensor.reshape(2**num_qubits)


def get_probabilities(state: np.ndarray) -> dict[str, float]:
    """Calculate measurement probabilities for all basis states via Born rule.

    P(|x>) = |<x|psi>|^2

    Args:
        state: State vector.

    Returns:
        Dict mapping basis state labels (e.g. '00', '01') to probabilities.
    """
    num_qubits = int(np.log2(len(state)))
    probs = np.abs(state) ** 2
    return {format(i, f"0{num_qubits}b"): float(probs[i]) for i in range(len(state))}


def measure(state: np.ndarray) -> tuple[str, np.ndarray]:
    """Perform a projective measurement on the full state, collapsing it.

    Args:
        state: State vector before measurement.

    Returns:
        Tuple of (measurement result as bit string, collapsed state vector).
    """
    num_qubits = int(np.log2(len(state)))
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    outcome = np.random.choice(len(state), p=probs)
    collapsed = np.zeros_like(state)
    collapsed[outcome] = 1.0
    return format(outcome, f"0{num_qubits}b"), collapsed


def measure_qubit(state: np.ndarray, qubit: int, num_qubits: int) -> tuple[int, np.ndarray]:
    """Measure a single qubit, collapsing only that qubit.

    Args:
        state: Full system state vector.
        qubit: Index of the qubit to measure.
        num_qubits: Total number of qubits.

    Returns:
        Tuple of (measurement result 0 or 1, collapsed state vector).
    """
    dim = 2**num_qubits
    probs = np.abs(state) ** 2

    # Sum probabilities where the target qubit is 0
    prob_0 = 0.0
    for i in range(dim):
        bit = (i >> (num_qubits - 1 - qubit)) & 1
        if bit == 0:
            prob_0 += probs[i]

    # Randomly choose outcome
    outcome = 0 if np.random.random() < prob_0 else 1

    # Collapse: zero out inconsistent amplitudes, renormalize
    collapsed = state.copy()
    for i in range(dim):
        bit = (i >> (num_qubits - 1 - qubit)) & 1
        if bit != outcome:
            collapsed[i] = 0.0

    norm = np.linalg.norm(collapsed)
    if norm > 1e-12:
        collapsed /= norm

    return outcome, collapsed


def measure_shots(state: np.ndarray, num_qubits: int, num_shots: int) -> dict[str, int]:
    """Run multiple measurement shots and return outcome counts.

    Args:
        state: State vector (not collapsed -- each shot samples independently).
        num_qubits: Total number of qubits.
        num_shots: Number of measurement repetitions.

    Returns:
        Dict mapping basis state labels to count of times observed.
    """
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    outcomes = np.random.choice(len(state), size=num_shots, p=probs)
    counts: dict[str, int] = {}
    for o in outcomes:
        label = format(o, f"0{num_qubits}b")
        counts[label] = counts.get(label, 0) + 1
    # Sort by bit string
    return dict(sorted(counts.items()))


def reduced_density_matrix(state: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    """Compute the reduced density matrix for a single qubit by tracing out all others.

    Args:
        state: Full system state vector.
        qubit: Index of the qubit to keep.
        num_qubits: Total number of qubits.

    Returns:
        2x2 reduced density matrix for the specified qubit.
    """
    rho = np.outer(state, state.conj())
    shape = [2] * (2 * num_qubits)
    rho_tensor = rho.reshape(shape)

    bra_indices = list(range(num_qubits))
    ket_indices = list(range(num_qubits, 2 * num_qubits))

    qubits_to_trace = [q for q in range(num_qubits) if q != qubit]
    for q in qubits_to_trace:
        ket_indices[q] = bra_indices[q]

    out_bra = bra_indices[qubit]
    out_ket = ket_indices[qubit]

    rho_reduced = np.einsum(rho_tensor, bra_indices + ket_indices, [out_bra, out_ket])
    return rho_reduced


def bloch_vector(state: np.ndarray, qubit: int, num_qubits: int) -> list[float]:
    """Extract the Bloch sphere coordinates (x, y, z) for a single qubit.

    Args:
        state: Full system state vector.
        qubit: Index of the qubit.
        num_qubits: Total number of qubits.

    Returns:
        [x, y, z] Bloch vector components.
    """
    rho = reduced_density_matrix(state, qubit, num_qubits)

    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))

    return [float(x), float(y), float(z)]


def von_neumann_entropy(state: np.ndarray, qubit: int, num_qubits: int) -> float:
    """Compute the von Neumann entropy of a single qubit's reduced density matrix.

    S(rho) = -Tr(rho * log2(rho))

    Args:
        state: Full system state vector.
        qubit: Index of the qubit.
        num_qubits: Total number of qubits.

    Returns:
        Von Neumann entropy in bits.
    """
    rho = reduced_density_matrix(state, qubit, num_qubits)
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return float(entropy)


def system_entanglement_entropy(state: np.ndarray, num_qubits: int) -> float:
    """Compute the average von Neumann entropy across all qubits.

    Args:
        state: Full system state vector.
        num_qubits: Total number of qubits.

    Returns:
        Average von Neumann entropy across all qubits.
    """
    if num_qubits < 2:
        return 0.0
    total = sum(von_neumann_entropy(state, q, num_qubits) for q in range(num_qubits))
    return total / num_qubits


def simulate_circuit(
    num_qubits: int,
    circuit: list[list[dict[str, Any]]],
    shots: int | None = None,
    noise: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a full quantum circuit simulation.

    Args:
        num_qubits: Number of qubits (1-10).
        circuit: List of circuit steps. Each step is a list of gate operations:
                 {"gate": str, "target": int, "params": dict}
                 For 2-qubit gates: also "control": int
                 For 3-qubit gates: also "control": int, "control2": int
                 For measurement: {"gate": "MEASURE", "target": int}
        shots: If provided, run this many measurement shots and return counts.
        noise: If provided, dict with noise channel probabilities:
               {"depolarizing": float, "dephasing": float, "amplitude_damping": float}
               Applied to each qubit after every gate operation.

    Returns:
        Dict with state_vector, probabilities, bloch_vectors,
        entanglement_entropy, num_gates, and optionally counts and classical_bits.

    Raises:
        ValueError: If num_qubits is out of range or gates are invalid.
    """
    if not 1 <= num_qubits <= 10:
        raise ValueError("Number of qubits must be between 1 and 10.")

    noise_config = noise or {}
    noise_enabled = any(noise_config.get(k, 0) > 0 for k in ["depolarizing", "dephasing", "amplitude_damping"])

    state = initialize_state(num_qubits)
    gate_count = 0
    classical_bits: dict[int, int] = {}

    for step in circuit:
        for operation in step:
            gate_name = operation.get("gate", "")
            params = operation.get("params", {})
            target = operation.get("target")

            if target is None:
                continue

            # Handle measurement
            if gate_name.upper() == "MEASURE":
                if 0 <= target < num_qubits:
                    outcome, state = measure_qubit(state, target, num_qubits)
                    classical_bits[target] = outcome
                continue

            # Handle barrier (no-op)
            if gate_name.upper() == "BARRIER":
                continue

            gate_matrix = get_gate_matrix(gate_name, params)
            gate_size = int(np.log2(gate_matrix.shape[0]))

            if gate_size == 1:
                if not 0 <= target < num_qubits:
                    raise ValueError(f"Target qubit {target} out of range for {num_qubits}-qubit system.")
                state = apply_multi_qubit_gate(state, gate_matrix, [target], num_qubits)
                gate_count += 1
                # Apply noise to target qubit
                if noise_enabled:
                    state = apply_noise_to_state(state, target, num_qubits, noise_config)

            elif gate_size == 2:
                control = operation.get("control")
                if control is None:
                    raise ValueError(f"Two-qubit gate {gate_name} requires a 'control' qubit.")
                if not (0 <= control < num_qubits and 0 <= target < num_qubits):
                    raise ValueError(f"Qubit indices out of range for {num_qubits}-qubit system.")
                state = apply_multi_qubit_gate(state, gate_matrix, [control, target], num_qubits)
                gate_count += 1
                # Apply noise to both qubits involved
                if noise_enabled:
                    state = apply_noise_to_state(state, control, num_qubits, noise_config)
                    state = apply_noise_to_state(state, target, num_qubits, noise_config)

            elif gate_size == 3:
                control = operation.get("control")
                control2 = operation.get("control2")
                if control is None or control2 is None:
                    raise ValueError(f"Three-qubit gate {gate_name} requires 'control' and 'control2' qubits.")
                if not all(0 <= q < num_qubits for q in [control, control2, target]):
                    raise ValueError(f"Qubit indices out of range for {num_qubits}-qubit system.")
                state = apply_multi_qubit_gate(state, gate_matrix, [control, control2, target], num_qubits)
                gate_count += 1
                # Apply noise to all three qubits
                if noise_enabled:
                    state = apply_noise_to_state(state, control, num_qubits, noise_config)
                    state = apply_noise_to_state(state, control2, num_qubits, noise_config)
                    state = apply_noise_to_state(state, target, num_qubits, noise_config)

    # Compute results
    probabilities = get_probabilities(state)

    # Only compute Bloch vectors for <= 8 qubits (expensive for large systems)
    if num_qubits <= 8:
        bloch_vectors = [bloch_vector(state, q, num_qubits) for q in range(num_qubits)]
    else:
        bloch_vectors = [[0.0, 0.0, 0.0]] * num_qubits

    entanglement = system_entanglement_entropy(state, num_qubits)

    state_vector = [
        {
            "label": format(i, f"0{num_qubits}b"),
            "amplitude": [float(state[i].real), float(state[i].imag)],
            "probability": float(np.abs(state[i]) ** 2),
        }
        for i in range(len(state))
    ]

    result: dict[str, Any] = {
        "state_vector": state_vector,
        "probabilities": probabilities,
        "bloch_vectors": bloch_vectors,
        "entanglement_entropy": round(entanglement, 6),
        "num_gates": gate_count,
        "noise_enabled": noise_enabled,
    }

    if classical_bits:
        result["classical_bits"] = {str(k): v for k, v in sorted(classical_bits.items())}

    if shots is not None and shots > 0:
        result["counts"] = measure_shots(state, num_qubits, shots)

    return result
