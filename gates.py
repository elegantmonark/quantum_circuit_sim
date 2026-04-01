"""
QSIM Gate Definitions

All quantum gate matrices defined as NumPy complex128 arrays.
Includes single-qubit, two-qubit, three-qubit, controlled-rotation,
and universal gates needed for algorithms like Shor's and Grover's.
"""

import numpy as np
from typing import Any


# ============================================================
# Single Qubit Gates
# ============================================================

def hadamard() -> np.ndarray:
    """Hadamard gate: creates equal superposition from basis states.

    H = (1/sqrt(2)) * [[1, 1], [1, -1]]
    """
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def pauli_x() -> np.ndarray:
    """Pauli-X gate: quantum NOT / bit flip.

    X = [[0, 1], [1, 0]]
    """
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def pauli_y() -> np.ndarray:
    """Pauli-Y gate: bit and phase flip.

    Y = [[0, -i], [i, 0]]
    """
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def pauli_z() -> np.ndarray:
    """Pauli-Z gate: phase flip.

    Z = [[1, 0], [0, -1]]
    """
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def s_gate() -> np.ndarray:
    """S gate (phase gate): sqrt(Z), applies pi/2 phase shift.

    S = [[1, 0], [0, i]]
    """
    return np.array([[1, 0], [0, 1j]], dtype=np.complex128)


def s_dagger() -> np.ndarray:
    """S-dagger gate: inverse of S gate.

    Sdg = [[1, 0], [0, -i]]
    """
    return np.array([[1, 0], [0, -1j]], dtype=np.complex128)


def t_gate() -> np.ndarray:
    """T gate: sqrt(S), applies pi/4 phase shift.

    T = [[1, 0], [0, e^(i*pi/4)]]
    """
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def t_dagger() -> np.ndarray:
    """T-dagger gate: inverse of T gate.

    Tdg = [[1, 0], [0, e^(-i*pi/4)]]
    """
    return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    """Rotation around X-axis by angle theta.

    Rx(theta) = [[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]

    Args:
        theta: Rotation angle in radians.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    """Rotation around Y-axis by angle theta.

    Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]

    Args:
        theta: Rotation angle in radians.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    """Rotation around Z-axis by angle theta.

    Rz(theta) = [[e^(-i*t/2), 0], [0, e^(i*t/2)]]

    Args:
        theta: Rotation angle in radians.
    """
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def phase_gate(theta: float) -> np.ndarray:
    """Phase gate P(theta): applies a phase shift to |1>.

    P(theta) = [[1, 0], [0, e^(i*theta)]]

    Args:
        theta: Phase angle in radians.
    """
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """Universal single-qubit gate U3(theta, phi, lambda).

    U3 = [[cos(t/2),            -e^(i*lam)*sin(t/2)],
           [e^(i*phi)*sin(t/2),  e^(i*(phi+lam))*cos(t/2)]]

    Args:
        theta: Polar angle.
        phi: Azimuthal angle.
        lam: Lambda angle.
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=np.complex128,
    )


def identity() -> np.ndarray:
    """2x2 Identity gate (no operation)."""
    return np.eye(2, dtype=np.complex128)


# ============================================================
# Two Qubit Gates
# ============================================================

def cnot() -> np.ndarray:
    """Controlled-NOT (CNOT) gate: flips target qubit if control is |1>.

    CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    """
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex128,
    )


def swap() -> np.ndarray:
    """SWAP gate: exchanges the states of two qubits."""
    return np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def cz() -> np.ndarray:
    """Controlled-Z (CZ) gate: applies Z to target if control is |1>."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dtype=np.complex128,
    )


def cp(theta: float) -> np.ndarray:
    """Controlled-Phase gate CP(theta): applies phase to |11>.

    CP = diag(1, 1, 1, e^(i*theta))

    Essential for Quantum Fourier Transform.

    Args:
        theta: Phase angle in radians.
    """
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(np.complex128)


def crx(theta: float) -> np.ndarray:
    """Controlled-Rx gate: applies Rx(theta) to target if control is |1>.

    Args:
        theta: Rotation angle in radians.
    """
    m = np.eye(4, dtype=np.complex128)
    r = rx(theta)
    m[2:, 2:] = r
    return m


def cry(theta: float) -> np.ndarray:
    """Controlled-Ry gate: applies Ry(theta) to target if control is |1>.

    Args:
        theta: Rotation angle in radians.
    """
    m = np.eye(4, dtype=np.complex128)
    m[2:, 2:] = ry(theta)
    return m


def crz(theta: float) -> np.ndarray:
    """Controlled-Rz gate: applies Rz(theta) to target if control is |1>.

    Args:
        theta: Rotation angle in radians.
    """
    m = np.eye(4, dtype=np.complex128)
    m[2:, 2:] = rz(theta)
    return m


# ============================================================
# Three Qubit Gates
# ============================================================

def toffoli() -> np.ndarray:
    """Toffoli (CCX) gate: flips target only if both controls are |1>.

    8x8 identity with rows 6,7 swapped (|110> <-> |111>).
    """
    m = np.eye(8, dtype=np.complex128)
    m[6, 6] = 0
    m[7, 7] = 0
    m[6, 7] = 1
    m[7, 6] = 1
    return m


def fredkin() -> np.ndarray:
    """Fredkin (CSWAP) gate: swaps targets only if control is |1>.

    8x8 identity with rows 5,6 swapped (|101> <-> |110>).
    """
    m = np.eye(8, dtype=np.complex128)
    m[5, 5] = 0
    m[6, 6] = 0
    m[5, 6] = 1
    m[6, 5] = 1
    return m


def ccz() -> np.ndarray:
    """CCZ gate: applies phase flip only to |111>.

    8x8 identity with entry [7,7] = -1.
    """
    m = np.eye(8, dtype=np.complex128)
    m[7, 7] = -1
    return m


# ============================================================
# Gate Lookup and Catalogue
# ============================================================

def _matrix_to_list(m: np.ndarray) -> list[list[list[float]]]:
    """Convert a numpy matrix to JSON-serializable nested list of [real, imag] pairs."""
    return [[[float(x.real), float(x.imag)] for x in row] for row in m]


def get_gate_matrix(name: str, params: dict[str, Any] | None = None) -> np.ndarray:
    """Look up a gate matrix by name, applying any parameters.

    Args:
        name: Gate name (case-insensitive).
        params: Optional dict with 'theta', 'phi', 'lambda' for parameterized gates.

    Returns:
        The unitary matrix for the requested gate.

    Raises:
        ValueError: If the gate name is not recognized.
    """
    params = params or {}
    name_upper = name.upper()

    # Simple single-qubit gates
    simple_1q: dict[str, Any] = {
        "H": hadamard, "X": pauli_x, "Y": pauli_y, "Z": pauli_z,
        "S": s_gate, "SDG": s_dagger, "T": t_gate, "TDG": t_dagger,
        "I": identity,
    }
    if name_upper in simple_1q:
        return simple_1q[name_upper]()

    # Parameterized single-qubit gates
    if name_upper in ("RX", "RY", "RZ"):
        theta = float(params.get("theta", np.pi))
        return {"RX": rx, "RY": ry, "RZ": rz}[name_upper](theta)

    if name_upper == "P":
        theta = float(params.get("theta", np.pi))
        return phase_gate(theta)

    if name_upper == "U3":
        theta = float(params.get("theta", 0))
        phi = float(params.get("phi", 0))
        lam = float(params.get("lambda", 0))
        return u3(theta, phi, lam)

    # Simple two-qubit gates
    simple_2q: dict[str, Any] = {
        "CNOT": cnot, "CX": cnot, "SWAP": swap, "CZ": cz,
    }
    if name_upper in simple_2q:
        return simple_2q[name_upper]()

    # Parameterized two-qubit gates
    if name_upper in ("CP", "CPHASE"):
        theta = float(params.get("theta", np.pi))
        return cp(theta)

    if name_upper in ("CRX", "CRY", "CRZ"):
        theta = float(params.get("theta", np.pi))
        return {"CRX": crx, "CRY": cry, "CRZ": crz}[name_upper](theta)

    # Three-qubit gates
    three_q: dict[str, Any] = {
        "TOFFOLI": toffoli, "CCX": toffoli,
        "FREDKIN": fredkin, "CSWAP": fredkin,
        "CCZ": ccz,
    }
    if name_upper in three_q:
        return three_q[name_upper]()

    raise ValueError(f"Unknown gate: {name}")


def get_gate_catalogue() -> list[dict[str, Any]]:
    """Return the full gate catalogue with metadata for the API."""
    catalogue = [
        # Single qubit
        {"name": "H", "symbol": "H", "num_qubits": 1, "matrix": _matrix_to_list(hadamard()),
         "description": "Hadamard -- creates equal superposition. Maps |0> to |+> and |1> to |->.", "parameterized": False, "category": "single"},
        {"name": "X", "symbol": "X", "num_qubits": 1, "matrix": _matrix_to_list(pauli_x()),
         "description": "Pauli-X (NOT) -- bit flip. Swaps |0> and |1>.", "parameterized": False, "category": "single"},
        {"name": "Y", "symbol": "Y", "num_qubits": 1, "matrix": _matrix_to_list(pauli_y()),
         "description": "Pauli-Y -- combined bit and phase flip.", "parameterized": False, "category": "single"},
        {"name": "Z", "symbol": "Z", "num_qubits": 1, "matrix": _matrix_to_list(pauli_z()),
         "description": "Pauli-Z -- phase flip. Leaves |0> unchanged, maps |1> to -|1>.", "parameterized": False, "category": "single"},
        {"name": "S", "symbol": "S", "num_qubits": 1, "matrix": _matrix_to_list(s_gate()),
         "description": "S (phase) gate -- sqrt(Z). Applies pi/2 phase shift to |1>.", "parameterized": False, "category": "single"},
        {"name": "Sdg", "symbol": "S\u2020", "num_qubits": 1, "matrix": _matrix_to_list(s_dagger()),
         "description": "S-dagger -- inverse of S gate. Applies -pi/2 phase shift.", "parameterized": False, "category": "single"},
        {"name": "T", "symbol": "T", "num_qubits": 1, "matrix": _matrix_to_list(t_gate()),
         "description": "T gate -- sqrt(S). Applies pi/4 phase shift to |1>.", "parameterized": False, "category": "single"},
        {"name": "Tdg", "symbol": "T\u2020", "num_qubits": 1, "matrix": _matrix_to_list(t_dagger()),
         "description": "T-dagger -- inverse of T gate. Applies -pi/4 phase shift.", "parameterized": False, "category": "single"},

        # Parameterized single qubit
        {"name": "Rx", "symbol": "Rx", "num_qubits": 1, "matrix": _matrix_to_list(rx(np.pi)),
         "description": "Rotation around X-axis by angle theta (radians).", "parameterized": True, "category": "rotation"},
        {"name": "Ry", "symbol": "Ry", "num_qubits": 1, "matrix": _matrix_to_list(ry(np.pi)),
         "description": "Rotation around Y-axis by angle theta (radians).", "parameterized": True, "category": "rotation"},
        {"name": "Rz", "symbol": "Rz", "num_qubits": 1, "matrix": _matrix_to_list(rz(np.pi)),
         "description": "Rotation around Z-axis by angle theta (radians).", "parameterized": True, "category": "rotation"},
        {"name": "P", "symbol": "P", "num_qubits": 1, "matrix": _matrix_to_list(phase_gate(np.pi)),
         "description": "Phase gate P(theta) -- applies e^(i*theta) phase to |1>.", "parameterized": True, "category": "rotation"},
        {"name": "U3", "symbol": "U", "num_qubits": 1, "matrix": _matrix_to_list(u3(np.pi, 0, np.pi)),
         "description": "Universal gate U3(theta, phi, lambda) -- any single-qubit unitary.", "parameterized": True, "category": "rotation", "multi_param": True},

        # Two qubit
        {"name": "CNOT", "symbol": "CX", "num_qubits": 2, "matrix": _matrix_to_list(cnot()),
         "description": "Controlled-NOT -- flips target when control is |1>. Creates entanglement.", "parameterized": False, "category": "two_qubit"},
        {"name": "SWAP", "symbol": "SW", "num_qubits": 2, "matrix": _matrix_to_list(swap()),
         "description": "SWAP -- exchanges the quantum states of two qubits.", "parameterized": False, "category": "two_qubit"},
        {"name": "CZ", "symbol": "CZ", "num_qubits": 2, "matrix": _matrix_to_list(cz()),
         "description": "Controlled-Z -- applies Z to target when control is |1>.", "parameterized": False, "category": "two_qubit"},

        # Controlled rotations (two-qubit parameterized)
        {"name": "CP", "symbol": "CP", "num_qubits": 2, "matrix": _matrix_to_list(cp(np.pi)),
         "description": "Controlled-Phase CP(theta) -- applies phase e^(i*theta) to |11>. Key gate for QFT.", "parameterized": True, "category": "ctrl_rotation"},
        {"name": "CRx", "symbol": "CRx", "num_qubits": 2, "matrix": _matrix_to_list(crx(np.pi)),
         "description": "Controlled-Rx -- applies Rx(theta) to target when control is |1>.", "parameterized": True, "category": "ctrl_rotation"},
        {"name": "CRy", "symbol": "CRy", "num_qubits": 2, "matrix": _matrix_to_list(cry(np.pi)),
         "description": "Controlled-Ry -- applies Ry(theta) to target when control is |1>.", "parameterized": True, "category": "ctrl_rotation"},
        {"name": "CRz", "symbol": "CRz", "num_qubits": 2, "matrix": _matrix_to_list(crz(np.pi)),
         "description": "Controlled-Rz -- applies Rz(theta) to target when control is |1>.", "parameterized": True, "category": "ctrl_rotation"},

        # Three qubit
        {"name": "Toffoli", "symbol": "CCX", "num_qubits": 3, "matrix": _matrix_to_list(toffoli()),
         "description": "Toffoli (CCX) -- flips target only when both controls are |1>. Universal for classical computation.", "parameterized": False, "category": "three_qubit"},
        {"name": "Fredkin", "symbol": "CSW", "num_qubits": 3, "matrix": _matrix_to_list(fredkin()),
         "description": "Fredkin (CSWAP) -- swaps two target qubits when control is |1>.", "parameterized": False, "category": "three_qubit"},
        {"name": "CCZ", "symbol": "CCZ", "num_qubits": 3, "matrix": _matrix_to_list(ccz()),
         "description": "CCZ -- applies phase flip (-1) only to |111>.", "parameterized": False, "category": "three_qubit"},
    ]
    return catalogue
