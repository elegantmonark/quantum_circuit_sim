"""
QSIM Pre-built Circuit Templates

Standard quantum algorithm circuits ready to load into the simulator.
Each template returns a dict with num_qubits, circuit steps, name, and description.
"""

import math
from typing import Any


def _op(gate: str, target: int, **kwargs: Any) -> dict[str, Any]:
    """Helper to construct a gate operation dict."""
    op: dict[str, Any] = {"gate": gate, "target": target, "params": {}}
    op.update(kwargs)
    return op


def bell_state() -> dict[str, Any]:
    """Bell state: maximally entangled 2-qubit state (|00> + |11>) / sqrt(2)."""
    return {
        "name": "Bell State",
        "description": "Creates a maximally entangled Bell pair using H + CNOT. Produces (|00> + |11>)/sqrt(2).",
        "num_qubits": 2,
        "circuit": [
            [_op("H", 0)],
            [_op("CNOT", 1, control=0)],
        ],
    }


def ghz_state() -> dict[str, Any]:
    """GHZ state: 3-qubit entangled state (|000> + |111>) / sqrt(2)."""
    return {
        "name": "GHZ State",
        "description": "Greenberger-Horne-Zeilinger state -- 3-qubit entanglement. Produces (|000> + |111>)/sqrt(2).",
        "num_qubits": 3,
        "circuit": [
            [_op("H", 0)],
            [_op("CNOT", 1, control=0)],
            [_op("CNOT", 2, control=1)],
        ],
    }


def quantum_teleportation() -> dict[str, Any]:
    """Quantum teleportation circuit (3 qubits).

    Teleports the state of q0 to q2 using a Bell pair (q1, q2).
    Uses deferred measurement principle: CNOT and CZ replace classically-conditioned ops.
    """
    return {
        "name": "Quantum Teleportation",
        "description": "Teleports q0's state to q2 via entanglement. Apply a gate to q0 first to see it appear on q2.",
        "num_qubits": 3,
        "circuit": [
            # Prepare Bell pair on q1-q2
            [_op("H", 1)],
            [_op("CNOT", 2, control=1)],
            # Bell measurement on q0-q1
            [_op("CNOT", 1, control=0)],
            [_op("H", 0)],
            # Corrections (deferred measurement)
            [_op("CNOT", 2, control=1)],
            [_op("CZ", 2, control=0)],
        ],
    }


def qft_4() -> dict[str, Any]:
    """Quantum Fourier Transform on 4 qubits.

    The QFT maps computational basis states to Fourier basis states.
    Essential subroutine in Shor's algorithm.
    """
    pi = math.pi
    return {
        "name": "QFT (4 qubit)",
        "description": "Quantum Fourier Transform on 4 qubits. Key subroutine for Shor's algorithm and phase estimation.",
        "num_qubits": 4,
        "circuit": [
            # q0
            [_op("H", 0)],
            [_op("CP", 0, control=1, params={"theta": pi / 2})],
            [_op("CP", 0, control=2, params={"theta": pi / 4})],
            [_op("CP", 0, control=3, params={"theta": pi / 8})],
            # q1
            [_op("H", 1)],
            [_op("CP", 1, control=2, params={"theta": pi / 2})],
            [_op("CP", 1, control=3, params={"theta": pi / 4})],
            # q2
            [_op("H", 2)],
            [_op("CP", 2, control=3, params={"theta": pi / 2})],
            # q3
            [_op("H", 3)],
            # Swap to reverse order
            [_op("SWAP", 3, control=0)],
            [_op("SWAP", 2, control=1)],
        ],
    }


def inverse_qft_4() -> dict[str, Any]:
    """Inverse QFT on 4 qubits (used in Shor's algorithm)."""
    pi = math.pi
    return {
        "name": "Inverse QFT (4 qubit)",
        "description": "Inverse Quantum Fourier Transform. Reverses the QFT to extract phase information.",
        "num_qubits": 4,
        "circuit": [
            # Swap to reverse order first
            [_op("SWAP", 3, control=0)],
            [_op("SWAP", 2, control=1)],
            # q3 (reverse order of QFT)
            [_op("H", 3)],
            # q2
            [_op("CP", 2, control=3, params={"theta": -pi / 2})],
            [_op("H", 2)],
            # q1
            [_op("CP", 1, control=3, params={"theta": -pi / 4})],
            [_op("CP", 1, control=2, params={"theta": -pi / 2})],
            [_op("H", 1)],
            # q0
            [_op("CP", 0, control=3, params={"theta": -pi / 8})],
            [_op("CP", 0, control=2, params={"theta": -pi / 4})],
            [_op("CP", 0, control=1, params={"theta": -pi / 2})],
            [_op("H", 0)],
        ],
    }


def grover_2qubit() -> dict[str, Any]:
    """Grover's search algorithm for 2 qubits, searching for |11>.

    One iteration of Grover's gives the correct answer with certainty for N=4.
    Oracle marks |11>, diffusion amplifies its amplitude.
    """
    return {
        "name": "Grover's Search (2-qubit)",
        "description": "Grover's algorithm searching for |11> in a 2-qubit space. One iteration finds the answer with 100% probability.",
        "num_qubits": 2,
        "circuit": [
            # Superposition
            [_op("H", 0), _op("H", 1)],
            # Oracle: mark |11> with phase -1 (CZ does this)
            [_op("CZ", 1, control=0)],
            # Diffusion operator
            [_op("H", 0), _op("H", 1)],
            [_op("X", 0), _op("X", 1)],
            [_op("CZ", 1, control=0)],
            [_op("X", 0), _op("X", 1)],
            [_op("H", 0), _op("H", 1)],
        ],
    }


def shor_15() -> dict[str, Any]:
    """Shor's algorithm to factor N=15 with a=2.

    Uses 7 qubits: q0-q2 are the 3-bit counting register, q3-q6 are the 4-bit work register.
    Work register encoding: q3 = MSB (bit 3), q6 = LSB (bit 0).

    The circuit implements controlled modular exponentiation of 2^(2^j) mod 15
    using Fredkin (CSWAP) gates, followed by inverse QFT on the counting register.

    For a=2, N=15:
      2^1 mod 15 = 2   -> multiply by 2  = cyclic left shift by 1
      2^2 mod 15 = 4   -> multiply by 4  = cyclic left shift by 2
      2^4 mod 15 = 1   -> identity

    Phase estimation qubit assignments (3 counting qubits, LSB controls lowest power):
      q0 controls U^(2^2) = U^4 = identity (2^4 mod 15 = 1, no gates)
      q1 controls U^(2^1) = U^2 = multiply by 4 (cyclic left shift by 2)
      q2 controls U^(2^0) = U^1 = multiply by 2 (cyclic left shift by 1)

    The period r=4. With 3 counting qubits (8 values), expected peaks at
    multiples of 8/r = 2: counting register peaks at 0, 2, 4, 6.
    Using continued fractions: 0/8=0, 2/8=1/4, 4/8=1/2, 6/8=3/4 -> r=4.
    Factors: gcd(2^2 - 1, 15) = gcd(3, 15) = 3 and gcd(2^2 + 1, 15) = gcd(5, 15) = 5.
    """
    pi = math.pi
    steps: list[list[dict[str, Any]]] = []

    # Initialize work register to |0001> (decimal 1): set q6 = |1>
    steps.append([_op("X", 6)])

    # Superposition on counting register
    steps.append([_op("H", 0), _op("H", 1), _op("H", 2)])

    # --- Controlled-U^1: multiply by 2 mod 15 (controlled by q2) ---
    # Cyclic left shift by 1 on work register q3-q6: |b3 b2 b1 b0> -> |b2 b1 b0 b3>
    # CSWAP(q3,q4), CSWAP(q4,q5), CSWAP(q5,q6) -- each controlled by q2
    steps.append([_op("Fredkin", 4, control=2, control2=3)])   # CSWAP q3,q4
    steps.append([_op("Fredkin", 5, control=2, control2=4)])   # CSWAP q4,q5
    steps.append([_op("Fredkin", 6, control=2, control2=5)])   # CSWAP q5,q6

    # --- Controlled-U^2: multiply by 4 mod 15 (controlled by q1) ---
    # Cyclic left shift by 2: |b3 b2 b1 b0> -> |b1 b0 b3 b2>
    # CSWAP(q3,q5), CSWAP(q4,q6) -- each controlled by q1
    steps.append([_op("Fredkin", 5, control=1, control2=3)])   # CSWAP q3,q5
    steps.append([_op("Fredkin", 6, control=1, control2=4)])   # CSWAP q4,q6

    # --- Controlled-U^4: multiply by 2^4 mod 15 = 1 (controlled by q0) ---
    # Identity -- no gates needed

    # Inverse QFT on 3-qubit counting register (q0-q2)
    # Reverse qubit order: swap q0 <-> q2
    steps.append([_op("SWAP", 2, control=0)])

    # IQFT rotations (reverse order of QFT with negative angles)
    steps.append([_op("H", 2)])
    steps.append([_op("CP", 1, control=2, params={"theta": -pi / 2})])
    steps.append([_op("H", 1)])
    steps.append([_op("CP", 0, control=2, params={"theta": -pi / 4})])
    steps.append([_op("CP", 0, control=1, params={"theta": -pi / 2})])
    steps.append([_op("H", 0)])

    return {
        "name": "Shor's Algorithm (N=15)",
        "description": "Shor's factoring algorithm for N=15 with a=2. Uses 7 qubits (3 counting + 4 work). Controlled modular multiplication via Fredkin (CSWAP) gates as cyclic bit shifts. Counting register peaks at 0, 2, 4, 6 -- period r=4 yields factors 3 and 5. Run with 1024+ shots.",
        "num_qubits": 7,
        "circuit": steps,
    }


def deutsch_jozsa() -> dict[str, Any]:
    """Deutsch-Jozsa algorithm for a balanced oracle on 3 qubits.

    Determines in one query whether a function is constant or balanced.
    Uses q0-q1 as input and q2 as output ancilla.
    """
    return {
        "name": "Deutsch-Jozsa (Balanced)",
        "description": "Deutsch-Jozsa algorithm with a balanced oracle. Determines in a single query that f is balanced (non-zero measurement).",
        "num_qubits": 3,
        "circuit": [
            # Initialize ancilla to |1>
            [_op("X", 2)],
            # Superposition
            [_op("H", 0), _op("H", 1), _op("H", 2)],
            # Balanced oracle: CNOT from each input to ancilla
            [_op("CNOT", 2, control=0)],
            [_op("CNOT", 2, control=1)],
            # Hadamard on inputs
            [_op("H", 0), _op("H", 1)],
        ],
    }


def superdense_coding() -> dict[str, Any]:
    """Superdense coding: send 2 classical bits using 1 qubit.

    Encodes classical message '11' by applying Z then X to q0.
    """
    return {
        "name": "Superdense Coding",
        "description": "Transmits 2 classical bits via 1 qubit using shared entanglement. Encodes message '11'.",
        "num_qubits": 2,
        "circuit": [
            # Create Bell pair
            [_op("H", 0)],
            [_op("CNOT", 1, control=0)],
            # Encode message '11': apply Z then X to q0
            [_op("Z", 0)],
            [_op("X", 0)],
            # Decode
            [_op("CNOT", 1, control=0)],
            [_op("H", 0)],
        ],
    }


# Registry of all templates
TEMPLATES: dict[str, Any] = {
    "bell": bell_state,
    "ghz": ghz_state,
    "teleportation": quantum_teleportation,
    "qft4": qft_4,
    "iqft4": inverse_qft_4,
    "grover2": grover_2qubit,
    "shor15": shor_15,
    "deutsch_jozsa": deutsch_jozsa,
    "superdense": superdense_coding,
}


def get_template(name: str) -> dict[str, Any]:
    """Get a circuit template by name.

    Args:
        name: Template identifier.

    Returns:
        Template dict with name, description, num_qubits, circuit.

    Raises:
        ValueError: If template name is not found.
    """
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]()


def list_templates() -> list[dict[str, str]]:
    """List all available templates with names and descriptions."""
    result = []
    for key, fn in TEMPLATES.items():
        t = fn()
        result.append({
            "id": key,
            "name": t["name"],
            "description": t["description"],
            "num_qubits": t["num_qubits"],
        })
    return result
