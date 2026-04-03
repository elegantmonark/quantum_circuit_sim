"""
QSIM -- Quantum Circuit Simulator

FastAPI application serving the quantum circuit simulator API
and the single-page frontend UI.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


from gates import get_gate_catalogue
from simulator import simulate_circuit
from circuit_templates import get_template, list_templates

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="QSIM -- Quantum Circuit Simulator",
    description="A production-quality quantum circuit simulator with visual circuit builder.",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# --- Request / Response Models ---


class GateOperation(BaseModel):
    gate: str
    target: int
    control: int | None = None
    control2: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class NoiseConfig(BaseModel):
    depolarizing: float = Field(default=0.0, ge=0.0, le=1.0)
    dephasing: float = Field(default=0.0, ge=0.0, le=1.0)
    amplitude_damping: float = Field(default=0.0, ge=0.0, le=1.0)


class SimulateRequest(BaseModel):
    num_qubits: int = Field(ge=1, le=10)
    circuit: list[list[GateOperation]]
    shots: int | None = Field(default=None, ge=1, le=100000)
    noise: NoiseConfig | None = None


# --- Routes ---


@app.get("/")
async def index(request: Request):
    """Serve the main frontend UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "qsim", "version": "2.0.0"}


@app.get("/gates")
async def gates():
    """Return the full gate catalogue with matrices and descriptions."""
    return {"gates": get_gate_catalogue()}


@app.post("/simulate")
async def simulate(req: SimulateRequest):
    """Run a quantum circuit simulation."""
    try:
        circuit_data = [
            [op.model_dump() for op in step] for step in req.circuit
        ]
        noise_dict = req.noise.model_dump() if req.noise else None
        result = simulate_circuit(req.num_qubits, circuit_data, shots=req.shots, noise=noise_dict)
        return result
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Simulation failed: {str(e)}"})


@app.get("/templates")
async def get_templates():
    """List all available circuit templates."""
    return {"templates": list_templates()}


@app.get("/templates/{name}")
async def get_template_circuit(name: str):
    """Get a specific circuit template."""
    try:
        template = get_template(name)
        return template
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
