from dataclasses import dataclass
from typing import Callable, Dict

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit.circuit import QuantumCircuit


@dataclass
class QiskitConfig:
    backend: IBMBackend
    service: QiskitRuntimeService
    parametrized_circuit: Callable[QuantumCircuit, None]
    options: Dict