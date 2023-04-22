# Target gate: CNOT gate
from qiskit.extensions import CXGate
from qiskit.opflow import Zero, One, Plus, Minus, H, I, X, CX, S, Z
from qiskit.opflow.primitive_ops.pauli_op import PauliOp

from dataclasses import dataclass, field

from typing import ClassVar, List

@dataclass
class QinputState:
    #TODO: Make a list of commonly used input states in an enumerator
    name: str
    circuit: PauliOp
    register: List[int] = field(default_factory=lambda: [0, 1])

@dataclass
class CNOT:
    circuit_plus_i = S @ H
    circuit_minus_i = S @ H @ X
    
    gate_name: str = "CNOT"
    target_type:str = "gate"
    
    def __post_init__(self):
        self.gate = CXGate(self.gate_name)
        self.input_states=[
            QinputState("|00>", I ^ 2),
            QinputState("|01>", X ^ I),
            QinputState("|10>", I ^ X),
            QinputState("|11>", X ^ X),
            QinputState("|+_1>", X ^ H),
            QinputState("|0_->", (H @ X) ^ I),
            QinputState("|+_->", (H @ X) ^ H),
            QinputState("|1_->", (H @ X) ^ X),
            QinputState("|+_0>", I ^ H),
            QinputState("|0_->", (H @ X) ^ I),
            QinputState("|i_0>", I ^ self.circuit_plus_i),
            QinputState("|i_1>", X ^ self.circuit_plus_i),
            QinputState("|0_i>", self.circuit_plus_i ^ I),
            QinputState("|i_i>", self.circuit_plus_i ^ self.circuit_plus_i),
            QinputState("|i_->", (H @ X) ^ self.circuit_plus_i),
            QinputState("|+_i->", self.circuit_minus_i ^ H)]