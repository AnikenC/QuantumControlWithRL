from dataclasses import dataclass
from itertools import product
from typing import Dict, Optional

import torch

import numpy as np
from qiskit import IBMQ, ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.opflow import Zero
from qiskit.primitives import Estimator
from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Pauli,
    SparsePauliOp,
    Statevector,
    average_gate_fidelity,
    process_fidelity,
    state_fidelity,
)

from qiskit_ibm_runtime import QiskitRuntimeService, Session

from quantum_envs.envs.CNOTGateCalibration_v1.qconfig import QiskitConfig
from quantum_envs.envs.CNOTGateCalibration_v1.static import AbstractionLevel
from quantum_envs.envs.CNOTGateCalibration_v1.simple_utils import apply_parametrized_circuit

@dataclass
class QuantumEnvironment:
    abstraction_level: AbstractionLevel
    target: Dict  # TODO: Create its own type for clarity
    QUA_setup: Optional[Dict] = None

    def init_qiskit_setup(self):
        #IBMQ.load_account()
        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token="4c8ba089b15930777fc8d5501ed739bccaee58f5d9c3b13541776638606834593af6e9b1467ab224ffca2022914cea2e0782958a62c5eb4f2a84d008e12c42e2",
        )
        backend = service.backends(simulator=True)[
            0
        ]  # Simulation backend (mock quantum computer)
        options = {
            "seed_simulator": None, 
            "resilience_level": 0
        }

        self.qiskit_config = QiskitConfig(
            backend, service, apply_parametrized_circuit, options
        )

        self.n_qubits = 2
        self.n_shots = 1
        self.sampling_pauli_space = 100
        self.c_factor = 1.
        self.n_actions = 7

    def init_from_abstraction_level(self):
        if self.abstraction_level == "circuit":
            assert isinstance(self.qiskit_config, QiskitConfig), (
                "Qiskit setup argument not provided whereas circuit abstraction "
                "was provided"
            )
            self.q_register = QuantumRegister(self.n_qubits)
            self.c_register = ClassicalRegister(self.n_qubits)
            self.qc = QuantumCircuit(self.q_register)
            try:
                self.service = self.qiskit_config.service
                self.options = self.qiskit_config.options
                self.backend = self.qiskit_config.backend
                self.parametrized_circuit_func = self.qiskit_config.parametrized_circuit
            except KeyError:
                print(
                    "Circuit abstraction on Qiskit uses Runtime, need to provide"
                    "service, backend (Runtime), and options for the Estimator primitive"
                )
            return
        # TODO: Define pulse level (Schedule most likely, cf Qiskit Pulse doc)
        # TODO: Add a QUA program
        if self.QUA_setup is not None:
            self.qua_setup = self.QUA_setup
        elif self.qiskit_config is not None:
            self.q_register = QuantumRegister(self.n_qubits)
            self.c_register = ClassicalRegister(self.n_qubits)
            self.qc = QuantumCircuit(self.q_register)
            self.backend = self.qiskit_config.backend
            self.parametrized_circuit_func = self.qiskit_config.parametrized_circuit
            self.options = self.qiskit_config.options
            return

    def __post_init__(self):
        self.init_qiskit_setup()
        self.init_from_abstraction_level()
        self.Pauli_ops = [
            {"name": "".join(s), "matrix": Pauli("".join(s)).to_matrix()}
            for s in product(["I", "X", "Y", "Z"], repeat=self.n_qubits)
        ]

        self.d = 2**self.n_qubits  # Dimension of Hilbert space
        self.density_matrix = np.zeros([self.d, self.d], dtype="complex128")

        if (
            self.target.target_type == "state" or self.target.target_type is None
        ):  # Default mode is
            # State preparation if no argument target_type is found
            if "circuit" in self.target:
                self.target["dm"] = DensityMatrix(
                    self.target["circuit"] @ (Zero ^ self.n_qubits)
                )
            assert (
                "dm" in self.target
            ), "no DensityMatrix or circuit argument provided to target dictionary"
            assert (
                type(self.target["dm"]) == DensityMatrix
            ), "Provided dm is not a DensityMatrix object"
            self.target = self.calculate_chi_target_state(self.target)
            self.target_type = "state"
        elif self.target.target_type == "gate":
            # input_states = [self.calculate_chi_target_state(input_state) for input_state in target["input_states"]]
            self.target = self.target
            # self.target.input_states = input_states
            self.target_type = "gate"
        else:
            raise KeyError("target type not identified, must be either gate or state")

    def calculate_chi_target_state(self, target_state: Dict):
        """
        Calculate for all P
        :param target_state: Dictionary containing info on target state (name, density matrix)
        :return: target state
        """
        assert (
            "dm" in target_state
        ), "No input data for target state, provide DensityMatrix"
        # assert np.imag([np.array(target_state["dm"].to_operator()) @ self.Pauli_ops[k]["matrix"]
        #                 for k in range(self.d ** 2)]).all() == 0.
        target_state["Chi"] = np.array(
            [
                np.trace(
                    np.array(target_state["dm"].to_operator())
                    @ self.Pauli_ops[k]["matrix"]
                ).real
                for k in range(self.d**2)
            ]
        )  # Real part is taken to convert it in a good format,
        # but im is 0 systematically as dm is hermitian and Pauli is traceless
        return target_state

    def perform_action_gate_cal(self, actions):
        """
        Execute quantum circuit with parametrized amplitude, retrieve measurement result and assign rewards accordingly
        :param actions: action vector to execute on quantum system
        :return: Reward table (reward for each run in the batch), observations (measurement outcomes),
        obtained density matrix
        """ 

        repeat_size = 300

        if actions.ndim == 1:
            actions = np.expand_dims(actions, 0)
        angles, batch_size = np.array(actions), len(np.array(actions))

        assert self.target_type == "gate", "Environment only supports Gate Target Type"

        return_reward = 0.
        prc_fidelity = 0.0
        avg_fidelity = 0.0

        for i in range(repeat_size):
            index = np.random.randint(len(self.target.input_states))
            input_state = self.target.input_states[index]
            # Deduce target state to aim for by applying target operation on it
            target_state = {"target_type": "state"}
            if hasattr(input_state, "dm"):
                target_state["dm"] = Operator(self.target.gate) @ input_state.dm
            elif hasattr(input_state, "circuit"):
                target_state_fn = (
                    Operator(self.target.gate)
                    @ input_state.circuit
                    @ (Zero ^ self.n_qubits)
                )
                target_state["dm"] = DensityMatrix(target_state_fn)
            target_state = self.calculate_chi_target_state(target_state)

            # Direct fidelity estimation protocol  (https://doi.org/10.1103/PhysRevLett.106.230501)
            arr = target_state["Chi"]
            distribution = torch.distributions.Categorical(probs=torch.tensor(arr ** 2))
            k_samples = distribution.sample((self.sampling_pauli_space,))
            pauli_index = torch.unique(k_samples).numpy()

            reward_factor = np.round(
                [
                    self.c_factor * target_state["Chi"][p] / (self.d * torch.exp(distribution.log_prob(torch.tensor(p))).numpy())
                    for p in pauli_index
                ],
                5,
            )

            # Figure out which observables to sample
            # observables = [SparsePauliOp(self.Pauli_ops[p]["name"]) for p in pauli_index]
            observables = SparsePauliOp.from_list(
                [
                    (self.Pauli_ops[p]["name"], reward_factor[i])
                    for i, p in enumerate(pauli_index)
                ]
            )

            # Prepare input state
            self.qc.append(input_state.circuit.to_instruction(), input_state.register)

            # Apply parametrized quantum circuit (action)
            parametrized_circ = QuantumCircuit(self.n_qubits)
            self.parametrized_circuit_func(parametrized_circ)

            # Keep track of process for benchmarking purposes only
            if i == 0:
                for angle_set in angles:
                    qc_2 = parametrized_circ.bind_parameters(angle_set)
                    q_process = Operator(qc_2)
                    prc_fidelity += process_fidelity(q_process, Operator(self.target.gate))
                    avg_fidelity += average_gate_fidelity(q_process, Operator(self.target.gate))
                proc_fidelity = prc_fidelity / batch_size
                aver_fidelity = avg_fidelity / batch_size

            # Build full quantum circuit: concatenate input state prep and parametrized unitary
            self.qc.append(parametrized_circ.to_instruction(), input_state.register)
            # total_shots = self.n_shots * pauli_shots

            estimator = Estimator(options=self.options)
            job = estimator.run(
                circuits=[self.qc] * batch_size,
                observables=[observables] * batch_size,
                parameter_values=angles,
                shots=self.sampling_pauli_space,
            )

            self.qc.clear()  # Reset the QuantumCircuit instance for next iteration

            return_reward += job.result().values[0]

        mean_reward = return_reward / repeat_size

        return mean_reward, aver_fidelity, proc_fidelity