import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit


# Ansatz function, could be at pulse level or circuit level
def apply_parametrized_circuit(qc: QuantumCircuit):
    """
    Define ansatz circuit to be played on Quantum Computer. Should be parametrized with Qiskit ParameterVector
    :param qc: Quantum Circuit instance to add the gates on
    :return:
    """
    # global n_actions
    # TODO: Parametrize the n_action (7)
    params = ParameterVector("theta", 7)
    qc.u(2 * np.pi * params[0], 2 * np.pi * params[1], 2 * np.pi * params[2], 0)
    #qc.rx(2*np.pi*params[0], 0) # Added
    #qc.u(2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], 0) # Added
    qc.u(2 * np.pi * params[3], 2 * np.pi * params[4], 2 * np.pi * params[5], 1)
    qc.rzx(2 * np.pi * params[6], 0, 1)
