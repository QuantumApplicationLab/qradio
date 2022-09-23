import numpy as np
from numpy import matrix
from matplotlib import pyplot as plt
from pathlib import Path
from unitary_decomp import unitary_decomposition, manual_decomp
from qiskit.quantum_info.operators import Operator

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit import Aer, transpile, assemble

def diag(x):
    return np.asmatrix(np.diag(x))

def get_antnennas_response(xpos):

    n_ant = len(xpos)
    freq = 150e6    # measurement frequency in MHz
    c = 2.99792e8   # speed of light in m/s

    l = np.r_[-0.5, 0.2, 0.7]
    sigma = np.r_[0.8, 1, 0.4]

    g  = 1 + 0.3 * (np.random.normal(size=n_ant) \
       + 1j * np.random.normal(size=n_ant))

    A = np.matrix(np.exp(-(2 * np.pi * 1j * freq / c) * (xpos * l)))

    R = diag(g) @ A @ diag(sigma) @ A.H @ diag(g).H

    # sel = np.c_[6, 12, 18, 24, 11, 17, 23, 16, 22].T - 1
    sel = [i + j * n_ant + j             # indexing row-major
        for i in range(1, n_ant - 1)  # from first off-diagonal
                                        # not including the corner element
        for j in range(n_ant - i)]    # the first off-diagonal has (n_ant - 1) elements
                                        # and continuing, one less each time

    b = np.c_[np.log10(np.abs(R.flat[sel])),0].T

    return b

def apply_fixed_ansatz(circ, qubits, parameters):

    for iz in range (0, len(qubits)):
        circ.ry(parameters[0][iz], qubits[iz])

    circ.cz(qubits[0], qubits[1])
    circ.cz(qubits[2], qubits[0])

    for iz in range (0, len(qubits)):
        circ.ry(parameters[1][iz], qubits[iz])

    circ.cz(qubits[1], qubits[2])
    circ.cz(qubits[2], qubits[0])

    for iz in range (0, len(qubits)):
        circ.ry(parameters[2][iz], qubits[iz])

def apply_control_fixed_ansatz(circ, qubits, parameters, auxiliary, reg):

    for i in range (0, len(qubits)):
        circ.cry(parameters[0][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))

    circ.ccx(auxiliary, qubits[1], 4)
    circ.cz(qubits[0], 4)
    circ.ccx(auxiliary, qubits[1], 4)

    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)

    for i in range (0, len(qubits)):
        circ.cry(parameters[1][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))

    circ.ccx(auxiliary, qubits[2], 4)
    circ.cz(qubits[1], 4)
    circ.ccx(auxiliary, qubits[2], 4)

    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)

    for i in range (0, len(qubits)):
        circ.cry(parameters[2][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))

def get_controlled_operator_matrix(matrices, auxiliary, qubits):

    assert(auxiliary==0)
    nqbit = len(qubits)+1
    ctrl_matrices = []
    for m in matrices:
        cmat = np.eye(2**nqbit).astype('complex')
        cmat[1::2,1::2] = m
        ctrl_matrices.append(cmat)

    return ctrl_matrices

def apply_controlled_gate(circ, mat, auxiliary, qubits, name=None):

    qc1 = QuantumCircuit(len(qubits)+1, name=name)
    op = Operator(mat)
    qc1.append(op, [auxiliary] + qubits)
    circ.append(qc1, [auxiliary] + qubits)


def hadammard_test(circ, matrices, matnames, qubits,
                   auxiliary_index, parameters, imag=False):

    circ.h(auxiliary_index)

    if imag:
        circ.sdg(auxiliary_index)

    apply_fixed_ansatz(circ, qubits, parameters)

    for mat, name in zip(matrices, matnames):
        apply_controlled_gate(circ, mat, auxiliary_index, qubits, name)

    circ.h(auxiliary_index)


def special_hadammard_test(circ, op_mat, op_name, cub_mat, qubits,
                           auxiliary_index, parameters, reg, imag=False):

    circ.h(auxiliary_index)

    if imag:
        circ.sdg(auxiliary_index)

    apply_control_fixed_ansatz(circ, qubits, parameters, auxiliary_index, reg)

    apply_controlled_gate(circ, op_mat, 0, [1,2,3], op_name)

    apply_controlled_gate(circ, cub_mat, 0, [1,2,3], '$U_b$')

    circ.h(auxiliary_index)


# Implements the entire cost function on the quantum circuit
def calculate_cost_function(parameters, *args):

    overall_sum_1 = 0.0 + 0.0j

    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    gate_set, gate_name, conj_gate_set, conj_gate_name, coefficient_set, ctrl_ub_mat_dagger = args

    for i in range(0, len(gate_set)):
        for j in range(0, len(conj_gate_set)):

            multiply = coefficient_set[i].conj()*coefficient_set[j]

            beta_ij = 0.0 + 0.0j

            for imag in [False, True]:

                qctl = QuantumRegister(4)
                qc = ClassicalRegister(4)
                circ = QuantumCircuit(qctl, qc)

                backend = Aer.get_backend('aer_simulator')

                hadammard_test(circ, [conj_gate_set[i], gate_set[j]],
                         [conj_gate_name[i], gate_name[j]],
                         [1, 2, 3], 0, parameters,
                         imag=imag)

                circ.save_statevector()
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)

                result = job.result()
                outputstate = result.get_statevector(circ, decimals=100)
                o = outputstate

                m_sum = 0
                for l in range (0, len(o)):
                    if (l%2 == 1):
                        n = o[l]*o[l].conj()
                        m_sum+=n

                p01 = 1.0 - 2.0*m_sum

                if imag:
                    beta_ij += 1.0j * p01
                else:
                    beta_ij += p01

            overall_sum_1 += multiply*beta_ij


    overall_sum_2 = 0.0 + 0.0j

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):

            multiply = coefficient_set[i]*np.conjugate(coefficient_set[j])
            mult = 1
            gama_ij = 0.0 + 0.0j

            for extra in range(0, 2):

                term = 0.0 + 0.0j

                for imag in [False, True]:


                    qctl = QuantumRegister(5)
                    qc = ClassicalRegister(5)
                    circ = QuantumCircuit(qctl, qc)

                    backend = Aer.get_backend('aer_simulator')

                    if (extra == 0):
                        special_hadammard_test(circ, gate_set[i], gate_name[i],
                                                ctrl_ub_mat_dagger, [1, 2, 3],
                                                0, parameters, qctl,
                                                imag=imag)
                    if (extra == 1):
                        special_hadammard_test(circ, gate_set[j], gate_name[j],
                                                ctrl_ub_mat_dagger, [1, 2, 3],
                                                0, parameters, qctl,
                                                imag=imag)

                    circ.save_statevector()
                    t_circ = transpile(circ, backend)
                    qobj = assemble(t_circ)
                    job = backend.run(qobj)

                    result = job.result()
                    outputstate = result.get_statevector(circ, decimals=100)
                    o = outputstate

                    m_sum = 0
                    for l in range (0, len(o)):
                        if (l%2 == 1):
                            n = o[l]*o[l].conj()
                            m_sum+=n

                    p01 = 1.0 - 2.0*m_sum

                    if imag:
                        term += 1.0j * p01
                    else:
                        term += p01

                    mult = mult*(1-(2*m_sum))

                if extra == 0:
                    gama_ij += term
                else:
                    gama_ij *= term.conj()

            overall_sum_2 += multiply*gama_ij

    out = 1.0 - (overall_sum_2/overall_sum_1)
    print(out)

    return out.real


def sample_cost_function(parameters, *args):

    gate_set, gate_name, conj_gate_set, conj_gate_name, coefficient_set, ctrl_ub_mat_dagger = args

    overall_sum_1 = 0.0 + 0.0j

    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]

    for i in range(0, len(gate_set)):
        for j in range(0, len(conj_gate_set)):

            beta_ij = 0.0 + 0.0j

            for imag in [False, True]:

                qctl = QuantumRegister(5)
                qc = ClassicalRegister(1)
                circ = QuantumCircuit(qctl, qc)

                backend = Aer.get_backend('aer_simulator')

                multiply = coefficient_set[i].conj()*coefficient_set[j]

                hadammard_test(circ, [conj_gate_set[i], gate_set[j]],
                            [conj_gate_name[i], gate_name[j]],
                            [1, 2, 3], 0, parameters,
                            imag=imag)

                circ.measure(0, 0)

                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ, shots=10000)
                job = backend.run(qobj)

                result = job.result()
                outputstate = result.get_counts(circ)

                if ('1' in outputstate.keys()):
                    m_sum = float(outputstate["1"])/100000
                else:
                    m_sum = 0

                p01 = 1.0 - 2.0*m_sum

                if imag:
                    beta_ij += 1.0j * p01
                else:
                    beta_ij += p01

            overall_sum_1 += multiply*beta_ij


    overall_sum_2 = 0.0 + 0.0j

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):

            multiply = coefficient_set[i]*np.conjugate(coefficient_set[j])
            mult = 1
            gama_ij = 0.0 + 0.0j

            for extra in range(0, 2):

                term = 0.0 + 0.0j

                for imag in [False, True]:

                    qctl = QuantumRegister(5)
                    qc = ClassicalRegister(1)
                    circ = QuantumCircuit(qctl, qc)

                    backend = Aer.get_backend('aer_simulator')

                    if (extra == 0):
                        special_hadammard_test(circ, gate_set[i], gate_name[i],
                                                ctrl_ub_mat_dagger, [1, 2, 3],
                                                0, parameters, qctl,
                                                imag=imag)
                    if (extra == 1):
                        special_hadammard_test(circ, gate_set[j], gate_name[j],
                                                ctrl_ub_mat_dagger, [1, 2, 3],
                                                0, parameters, qctl,
                                                imag=imag)

                    circ.measure(0, 0)

                    t_circ = transpile(circ, backend)
                    qobj = assemble(t_circ, shots=10000)
                    job = backend.run(qobj)

                    result = job.result()
                    outputstate = result.get_counts(circ)

                    if ('1' in outputstate.keys()):
                        m_sum = float(outputstate["1"])/100000
                    else:
                        m_sum = 0

                    p01 = 1.0 - 2.0*m_sum

                    if imag:
                        term += 1.0j * p01
                    else:
                        term += p01

                    mult = mult*(1-2*m_sum)

                if extra == 0:
                    gama_ij += term
                else:
                    gama_ij *= np.conjugate(term)

            overall_sum_2 += multiply*gama_ij


    out = 1.0-(overall_sum_2/overall_sum_1)
    print(out)

    return out.real