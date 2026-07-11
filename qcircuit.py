import math
import random

import numpy as np
from qrf import RandomForestForQuantumPrediction
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute, Aer, BasicAer
from qiskit.circuit.library import ZGate, RGate, CCXGate, XGate, MCXGate, RC3XGate, QFT


def int_to_bin(int_numb, length):
    s = [0] * length
    i = 0
    while i < length:
        s[i] = int_numb % 2
        int_numb //= 2
        i += 1
    return s


def mct_ucg(num_control):
    qc = QuantumCircuit(QuantumRegister(num_control + 1))
    gate_list = []
    for _ in range(2 ** num_control - 1):
        gate_list.append([[1, 0], [0, 1]])
    gate_list.append([[0, 1], [1, 0]])
    gate_list = np.array(gate_list)
    qc.uc(list(gate_list), list(range(num_control)), num_control, True)
    return qc.to_gate()


def dec_ucg(num_control=3):
    qc = QuantumCircuit(QuantumRegister(num_control + 1))
    gate_list = []
    for i in range(2 ** 3):
        gate_list.append([[1, 0], [0, 1]])
    gate_list[3] = [[0, 1], [1, 0]]  # 011 x=0, |help>=1, |control>=1
    gate_list = np.array(gate_list)
    qc.uc(list(gate_list), list(range(num_control)), num_control, True)
    return qc.to_gate()


def int_to_binary_str(a, length):
    s = ''
    while a > 0:
        s = str(a % 2) + s
        a //= 2
    if len(s) < length:
        s = '0' * (length - len(s)) + s
    return s


def print_state_vector(vector, n, print_proba=True):
    for i in range(len(vector)):
        if abs((vector[i] ** 2).real) >= 0.0001:
            if print_proba:
                print(int_to_binary_str(i, n), abs((vector[i] ** 2)))
            else:
                print(int_to_binary_str(i, n), vector[i])
    print()


def check_measure(counts, lim):
    if '0' in counts and counts['0'] >= lim:
        return True
    return False


# предсказание на дереве, доводит до листового узла
def tree_predict_controlled(tree, x_len):
    qc = QuantumCircuit(
        QuantumRegister(1),  # |class>
        QuantumRegister(tree.h - 1),  # |j>
        QuantumRegister(1),  # |ancj>
        QuantumRegister(x_len),  # |x>
        QuantumRegister(1),  # |control>
        QuantumRegister(1)  # |anc_mcx>
    )

    # Improved readability by extracting base indices
    class_idx = 0
    j_start = 1
    anc_j_idx = j_start + tree.h - 1
    x_start = anc_j_idx + 1
    control_idx = x_start + x_len
    anc_mcx_idx = control_idx + 1

    qc.cx(x_start + x_len, class_idx + 1)

    for i in range(1, 2 ** (tree.h - 2)):
        bin_i = int_to_bin(i, tree.h - 2)

        for j in range(len(bin_i)):
            if bin_i[j] == 0:
                qc.cx(x_start + x_len, j + 1)

        if tree.h >= 8:
            qc.mcx(
                list(range(j_start, anc_j_idx)) + [x_start + x_len],
                [j_start + tree.h - 1],
                anc_mcx_idx,
                mode='recursion'
            )
        else:
            qc.append(mct_ucg(tree.h - 1),
                      list(range(j_start, anc_j_idx + 1)) + [j_start + tree.h - 1])

        for j in range(len(bin_i)):
            if bin_i[j] == 0:
                qc.cx(x_start + x_len, j + 1)

        # mult_by_2.controlled()
        for k in range(tree.h - 2, 0, -1):
            qc.rcccx([x_start + x_len], [j_start + tree.h - 1], [j_start + k], [j_start + k - 1])
            qc.rcccx([x_start + x_len], [j_start + tree.h - 1], [j_start + k - 1], [j_start + k])
            qc.rcccx([x_start + x_len], [j_start + tree.h - 1], [j_start + k], [j_start + k - 1])

        # inc.controlled()
        qc.rcccx(
            [j_start + tree.h - 1],
            [j_start + tree.h + tree.attr_ind[i]],
            [x_start + x_len],
            [class_idx + 1]
        )

        # inverse |ancj>
        qc.append(
            mct_ucg(2),
            [j_start + math.floor(math.log2(i * 2))] + [x_start + x_len] + [anc_j_idx]
        )

    for i in range(2 ** (tree.h - 2), 2 ** (tree.h - 1)):
        bin_i = int_to_bin(i, tree.h - 1)
        for j in range(len(bin_i)):
            if bin_i[j] == 0:
                qc.cx(x_start + x_len, j + 1)

        if tree.h >= 8:
            qc.mcx(
                list(range(j_start, j_start + tree.h)) + [x_start + x_len],
                [j_start + tree.h - 1],
                anc_mcx_idx,
                mode='recursion'
            )
        else:
            qc.append(mct_ucg(tree.h),
                      list(range(j_start, j_start + tree.h + 1)) + [j_start + tree.h - 1])

        for j in range(len(bin_i)):
            if bin_i[j] == 0:
                qc.cx(x_start + x_len, j + 1)

        # mult_by_2.controlled()
        for k in range(tree.h - 2, 0, -1):
            qc.rcccx([j_start + tree.h - 1], [j_start + k], [x_start + x_len], [j_start + k - 1])
            qc.rcccx([j_start + tree.h - 1], [j_start + k - 1], [x_start + x_len], [j_start + k])
            qc.rcccx([j_start + tree.h - 1], [j_start + k], [x_start + x_len], [j_start + k - 1])

        # dec.controlled()
        qc.append(
            dec_ucg(),
            [x_start + x_len] + [j_start + tree.h - 1] + [j_start + tree.h + tree.attr_ind[i]] + [class_idx + 1]
        )

        # inverse ancj
        bin_i = int_to_bin(i * 2, tree.h - 1)
        for j in range(1, len(bin_i)):
            if bin_i[j] == 0:
                qc.cx(x_start + x_len, j + 1)

        if tree.h >= 8:
            qc.mcx(
                list(range(j_start + 1, j_start + tree.h)) + [x_start + x_len],
                [j_start + tree.h - 1],
                anc_mcx_idx,
                mode='recursion'
            )
        else:
            qc.append(mct_ucg(tree.h - 1),
                      list(range(j_start + 1, j_start + tree.h)) + [x_start + x_len] + [j_start + tree.h - 1])

        for j in range(1, len(bin_i)):
            if bin_i[j] == 0:
                qc.cx(x_start + x_len, j + 1)

    return qc.to_gate()


def prepare_quantum_x(x):
    qc = QuantumCircuit(QuantumRegister(len(x)))
    for i in range(len(x)):
        if x[i] == '1':
            qc.x(i)
    return qc.to_gate()


def rf_predict(rf=None, len_x=None):
    len_i = math.ceil(math.log2(rf.n_trees))
    qc = QuantumCircuit(
        QuantumRegister(1),  # |class>
        QuantumRegister(rf.h_trees - 1),  # |j>
        QuantumRegister(1),  # |anc_j>
        QuantumRegister(len_i),  # |i>
        QuantumRegister(1),  # |anc_i>
        QuantumRegister(len_x),  # |X>
        QuantumRegister(1)  # |anc_mcx>
    )

    i_start = 1 + rf.h_trees
    anc_rf_idx = 1 + rf.h_trees + len_i
    ancilla_idx = anc_rf_idx + 1 + len_x

    qc.h(list(range(i_start, i_start + len_i)))

    for i in range(rf.n_trees):
        bin_i = int_to_bin(i, len_i)
        for j in range(len(bin_i)):
            if bin_i[j] == 0:
                qc.x(i_start + j)

        if len_i >= 8:
            qc.mcx(
                list(range(i_start, i_start + len_i)),
                [anc_rf_idx],
                [ancilla_idx],
                mode='recursion'
            )
        else:
            qc.append(mct_ucg(len_i), list(range(i_start, i_start + len_i + 1)))

        qc.append(
            tree_predict_controlled(rf.trees[i], len_x),
            list(range(1, 1 + rf.h_trees))
            + list(range(anc_rf_idx, anc_rf_idx + len_x))
            + [anc_rf_idx]
            + [ancilla_idx]
        )

        qc.append(mct_ucg(len_i), list(range(i_start, i_start + len_i + 1)))

        for j in range(len(bin_i)):
            if bin_i[j] == 0:
                qc.x(i_start + j)

    # Added missing target qubit index (0) for ucry rotation
    qc.ucry(
        rf.angle_list, list(range(1, 1 + rf.h_trees - 1)) + list(range(i_start, i_start + len_i)), 0
    )
    return qc.to_gate()


def oracle_for_binary_classification():
    qc = QuantumCircuit(QuantumRegister(1))
    qc.x(0)
    qc.z(0)
    qc.x(0)
    return qc.to_gate()


def s0(n_qbits):
    qc = QuantumCircuit(QuantumRegister(n_qbits))
    qc.x(list(range(n_qbits)))
    qc.append(ZGate().control(n_qbits - 1), list(range(n_qbits)))  # декомпозиция
    qc.x(list(range(n_qbits)))
    return qc.to_gate()


def get_proba(rf, X, lim=0):
    #число запусков схемы = 1
    len_X = len(X)
    len_i = math.ceil(math.log(rf.n_trees, 2))
    len_j = rf.h_trees
    n = 1 + len_j + len_i + len_X + 1 + 1

    for k in range(0, 16):
        qc = QuantumCircuit(
            QuantumRegister(1),  # |class>
            QuantumRegister(len_j - 1),  # |j>
            QuantumRegister(1),  # |help>
            QuantumRegister(len_i),  # |i>
            QuantumRegister(1),  # |help>
            QuantumRegister(len_X),  # |X>
            QuantumRegister(1),  # |anc_mcx> for tree_predict
            ClassicalRegister(1)
        )
        qc.append(prepare_quantum_x(X), list(range(1 + len_j + len_i + 1, 1 + len_j + len_i + 1 + len_X)))
        qc.append(rf_predict(rf, len_X), list(range(n)))

        for _ in range(k):
            # oracle
            qc.append(oracle_for_binary_classification(), [0])

            # diffuser
            qc.append(rf_predict(rf, len_X).inverse(), list(range(n)))
            qc.append(s0(n - len_X - 2), list(range(n - len_X - 2)))
            qc.append(rf_predict(rf, len_X), list(range(n)))

        qc.measure(0, 0)
        job = execute(qc, BasicAer.get_backend('qasm_simulator'), shots=1)
        counts = job.result().get_counts(qc)

        success = check_measure(counts, lim)
        if success:
            if k != 0:
                return k, math.sin((math.pi / 4) / k) ** 2
            else:
                return 0, 1
    return '>15', 0


def QSearch(rf, X, lim):
    if rf.n_trees <= 0 or len(X) == 0:
        raise ValueError("Random forest must have trees and input X must not be empty.")

    len_X = len(X)
    len_i = math.ceil(math.log(rf.n_trees, 2))
    len_j = rf.h_trees
    n = 1 + len_j + len_i + len_X + 1 + 1
    l = 0
    c = random.random() + 1

    def _build_base_circuit():
        qc_local = QuantumCircuit(
            QuantumRegister(1),  # |class>
            QuantumRegister(len_j - 1),  # |j>
            QuantumRegister(1),  # |help>
            QuantumRegister(len_i),  # |i>
            QuantumRegister(1),  # |help>
            QuantumRegister(len_X),  # |X>
            QuantumRegister(1),  # |anc_mcx> for tree_predict
            ClassicalRegister(1)
        )
        qc_local.append(prepare_quantum_x(X), list(range(1 + len_j + len_i + 1, 1 + len_j + len_i + 1 + len_X)))
        qc_local.append(rf_predict(rf, len_X), list(range(n)))
        return qc_local

    qc = _build_base_circuit()
    qc.measure([0], [0])
    job = execute(qc, Aer.get_backend('qasm_simulator'), shots=100)
    counts = job.result().get_counts(qc)

    if check_measure(counts, lim):
        return 0, 1

    while True:
        l += 1
        M = math.ceil(c ** l)
        j = random.randint(1, M + 1)

        qc = _build_base_circuit()
        # применить алгоритм A\0>
        qc.append(rf_predict(rf, len_X), list(range(n)))
        for _ in range(j):
            q(qc, n, len_X, -1)

        qc.measure([0], [0])
        job = execute(qc, Aer.get_backend('qasm_simulator'), shots=100)
        counts = job.result().get_counts(qc)

        if check_measure(counts, lim):
            return j, math.sin((math.pi / 4) / j) ** 2


def q(target_rf, qc, n, len_X, cntrl):
    if cntrl > -1:
        # oracle
        qc.append(oracle_for_binary_classification().control(), [cntrl] + [0])
        # diffuser
        qc.append(rf_predict(target_rf, len_X).inverse().control(), [cntrl] + list(range(n)))
        qc.append(s0(n - len_X - 2).control(), [cntrl] + list(range(n - len_X - 2)))
        qc.append(rf_predict(target_rf, len_X).control(), [cntrl] + list(range(n)))
    else:
        # oracle
        qc.append(oracle_for_binary_classification(), [0])
        # diffuser
        qc.append(rf_predict(target_rf, len_X).inverse(), list(range(n)))
        qc.append(s0(n - len_X - 2), list(range(n - len_X - 2)))
        qc.append(rf_predict(target_rf, len_X), list(range(n)))


def est_amp(k, X, rf):
    """
    k the number of qubits |j> Q^j
    X an input object
    rf a random forest model
    """
    if rf.n_trees <= 0 or len(X) == 0:
        raise ValueError("Random forest must have trees and input X must not be empty.")

    len_i = math.ceil(math.log(rf.n_trees, 2))
    len_j = rf.h_trees
    len_X = len(X)
    n = 1 + len_j + len_i + len_X + 1 + 1

    amp_reg_start = 1 + len_j + len_i + 1 + len_X + 1

    qc = QuantumCircuit(
        QuantumRegister(1),  # |class>
        QuantumRegister(len_j - 1),  # |j>
        QuantumRegister(1),  # |anc_j>
        QuantumRegister(len_i),  # |i>
        QuantumRegister(1),  # |anc_i>
        QuantumRegister(len_X),  # |X>
        QuantumRegister(1),  # |anc_mcx> for tree_predict
        QuantumRegister(k),
        ClassicalRegister(k)
    )

    qc.append(prepare_quantum_x(X), list(range(1 + len_j + len_i + 1, 1 + len_j + len_i + 1 + len_X)))
    qc.append(rf_predict(rf, len_X), list(range(n)))

    qc.append(QFT(k), list(range(amp_reg_start, amp_reg_start + k)))
    q(rf, qc, n, len_X, amp_reg_start)

    if k >= 2:
        for _ in range(2):
            q(rf, qc, n, len_X, amp_reg_start + 1)

        if k >= 3:
            for _ in range(4):
                q(rf, qc, n, len_X, amp_reg_start + 2)

            if k >= 4:
                for _ in range(8):
                    q(rf, qc, n, len_X, amp_reg_start + 3)

                if k >= 5:
                    for _ in range(16):
                        q(rf, qc, n, len_X, amp_reg_start + 4)

                    if k >= 6:
                        for _ in range(32):
                            q(rf, qc, n, len_X, amp_reg_start + 5)

                        if k >= 7:
                            for _ in range(64):
                                q(rf, qc, n, len_X, amp_reg_start + 6)

    qc.append(QFT(k).inverse(), list(range(amp_reg_start, amp_reg_start + k)))

    qc.measure(list(range(amp_reg_start, amp_reg_start + k)), list(range(k)))
    job = execute(qc, BasicAer.get_backend('qasm_simulator'), shots=100)
    counts = job.result().get_counts(qc)
    stat_y = []
    for c in counts:
        stat_y.append((counts[c], int(c[::-1], 2)))
    return counts, stat_y


def test():
    rf = RandomForestForQuantumPrediction(3)
    X_test = []
    for i in range(2 ** 3):
        x = int_to_binary_str(i, 3)
        X_test.append(x)
    len_X = 3
    len_i = math.ceil(math.log(rf.n_trees, 2))
    len_j = rf.h_trees
    n = 1 + len_j + len_i + len_X + 1 + 1
    res_stat = {}
    for x in X_test:
        res_stat[x] = 0
    for _ in range(100):
        for i in range(len(X_test)):  # len(X_test)
            qc = QuantumCircuit(
                QuantumRegister(1),  # |class>
                QuantumRegister(len_j - 1),  # |j>
                QuantumRegister(1),  # |anc_j>
                QuantumRegister(len_i),  # |i>
                QuantumRegister(1),  # |anc_i>
                QuantumRegister(len_X),  # |X>
                QuantumRegister(1),  # |anc_mcx> for tree_predict
                ClassicalRegister(1)
            )
            qc.append(prepare_quantum_x(X_test[i]), list(range(1 + len_j + len_i + 1, 1 + len_j + len_i + 1 + len_X)))
            qc.append(rf_predict(rf, len_X), list(range(n)))
            qc.measure(0, 0)
            job = execute(qc, BasicAer.get_backend('qasm_simulator'), shots=1)
            counts = job.result().get_counts(qc)
            if rf.predict_proba(X_test[i]) > 0.5:
                if '0' not in counts:
                    res_stat[X_test[i]] += 1
            else:
                if '1' not in counts:
                    res_stat[X_test[i]] += 1
    print(res_stat)
