from .encoded_circuit import EncodedCircuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister # type: ignore
from qiskit.circuit.library.standard_gates import XGate # type: ignore
import warnings

class SteaneCodeCircuit(EncodedCircuit):
    '''
    A class representing a Steane code quantum error-correcting circuit.

    This class implements the [[7,1,3]] Steane code, which encodes a single logical qubit
    into seven physical qubits. It provides methods to encode, apply quantum gates, correct
    errors, and manage interactions between the logical and physical layers. The class inherits
    from the `EncodedCircuit` class, adding further functionality such as error injection and
    circuit measurement.

    Attributes:
        logical_qubit_count (int): The number of logical qubits in the circuit.
        physical_qubit_count (int): The number of physical qubits (excluding ancillas for error correction and measurement).
        encode_applied (bool): Indicates whether the encoding step has been applied.
        logical_qubit_register (QuantumRegister): Register of logical qubits.
        logical_bit_register (ClassicalRegister): Classical register to store logical measurement results.
        logical_quantum_circuit (QuantumCircuit): Logical layer of the quantum circuit.
        physical_qubit_register (QuantumRegister): Register of physical qubits.
        physical_ancilla_qubit_register (QuantumRegister): Register of ancilla qubits used for error correction.
        physical_measurement_qubit_register (QuantumRegister): Register of qubits used for measurements.
        physical_bit_register (ClassicalRegister): Classical register to store physical measurement results.
        physical_quantum_circuit (QuantumCircuit): Physical layer of the quantum circuit.

    Methods:
        get_physical_qubits_from_logical(logical_qubits: list | int) -> list[int]:
            Maps logical qubits to their corresponding physical qubits.

        encode(re_encode: bool = False) -> None:
            Encodes each logical qubit using the Steane code. If encoding is already applied, a
            warning is issued unless `re_encode` is set to True.

        correct() -> None:
            Applies quantum error correction to the physical circuit.

        x(logical_qubits: list | int) -> None:
            Applies the Pauli-X gate to the specified logical and corresponding physical qubits.

        y(logical_qubits: list | int) -> None:
            Applies the Pauli-Y gate to the specified logical and corresponding physical qubits.

        z(logical_qubits: list | int) -> None:
            Applies the Pauli-Z gate to the specified logical and corresponding physical qubits.

        h(logical_qubits: list | int) -> None:
            Applies the Hadamard gate to the specified logical and corresponding physical qubits.

        cx(control: list | int, target: list | int) -> None:
            Applies a CNOT gate between the specified control and target logical qubits in both the
            logical and physical quantum circuits.

        measure_all() -> None:
            Measures all logical and physical qubits, storing the results in their respective classical registers.

        append(U, logical_qubits: list | int) -> None:
            Appends a quantum gate to the logical and physical circuits unless encoding has been applied.
            If encoding is applied, a warning is issued.

        append_unitary_error(E, physical_qubits: list | int) -> None:
            Appends an error gate to the specified physical qubits in the physical circuit.

        barrier(num_qubits: int = None, label: str = None) -> None:
            Appends a barrier to the logical and physical circuits for visualization and grouping.

        draw_logical_circuit(output: str = 'mpl') -> None:
            Draws the logical quantum circuit using the specified output format.

        draw_physical_circuit(output: str = 'mpl') -> None:
            Draws the physical quantum circuit using the specified output format.
    '''

    def __init__(self, logical_qubit_count):
        # 7 physical qubits for each logical qubit
        super().__init__(logical_qubit_count, physical_qubits_per_logical=7)

        # Redefine this part since we need to add the ancillas register and the registers used to measure
        self.gates_before_encode = False # Indicate is a gate had been applied before encode
        self.physical_qubit_register = QuantumRegister(self.physical_qubit_count, name = 'physical_qubits')
        self.physical_ancilla_qubit_register = QuantumRegister(6*self.logical_qubit_count, name = 'ancilla_qubits')
        self.physical_measurement_qubit_register = QuantumRegister(self.logical_qubit_count, name = 'measure_register')
        self.physical_bit_register = ClassicalRegister(self.logical_qubit_count, name = 'physical_bits')
        self.physical_quantum_circuit = QuantumCircuit(self.physical_qubit_register,
                                                       name = 'physical_circuit')


    def get_physical_qubits_from_logical(self, logical_qubits: list | int):
        '''
        Maps logical qubits to their corresponding physical qubits in the Steane code.

        If a single integer is provided, it will be converted into a list containing that integer.

        Parameters:
            logical_qubits (list[int] | int): A logical qubit or list of logical qubits to map.

        Returns:
            list[int]: The list of physical qubit indices representing the given logical qubits.
        '''

        # The function requires logical_qubit to be a list. If logical_qubits is of type int we create a list containing only logical_qubits:
        if isinstance(logical_qubits, int):
            logical_qubits = [logical_qubits]


        physical_qubits = []
        for i in logical_qubits:
            for j in range(7*i, 7*i + 7):
                physical_qubits.append(j)

        return physical_qubits


    def encode(self, re_encode = False, append = False, initial_state=None):
        '''
        Encodes each logical qubit using the Steane code.

        This method applies the CSS encoder to map each logical qubit into seven physical qubits.
        If encoding has already been applied and `re_encode` is set to `False`, a warning is issued
        and the method exits without re-encoding. If `re_encode` is `True`, the encoding is re-applied
        regardless of the `encode_applied` flag.

        Parameters:
            re_encode (bool): If `True`, forces re-encoding even if it was previously applied.
                            Default is `False`.
            append (bool): If `True`, it calls a function append_CSS_encoder(qc, qubit_list) which appends the encoder to a circuit.
                            If `False` it creates an auxiliary circuit, append the encoder to the auxiliary circuit, convert the
                            auxiliary circuit to a gate and then append this gate to the circuit.
                            Default is `False`.
            initial_state (String): Indicate the initial state, can be '0', '1' or None. If None we use the universal encoder
                                     Default is None

        Returns:
            None

        Side Effects:
            - Sets `self.encode_applied` to True if encoding is performed.
            - Issues a UserWarning if encoding is skipped due to being already applied and `re_encode` is `False`.
        '''

        if(not self.encode_applied or re_encode):
            if (not self.gates_before_encode) or initial_state == None:
                if append:
                    append_CSS_encoder(self.physical_quantum_circuit, self.physical_qubit_register, initial_state=initial_state)

                else:
                    encoder = CSS_encoder(initial_state=initial_state)
                    for i in range(self.logical_qubit_count):
                        self.physical_quantum_circuit.append(encoder,  self.physical_qubit_register[7*i:7*i + 7])

            else:
                warning_text = "\r\nWARNING! You applied gates to the circuit before encoding. Use Universal Encoder by Default."
                warnings.warn(warning_text, category=UserWarning)
                if append:
                    append_CSS_encoder(self.physical_quantum_circuit, self.physical_qubit_register)

                else:
                    encoder = CSS_encoder()
                    for i in range(self.logical_qubit_count):
                        self.physical_quantum_circuit.append(encoder,  self.physical_qubit_register[7*i:7*i + 7])

            self.encode_applied = True

        else:
            warning_text = "\r\nWARNING! Ecoder has already been applied. Use re_encode = True if you want to apply it regardless."
            warnings.warn(warning_text, category=UserWarning)


    def correct(self, append = False):
        '''
        Applies quantum error correction to the physical circuit.

        This method appends the CSS error correction gate to each set of 7 physical qubits
        along with 6 ancilla qubits. If the `correct()` method has already been applied,
        an error is raised. The ancilla register is added to the circuit only when this
        method is called.

        Parameters:
            append (bool): If `True`, it calls a function append_CSS_correction(qc, qubit_list, ancillas) which appends the correction.
                If `False` it creates an auxiliary circuit, append the correction to the auxiliary circuit, convert the
                auxiliary circuit to a gate and then append this gate to the circuit.
                Default is `False`.

        Returns:
            None

        Raises:
            RuntimeError: If `correct()` has already been applied.

        Side Effects:
            - Adds the ancilla qubit register to the physical circuit.
            - Modifies the physical circuit by appending error correction gates.
        '''
        self.physical_quantum_circuit.add_register(self.physical_ancilla_qubit_register)

        if append:
            append_CSS_correction(self.physical_quantum_circuit, self.physical_qubit_register, self.physical_ancilla_qubit_register)

        else:

            error_correction = CSS_correction()

            for i in range(self.logical_qubit_count):
                self.physical_quantum_circuit.append(error_correction, self.physical_qubit_register[7*i: 7*i + 7] + self.physical_ancilla_qubit_register[6*i: 6*i + 6])


    def x(self, logical_qubits: list | int):
        '''
        Applies the Pauli-X gate to specified logical qubits in both the logical and physical circuits.

        If encoding has been applied (`encode_applied` is True), the X-gate is applied to the
        corresponding physical qubits. If not, the X-gate is applied directly to the logical qubits
        and their mapped physical qubits.

        Parameters:
            logical_qubits (list[int] | int): Logical qubit(s) where the X-gate will be applied.

        Returns:
            None
        '''

        self.logical_quantum_circuit.x(logical_qubits)

        if(self.encode_applied):
            self.physical_quantum_circuit.x(self.get_physical_qubits_from_logical(logical_qubits))

        else:
            self.gates_before_encode = True
            if isinstance(logical_qubits, int):
                logical_qubits = [logical_qubits]
            self.physical_quantum_circuit.x([7*i for i in logical_qubits])


    def z(self, logical_qubits: list | int):
        '''
        Applies the Pauli-Z gate to specified logical qubits in both the logical and physical circuits.

        If encoding has been applied (`encode_applied` is True), the Z-gate is applied to the
        corresponding physical qubits. If not, the Z-gate is applied directly to the logical qubits
        and their mapped physical qubits.

        Parameters:
            logical_qubits (list[int] | int): Logical qubit(s) where the Z-gate will be applied.

        Returns:
            None
        '''

        self.logical_quantum_circuit.z(logical_qubits)

        if(self.encode_applied):
            self.physical_quantum_circuit.z(self.get_physical_qubits_from_logical(logical_qubits))

        else:
            self.gates_before_encode = True
            if isinstance(logical_qubits, int):
                logical_qubits = [logical_qubits]
            self.physical_quantum_circuit.z([7*i for i in logical_qubits])


    def y(self, logical_qubits: list | int):
        '''
        Applies the Pauli-Y gate to specified logical qubits in both the logical and physical circuits.

        If encoding has been applied (`encode_applied` is True), the Y-gate is applied to the
        corresponding physical qubits. If not, the Y-gate is applied directly to the logical qubits
        and their mapped physical qubits.

        Parameters:
            logical_qubits (list[int] | int): Logical qubit(s) where the Y-gate will be applied.

        Returns:
            None
        '''

        self.logical_quantum_circuit.y(logical_qubits)

        if(self.encode_applied):
            self.physical_quantum_circuit.y(self.get_physical_qubits_from_logical(logical_qubits))

        else:
            self.gates_before_encode = True
            if isinstance(logical_qubits, int):
                logical_qubits = [logical_qubits]
            self.physical_quantum_circuit.y([7*i for i in logical_qubits])


    def h(self, logical_qubits: list | int):
        '''
        Applies the Hadamard gate H to specified logical qubits in both the logical and physical circuits.

        If encoding has been applied (`encode_applied` is True), the H-gate is applied to the
        corresponding physical qubits. If not, the H-gate is applied directly to the logical qubits
        and their mapped physical qubits.

        Parameters:
            logical_qubits (list[int] | int): Logical qubit(s) where the H-gate will be applied.

        Returns:
            None
        '''

        self.logical_quantum_circuit.h(logical_qubits)

        if(self.encode_applied):
            self.physical_quantum_circuit.h(self.get_physical_qubits_from_logical(logical_qubits))

        else:
            self.gates_before_encode = True
            if isinstance(logical_qubits, int):
                logical_qubits = [logical_qubits]
            self.physical_quantum_circuit.h([7*i for i in logical_qubits])


    def id(self, logical_qubits: list | int):
        '''
        Applies the Identity gate I to specified logical qubits in both the logical and physical circuits.

        If encoding has been applied (`encode_applied` is True), the I-gate is applied to the
        corresponding physical qubits. If not, the I-gate is applied directly to the logical qubits
        and their mapped physical qubits.

        Parameters:
            logical_qubits (list[int] | int): Logical qubit(s) where the I-gate will be applied.

        Returns:
            None
        '''

        self.logical_quantum_circuit.id(logical_qubits)

        if(self.encode_applied):
            self.physical_quantum_circuit.id(self.get_physical_qubits_from_logical(logical_qubits))

        else:
            self.gates_before_encode = True
            if isinstance(logical_qubits, int):
                logical_qubits = [logical_qubits]
            self.physical_quantum_circuit.id([7*i for i in logical_qubits])


    def cx(self, control: list | int, target: list | int):
        '''
        Applies a CNOT (CX) gate between control and target logical qubits in both the logical and physical circuits.

        If encoding has been applied (`encode_applied` is True), the CNOT gate is applied between
        the corresponding physical qubits. If encoding is not applied, the gate is applied directly
        to the logical qubits and their mapped physical qubits.

        Parameters:
            control (list[int] | int): Logical qubit(s) acting as the control qubit(s).
            target (list[int] | int): Logical qubit(s) acting as the target qubit(s).

        Returns:
            None
        '''

        self.logical_quantum_circuit.cx(control, target)

        if(self.encode_applied):
            self.physical_quantum_circuit.cx(self.get_physical_qubits_from_logical(control), self.get_physical_qubits_from_logical(target))

        else:
            self.gates_before_encode = True
            if isinstance(control, int):
                control = [control]

            if isinstance(target, int):
                target = [target]

            self.physical_quantum_circuit.cx([7*i for i in control], [7*i for i in target])


    def measure_all(self, basis = 'z'):
        '''
        Measures all qubits in both the logical and physical circuits.

        The logical qubits are measured directly, while the physical qubits are
        measured using the `physical_measurement_qubit_register` through intermediary
        CNOT operations.

        Parameters:
            basis (String): Indicates the basis of the measurement. If is 'z', 'x' or 'all'

        Returns:
            None
        '''

        if basis == 'z':

            self.physical_quantum_circuit.add_register(self.physical_measurement_qubit_register)
            self.physical_quantum_circuit.add_register(self.physical_bit_register)

            self.logical_quantum_circuit.measure(self.logical_qubit_register, self.logical_bit_register)

            for i in range(self.logical_qubit_count):
                for j in range(7*i, 7*i+7):
                    self.physical_quantum_circuit.cx(self.physical_qubit_register[j], self.physical_measurement_qubit_register[i])

            self.physical_quantum_circuit.measure(self.physical_measurement_qubit_register, self.physical_bit_register)


        elif basis == 'x':

            self.physical_quantum_circuit.add_register(self.physical_measurement_qubit_register)
            self.physical_quantum_circuit.add_register(self.physical_bit_register)

            self.logical_quantum_circuit.h(self.logical_qubit_register)
            self.logical_quantum_circuit.measure(self.logical_qubit_register, self.logical_bit_register)

            self.physical_quantum_circuit.h(self.physical_measurement_qubit_register)
            for i in range(self.logical_qubit_count):
                for j in range(7*i, 7*i+7):
                    self.physical_quantum_circuit.cx(self.physical_measurement_qubit_register[i], self.physical_qubit_register[j])

            self.physical_quantum_circuit.h(self.physical_measurement_qubit_register)

            self.physical_quantum_circuit.measure(self.physical_measurement_qubit_register, self.physical_bit_register)


        elif basis == 'all':

            # Redefine physical_bit_register
            self.physical_bit_register = ClassicalRegister(7*self.logical_qubit_count, name = 'physical_bits')
            self.physical_quantum_circuit.add_register(self.physical_bit_register)

            self.logical_quantum_circuit.measure(self.logical_qubit_register, self.logical_bit_register)
            self.physical_quantum_circuit.measure(self.physical_qubit_register, self.physical_bit_register)

    def measure_operator(self, operator):
        '''
        Measures qubits in a Pauli operator indicating by 'operator' in both circuits

        The logical qubits are measured directly, while the physical qubits are
        measured using the `physical_measurement_qubit_register` through intermediary operations.

        Parameters:
            operator (list[String]): A list of lenght 7 indicating the Pauli operator.
                                      For example, ['X', 'Z', 'I'] represent the operator XZI
                                      The list might be in upper or lower cases

        Returns:
            None
        '''

        if len(operator) != 7:
            warning_text = "\r\nWARNING! operator must be a list of lenght 7."
            warnings.warn(warning_text, category=UserWarning)
            return

        circuit = self.physical_quantum_circuit

        for i in range(self.logical_qubit_count):

            qregs = self.physical_qubit_register[7*i :7*i + 7]
            creg = ClassicalRegister(1)
            qreg = QuantumRegister(1)

            circuit.add_register(qreg)
            circuit.add_register(creg)

            circuit.h(qreg) # 'anc' = |+>

            for i in range(7):
                match operator[i]:
                    case ('X' | 'x'):
                        circuit.cx(qreg, qregs[i])
                    case ('Z' | 'z'):
                        circuit.cz(qreg, qregs[i])
                    case ('Y' | 'y'):
                        circuit.cy(qreg, qregs[i])
                    case ('I' | 'i'):
                        pass

            circuit.h(qreg) # volver a base Z

            circuit.measure(qreg, creg) # medir extra_ancilla

    def delay(self, dt, unit='dt', logical_qubits = None):
        '''
        Applies a delay if time dt in both the logical and physical circuits.

        If encoding has been applied (`encode_applied` is True), the delay gate is applied between
        the corresponding physical qubits. If encoding is not applied, the gate is applied directly
        to the logical qubits and their mapped physical qubits.

        Parameters:
            dt (float): Time delay
            unit (String): Indicate the unit in which the time delay dt is expressed, might be 'dt', 'ms', 'us', etc
            logical_qubits (list[int] | int): Indicates the qubits in where the delay is applied. If None then apply delay to all qubits

        Returns:
            None
        '''

        if logical_qubits:
            self.logical_quantum_circuit.delay(dt, logical_qubits, unit = unit)

            if(self.encode_applied):
                self.physical_quantum_circuit.delay(dt, self.get_physical_qubits_from_logical(logical_qubits), unit = unit)

            else:
                self.gates_before_encode = True
                if isinstance(logical_qubits, int):
                    logical_qubits = [logical_qubits]
                self.physical_quantum_circuit.delay(dt, [7*i for i in logical_qubits], unit = unit)

        else:
            self.logical_quantum_circuit.delay(dt, unit = unit)

            if (self.encode_applied):
                self.physical_quantum_circuit.delay(dt, unit = unit)

            else:
                self.gates_before_encode = True
                self.physical_quantum_circuit.delay(dt, [7*i for i in range(self.logical_qubit_count)], unit = unit)





# DEFINITION OF FUNCTIONS USED IN THE CLASS:
def CSS_encoder(initial_state = True):
    '''
    Returns a CSS encoder as a quantum gate.

    This function constructs a CSS (Calderbank-Shor-Steane) encoder circuit that
    operates on a register of 7 qubits.

    Parameters:
        initial_state (String): Indicate the initial state, can be '0', '1' or None. If None we use the universal encoder
                            Default is None

    Returns:
        QuantumCircuit.to_gate: A quantum gate representing the Steane encoder,
                                ready to be appended to any set of 7 qubits.
    '''

    q_reg = QuantumRegister(7, "q")

    qc = QuantumCircuit(q_reg)
    qc.h(q_reg[4:7])

    # Si q0 = |1> tenemos que cambiar q1 y q2
    if initial_state:
        if initial_state == '0':
            pass

        elif initial_state == '1':
                qc.x(q_reg[0])
                qc.x(q_reg[1])
                qc.x(q_reg[2])

    else:
        qc.cx(q_reg[0], q_reg[1])
        qc.cx(q_reg[0], q_reg[2])

    # q3 is a parity check qubit for q4, q5 and q6
    qc.cx(q_reg[4], q_reg[3])
    qc.cx(q_reg[5], q_reg[3])
    qc.cx(q_reg[6], q_reg[3])

    # q2 is a parity check qubit for q4 and q5
    qc.cx(q_reg[4], q_reg[2])
    qc.cx(q_reg[5], q_reg[2])

    # q1 is a parity check qubit for q4 and q6
    qc.cx(q_reg[4], q_reg[1])
    qc.cx(q_reg[6], q_reg[1])

    # q0 is a parity check qubit for q5 and q6
    qc.cx(q_reg[5], q_reg[0])
    qc.cx(q_reg[6], q_reg[0])

    encoder = qc.to_gate(label="Steane \n Encoder")

    return encoder


# Append CSS_encoder
def append_CSS_encoder(qc, qubit_list, initial_state=None):
    '''
    Append the CSS encoder to a circuit qc and a list of qubits qubit_list.

    This function constructs a CSS (Calderbank-Shor-Steane) encoder circuit that
    operates on a register of 7 qubits.

    Parameters:
        qc (QuantumCircuit): qiskit quantum circuit in where to append the encoder
        qubit_list (list[int] | QuantumRegister): List of length 7*k in where to apply the encoder.
        initial_state (String): Indicate the initial state, can be '0', '1' or None. If None we use the universal encoder
                            Default is None

    Returns:
        True if we can append the CSS. False if there is some error
    '''

    qubit_count = len(qubit_list)

    if (qubit_count % 7) != 0:
        return False

    for i in range(qubit_count // 7):


        q_reg = qubit_list[i*7:i*7 + 7]


        qc.h(q_reg[4:7])

        if initial_state:
            if initial_state == '0':
                pass

            elif initial_state == '1':
                qc.x(q_reg[0])
                qc.x(q_reg[1])
                qc.x(q_reg[2])

        else:
            # Si q0 = |1> tenemos que cambiar q1 y q2
            qc.cx(q_reg[0], q_reg[1])
            qc.cx(q_reg[0], q_reg[2])

        # q3 is a parity check qubit for q4, q5 and q6
        qc.cx(q_reg[4], q_reg[3])
        qc.cx(q_reg[5], q_reg[3])
        qc.cx(q_reg[6], q_reg[3])

        # q2 is a parity check qubit for q4 and q5
        qc.cx(q_reg[4], q_reg[2])
        qc.cx(q_reg[5], q_reg[2])

        # q1 is a parity check qubit for q4 and q6
        qc.cx(q_reg[4], q_reg[1])
        qc.cx(q_reg[6], q_reg[1])

        # q0 is a parity check qubit for q5 and q6
        qc.cx(q_reg[5], q_reg[0])
        qc.cx(q_reg[6], q_reg[0])

    return True


# Error correction procedure for Steane code.
def CSS_correction():
    '''
    Returns a CSS error correction gate.

    This function constructs an error correction procedure for the Steane code,
    designed to correct both bit-flip and phase-flip errors. The procedure operates on
    13 qubits: 7 data qubits and 6 ancilla qubits. It follows the error correction scheme
    described in Nielsen's "Quantum Computation and Quantum Information."

    Parameters:
        None

    Returns:
        QuantumCircuit.to_gate: A quantum gate representing the CSS error correction operation.
    '''

    # FIRST PART: Bit flip error correction.
    q_reg = QuantumRegister(7, name="q")
    H1_reg = QuantumRegister(3, name="a")
    H2_reg = QuantumRegister(3, name="b")


    qc = QuantumCircuit(q_reg, H1_reg, H2_reg)

    # Firt element of He1:
    qc.cx(q_reg[0], H1_reg[0])
    qc.cx(q_reg[2], H1_reg[0])
    qc.cx(q_reg[4], H1_reg[0])
    qc.cx(q_reg[6], H1_reg[0])


    # Second element of He1:
    qc.cx(q_reg[1], H1_reg[1])
    qc.cx(q_reg[2], H1_reg[1])
    qc.cx(q_reg[5], H1_reg[1])
    qc.cx(q_reg[6], H1_reg[1])


    # Third element of He1:
    qc.cx(q_reg[3], H1_reg[2])
    qc.cx(q_reg[4], H1_reg[2])
    qc.cx(q_reg[5], H1_reg[2])
    qc.cx(q_reg[6], H1_reg[2])


    # Bit-flip error correction
    qc.append(XGate().control(3, ctrl_state="001"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[0]])
    qc.append(XGate().control(3, ctrl_state="010"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[1]])
    qc.append(XGate().control(3, ctrl_state="011"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[2]])
    qc.append(XGate().control(3, ctrl_state="100"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[3]])
    qc.append(XGate().control(3, ctrl_state="101"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[4]])
    qc.append(XGate().control(3, ctrl_state="110"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[5]])
    qc.append(XGate().control(3, ctrl_state="111"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[6]])


    # SECOND PART: Phase flip error correction:
    qc.h(q_reg)

    # First element of He2:
    qc.cx(q_reg[0], H2_reg[0])
    qc.cx(q_reg[2], H2_reg[0])
    qc.cx(q_reg[4], H2_reg[0])
    qc.cx(q_reg[6], H2_reg[0])


    # Second element of He2:
    qc.cx(q_reg[1], H2_reg[1])
    qc.cx(q_reg[2], H2_reg[1])
    qc.cx(q_reg[5], H2_reg[1])
    qc.cx(q_reg[6], H2_reg[1])


    # Third element of He2:
    qc.cx(q_reg[3], H2_reg[2])
    qc.cx(q_reg[4], H2_reg[2])
    qc.cx(q_reg[5], H2_reg[2])
    qc.cx(q_reg[6], H2_reg[2])


    # Phase-flip error correction:
    qc.append(XGate().control(3, ctrl_state="001"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[0]])
    qc.append(XGate().control(3, ctrl_state="010"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[1]])
    qc.append(XGate().control(3, ctrl_state="011"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[2]])
    qc.append(XGate().control(3, ctrl_state="100"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[3]])
    qc.append(XGate().control(3, ctrl_state="101"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[4]])
    qc.append(XGate().control(3, ctrl_state="110"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[5]])
    qc.append(XGate().control(3, ctrl_state="111"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[6]])

    qc.h(q_reg)

    # Transform circuit to quantum gate and return.
    error_correction = qc.to_gate(label = "CSS \n Error Correction")

    return error_correction



# Error correction procedure for Steane code.
def append_CSS_correction(qc, qubit_list, ancillas):
    '''
    Append the steane code error correction procedure to a circuit qc.

    This function constructs an error correction procedure for the Steane code,
    designed to correct both bit-flip and phase-flip errors. The procedure operates on
    13 qubits: 7 data qubits and 6 ancilla qubits. It follows the error correction scheme
    described in Nielsen's "Quantum Computation and Quantum Information."

    Parameters:
        qc (QuantumCircuit): Quantum circuit in where to append the correction procedure
        qubit_list (list[int] | QuantumRegister): List of encoded qubits. It must be of len 7*k for k > 0.
        ancillas (list[int] | QuantumRegister): List of ancillas for error correction. It must be of len 6*k for the same k as qubit_list

    Returns:
        True if the error correction procedure is appended correctly. False otherwise.
    '''

    qubit_count = len(qubit_list)
    ancillas_count = len(ancillas)

    if (qubit_count % 7 != 0 or ancillas_count % 6 != 0):
        return False

    if (qubit_count // 7 != ancillas_count // 6):
        return False

    k = qubit_count // 7
    for i in range(k):

        q_reg = qubit_list[i*7:i*7 + 7]
        H1_reg = ancillas[i*6:i*6 + 3]
        H2_reg = ancillas[i*6 + 3: i*6 + 6]

        # Firt element of He1:
        qc.cx(q_reg[0], H1_reg[0])
        qc.cx(q_reg[2], H1_reg[0])
        qc.cx(q_reg[4], H1_reg[0])
        qc.cx(q_reg[6], H1_reg[0])


        # Second element of He1:
        qc.cx(q_reg[1], H1_reg[1])
        qc.cx(q_reg[2], H1_reg[1])
        qc.cx(q_reg[5], H1_reg[1])
        qc.cx(q_reg[6], H1_reg[1])


        # Third element of He1:
        qc.cx(q_reg[3], H1_reg[2])
        qc.cx(q_reg[4], H1_reg[2])
        qc.cx(q_reg[5], H1_reg[2])
        qc.cx(q_reg[6], H1_reg[2])


        # Bit-flip error correction
        qc.append(XGate().control(3, ctrl_state="001"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[0]])
        qc.append(XGate().control(3, ctrl_state="010"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[1]])
        qc.append(XGate().control(3, ctrl_state="011"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[2]])
        qc.append(XGate().control(3, ctrl_state="100"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[3]])
        qc.append(XGate().control(3, ctrl_state="101"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[4]])
        qc.append(XGate().control(3, ctrl_state="110"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[5]])
        qc.append(XGate().control(3, ctrl_state="111"), [H1_reg[0], H1_reg[1], H1_reg[2], q_reg[6]])


        # SECOND PART: Phase flip error correction:
        qc.h(q_reg)

        # First element of He2:
        qc.cx(q_reg[0], H2_reg[0])
        qc.cx(q_reg[2], H2_reg[0])
        qc.cx(q_reg[4], H2_reg[0])
        qc.cx(q_reg[6], H2_reg[0])


        # Second element of He2:
        qc.cx(q_reg[1], H2_reg[1])
        qc.cx(q_reg[2], H2_reg[1])
        qc.cx(q_reg[5], H2_reg[1])
        qc.cx(q_reg[6], H2_reg[1])


        # Third element of He2:
        qc.cx(q_reg[3], H2_reg[2])
        qc.cx(q_reg[4], H2_reg[2])
        qc.cx(q_reg[5], H2_reg[2])
        qc.cx(q_reg[6], H2_reg[2])


        # Phase-flip error correction:
        qc.append(XGate().control(3, ctrl_state="001"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[0]])
        qc.append(XGate().control(3, ctrl_state="010"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[1]])
        qc.append(XGate().control(3, ctrl_state="011"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[2]])
        qc.append(XGate().control(3, ctrl_state="100"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[3]])
        qc.append(XGate().control(3, ctrl_state="101"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[4]])
        qc.append(XGate().control(3, ctrl_state="110"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[5]])
        qc.append(XGate().control(3, ctrl_state="111"), [H2_reg[0], H2_reg[1], H2_reg[2], q_reg[6]])

        qc.h(q_reg)

    return True


def append_multi_steane_syndrome_extraction(qc, qubit_list, ancillas):
    """
    Appends Steane code syndrome extraction (no correction) for multiple code blocks.

    Args:
        qc (QuantumCircuit): Circuit to append to.
        qubit_list (list[Qubit]): 7*k data qubits for k code blocks.
        ancillas (list[Qubit]): 6*k ancillas, 6 per block.

    Returns:
        True if successful, False otherwise.
    """
    qubit_count = len(qubit_list)
    ancilla_count = len(ancillas)
    if qubit_count % 7 != 0 or ancilla_count % 6 != 0:
        return False
    if qubit_count // 7 != ancilla_count // 6:
        return False

    k = qubit_count // 7  # number of blocks
    for i in range(k):
        data = qubit_list[i*7 : (i+1)*7]
        anc = ancillas[i*6 : (i+1)*6]
        # Z-type syndromes (first 3 ancillas)
        qc.cx(data[0], anc[0])
        qc.cx(data[2], anc[0])
        qc.cx(data[4], anc[0])
        qc.cx(data[6], anc[0])

        qc.cx(data[1], anc[1])
        qc.cx(data[2], anc[1])
        qc.cx(data[5], anc[1])
        qc.cx(data[6], anc[1])

        qc.cx(data[3], anc[2])
        qc.cx(data[4], anc[2])
        qc.cx(data[5], anc[2])
        qc.cx(data[6], anc[2])

        # X-type syndromes (last 3 ancillas)
        qc.h(data)
        qc.cx(data[0], anc[3])
        qc.cx(data[2], anc[3])
        qc.cx(data[4], anc[3])
        qc.cx(data[6], anc[3])

        qc.cx(data[1], anc[4])
        qc.cx(data[2], anc[4])
        qc.cx(data[5], anc[4])
        qc.cx(data[6], anc[4])

        qc.cx(data[3], anc[5])
        qc.cx(data[4], anc[5])
        qc.cx(data[5], anc[5])
        qc.cx(data[6], anc[5])
        qc.h(data)

    return True
