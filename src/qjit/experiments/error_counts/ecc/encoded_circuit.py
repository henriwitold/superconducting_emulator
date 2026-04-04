from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import warnings
import matplotlib.pyplot as plt

class EncodedCircuit(ABC):
    """
    An abstract base class representing an encoded quantum circuit framework for quantum error correction.

    Attributes:
        logical_qubit_count (int): Number of logical qubits.
        physical_qubit_count (int): Number of physical qubits derived from the number of logical qubits and
            physical qubits per logical qubit.
        logical_qubit_register (QuantumRegister): Quantum register for logical qubits.
        logical_bit_register (ClassicalRegister): Classical register for logical qubit measurements.
        logical_quantum_circuit (QuantumCircuit): Quantum circuit operating on logical qubits.
        physical_qubit_register (QuantumRegister): Quantum register for physical qubits.
        physical_bit_register (ClassicalRegister): Classical register for physical qubit measurements.
        physical_quantum_circuit (QuantumCircuit): Quantum circuit operating on physical qubits.
        encode_applied (bool): Flag indicating whether encoding has been applied to the circuit.

    Methods:
        get_physical_qubits_from_logical(logical_qubits): Map logical qubits to corresponding physical qubits.
        encode(): Initialize the circuit with an encoded state.
        x(logical_qubits): Apply Pauli X-gate to logical qubits in both logical and physical circuits.
        y(logical_qubit): Apply Pauli Y-gate to logical qubits in both logical and physical circuits.
        z(logical_qubits): Apply Pauli Z-gate to logical qubits in both logical and physical circuits.
        h(logical_qubits): Apply Hadamard gate to logical qubits in both logical and physical circuits.
        cx(control, target): Apply CNOT gate between specified logical qubits in both logical and physical circuits.
        measure_all(): Measure all logical and physical qubits.
        correct(): Apply quantum error correction to the physical circuit.
        barrier(num_qubits, label): Apply a barrier to separate operations in both circuits.
        draw_logical_circuit(output): Display the logical quantum circuit.
        draw_physical_circuit(output): Display the physical quantum circuit.
        append_unitary_error(unitary_error, physical_qubits): Append an error gate to specified physical qubits.
        append_init(U, logical_qubits): Append a quantum gate to both logical and physical circuits if encoding has not been applied.
    """

    def __init__(self, logical_qubit_count, physical_qubits_per_logical):
        """
        Initialize the EncodedCircuit with logical and physical qubits, registers, and circuits.

        Parameters:
            logical_qubit_count (int): Number of logical qubits.
            physical_qubits_per_logical (int): Number of physical qubits associated with each logical qubit.
        """
        self.logical_qubit_count = logical_qubit_count
        self.physical_qubits_per_logical = physical_qubits_per_logical
        self.physical_qubit_count = logical_qubit_count * physical_qubits_per_logical  # Important supposition!

        # Logical qubit registers and quantum circuit
        self.logical_qubit_register = QuantumRegister(self.logical_qubit_count, name='logical_qubits')
        self.logical_bit_register = ClassicalRegister(self.logical_qubit_count, name='logical_bits')
        self.logical_quantum_circuit = QuantumCircuit(self.logical_qubit_register, self.logical_bit_register, 
                                                      name='logical_circuit')

        # Physical qubit registers and quantum circuit
        self.physical_qubit_register = QuantumRegister(self.physical_qubit_count, name='physical_qubits')

        # Comment: Typically, measurement is done in stabilizers, requiring one ancilla per logical qubit.
        # Another approach is to decode and measure, which also requires self.logical_qubit_count bits.
        # Measuring all physical qubits is possible but less common.
        self.physical_bit_register = ClassicalRegister(self.logical_qubit_count, name='physical_bits')
        self.physical_quantum_circuit = QuantumCircuit(self.physical_qubit_register, self.physical_bit_register, 
                                                       name='physical_circuit')
        
        self.encode_applied = False

    @abstractmethod
    def get_physical_qubits_from_logical(self, logical_qubits: list | int):
        """Map certain logical qubits to corresponding physical qubits."""
        pass

    @abstractmethod
    def encode(self):
        """Initialize the circuit with an encoded state."""
        pass

    @abstractmethod
    def x(self, logical_qubits: list | int):
        """Apply Pauli X-gate to a certain logical qubit in both logical and physical circuits."""
        pass

    @abstractmethod
    def y(self, logical_qubit: list | int):
        """Apply Pauli Y-gate to a certain logical qubit in both logical and physical circuits."""
        pass

    @abstractmethod
    def z(self, logical_qubits: list | int):
        """Apply Pauli Z-gate to a certain logical qubit in both logical and physical circuits."""
        pass

    @abstractmethod
    def h(self, logical_qubits: list | int):
        """Apply Hadamard gate to a certain logical qubit in both logical and physical circuits."""
        pass

    @abstractmethod
    def cx(self, control: list | int, target: list | int):
        """Apply CNOT gate to certain logical qubits in both logical and physical circuits."""
        pass

    @abstractmethod
    def measure_all(self):
        """Measure the state in both logical and physical circuits."""
        pass

    @abstractmethod
    def correct(self):
        """Apply quantum error correction in physical circuit."""
        pass

    def barrier(self, num_qubits=None, label=None):
        """
        Apply a barrier to separate operations in both circuits.

        Parameters:
            num_qubits (int | list[int], optional): Number of qubits to which the barrier is applied.
            label (str, optional): Label for the barrier.

        Returns:
            None
        """

        # if num_qubits == None:
        #     num_qubits = [i for i in range(self.logical_qubit_count)]

        # logical_num_qubits = num_qubits

        # if isinstance(logical_num_qubits, int):
        #     logical_num_qubits = [num_qubits]

        # physical_num_qubits = []
        # for i in logical_num_qubits:
        #     for j in range(7):
        #         physical_num_qubits.append(7*i + j)

        # self.logical_quantum_circuit.barrier(logical_num_qubits, label=label)
        # self.physical_quantum_circuit.barrier(physical_num_qubits, label=label)

        self.logical_quantum_circuit.barrier(label=label)
        self.physical_quantum_circuit.barrier(label=label)

    def draw_logical_circuit(self, output='mpl'):
        """
        Display the logical quantum circuit.

        Parameters:
            output (str): Format for the output (default is 'mpl').

        Returns:
            Matplotlib figure or other specified format of the logical circuit diagram.
        """
        return self.logical_quantum_circuit.draw(output=output)

    def draw_physical_circuit(self, output='mpl'):
        """
        Display the physical quantum circuit.

        Parameters:
            output (str): Format for the output (default is 'mpl').

        Returns:
            Matplotlib figure or other specified format of the physical circuit diagram.
        """
        return self.physical_quantum_circuit.draw(output=output)
    
    def draw_both(self, output='mpl'):
        """
        Display both the logical and physical quantum circuits.

        Parameters:
            output (str): Format for the output (default is 'mpl').

        Returns:
            Matplotlib figure or other specified format of the circuit diagrams.
        """
        return self.logical_quantum_circuit.draw(output=output), self.physical_quantum_circuit.draw(output=output)



    def append_unitary_error(self, unitary_error, physical_qubits: list | int):
        """
        Append an error gate to the physical circuit.

        Parameters:
            unitary_error: A quantum gate representing the error to be applied.
            physical_qubits (list[int] | int): The physical qubit(s) where the gate will be applied.

        Returns:
            None
        """
        if isinstance(physical_qubits, int):
            physical_qubits = [physical_qubits]

        self.physical_quantum_circuit.append(unitary_error, physical_qubits)

    def append_init(self, U, logical_qubits: list | int):
        """
        Append a quantum gate at the beginning to both the logical and physical circuits.

        Parameters:
            U: A quantum gate acting on the specified logical qubits.
            logical_qubits (list[int] | int): The logical qubit(s) where the gate will be applied.

        Returns:
            None

        Raises:
            UserWarning: If encoding has already been applied (encode_applied=True).
        """
        if isinstance(logical_qubits, int):
            logical_qubits = [logical_qubits]

        if self.encode_applied:
            warning_text = "\r\nWARNING! Cannot use append() method after using the encode() method.\r\n" \
                           "To add a unitary error on physical qubit use the method .append_unitary_error()"
            warnings.warn(warning_text, category=UserWarning)
        else:
            self.logical_quantum_circuit.append(U, logical_qubits)
            self.physical_quantum_circuit.append(U, [self.physical_qubit_count * i for i in logical_qubits])