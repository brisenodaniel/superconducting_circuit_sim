"""
Transmon and Fluxonium definitions

Document contains function definitions for constructing operators on
fluxonia qudits and transmon coupler as described in Setiawan et. al. 2022
in the literature folder.

All operators are defined in the Quantum Harmonic Oscillator basis.
"""
from __future__ import annotations
from typing import TypeAlias
from dataclasses import dataclass 
from _collections_abc import Sequence
import warnings
import qutip as qt
import numpy as np

Qobj: TypeAlias = qt.Qobj
j: complex = complex(0,1)

@dataclass
class Subsystem:
    """Container class for operators defining a quantum system. Class can be\
    accessed as if it were a dictionary of operators with string labels. as:

            `Subsystem[str_label] = oper`, to add `oper` to `Subsystem`, or
            `oper = Subsystem[str_label]`, to retrieve `oper` from `Subystem`

    If the hamiltonian operator with label 'H' is not defined in the class, a \
    warning will be thrown when trying to access its operators.

    A Subsystem can be initialized by providing a dictionary of qt.Qobj operators\
    as qsys = Subystem({lbl:Qobj})

    Public attributes:
        base_space (str): A label indicating in which basis the operators in the class \
         are currently defined.
        tol (float): A decimal corresponding to numerical error tolerance used in comparison\
         operations with other Subsystems. See documentation for qt.isequal().
    """
    # _ops:dict[str:Qobj] # Dictionary of quantum operators. Should not be directly accessed. To access, use Subsystem[op_label] instead.
    # _nlev:int|None = None # This field should never be set manually
    # _base_space:str = "QHO" # basis in which operators are assumed to be originally defined
    # _tranf_list:np.ndarray[Qobj]|None = None # List of change-of-basis operators (or bases transformed into) applied to operators in system, in the order they were applied
    # _tol:int = 1e-9 # numerical error tolerance to be used in comparison operations
    # _state:Qobj|None = None # State vector of the quantum system.


    def __init__(self, ops:dict[str:Qobj], base_space='QHO', tol=1e-9)->Subsystem:
        self._ops:dict = {} # Dictionary of quantum operators. Should not be directly accessed. To access, use Subsystem[op_label] instead.
        self._nlev:int|None = None # Number of energy levels considered in system model. This field should never be set manually
        self._base_space:str = base_space # basis in which operators are assumed to be originally defined
        self._transf_list:list[Qobj|np.ndarray(Qobj)] = [] # List of change-of-basis operators (or bases transformed into) applied to operators in system, in the order they were applied
        self._tol:float = tol # numerical error tolerance to be used in comparison operations
        self._state: Qobj|None = None
        for key, value in ops.items():
            self.__setitem__(key,value)


    #### getters and setters
    #for _state
    @property
    def state(self)->Qobj:
        return self._state
    
    @state.setter
    def state(self, vect:int|Qobj)->None:
        assert isinstance(vect,'Qobj'), \
        f'State vector must be a Qobj ket, type {type(vect)} given'
        assert vect.isket, 'State vector must be a ket'
        vect_dims = vect.dims[0][0]
        assert vect_dims == self._nlev, \
        f'State vector must have {self._nlev} coordinates, but has {vect_dims}'
        self._state = vect

    # for _base_space
    @property
    def base_space(self)->str:
        return self._base_space
    
    @base_space.setter
    def base_space(self, basis_lbl:str)->None:
        self._base_space = basis_lbl

    # for _tol
    @property
    def tol(self)->float:
        return self._tol
    
    @tol.setter
    def tol(self, tolerance)->None:
        assert isinstance(tolerance, float),\
            "Cannot add non-float type as tol"
        self._tol = tolerance


    def __eq__(self, other:Subsystem)->bool:
        """Comparison operator for subsystems.
        WARNING: If _tol operator is not identical for both subsystems,\
            equality comparisons will may not be symmetric. That is , sys1==sys2 may\
            give different result than sys2==sys1. If _tol is not equal across both\
            subsystems, function will throw a warning.

        Args:
            other (Subsystem): other subsystem to compare

        Returns:
            bool: True if all fields of both subsytems are equal
        """
        if self._tol!=other._tol:
            warnings.warn(
                "Numerical error tolerance parameter _tol has been set to"\
                + " different values for the two subsytems. Equality comparison"\
                + f" may not be symmetric. For left system _tol={self._tol}."\
                + f" For second system _tol={other._tol}"
            )
        if self._nlev!=other._nlev: return False
        if self._nlev is not None: # self._nlev==None indicates empty subsysem
            # from here on out we can safely assume both subsystems are populated,
            # and their operators and basis are of equal dimension
            for key in self._ops: #compare operators
                if key not in other._ops: return False
                if not qt.isequal(self._ops[key], other._ops[key], self._tol): return False
            if self._base_space != other._base_space: return False
            #check basis, last thing to check
            if self._basis is None: #if no basis defined, check that other also has no basis defined
                if other._basis is not None: return False
            else:
                if other._basis is None: return False
                for idx, vector in enumerate(self._basis):
                    other_vector = other._basis[idx]
                    if not qt.isequal(vector, other_vector, self._tol): return False
        return True
    
    def __getitem__(self, key:str)->Qobj:
        assert key in self._ops, f"Operator {key} not defined for subsystem."
        if 'H' not in self._ops:
            warnings.warn(
                'Accessing operator from subsystem with no hamiltonian. Subsystem is ill-defined.'
                )
        return self._ops[key]
    
    def __setitem__(self, key:str, value:Qobj)->None:
        assert value.isoper, \
            f"Only Qobj operators may be added via hashing. No bras or kets."
        op_dims = value.dims[0][0]
        if self._nlev is None:
            self._nlev = op_dims
        assert self._nlev==op_dims,\
            f"Cannot add operator of dimension {op_dims}"\
            + f" to system of dimension {self.nlev}."
        self._ops[key] = value

    def __contains__(self, key:str)->bool:
        return key in self._ops
    
    def __iter__(self):
        return self._ops.__iter__()

    def items(self)-> dict[str:Qobj]:
        return self._ops.items()
    
    #### Non-dict Methods ###
    def transform(self, transf:Sequence) -> Subsystem:
        """Transform into basis indicated by `transf`

        Args:
            transf (Sequence): Array-like object specifying either a transfomration matrix or a list\
                of kets which make up the new basis for the operators in ops.
            

        Returns:
            Subsystem: Subsystem identical to `self`, but in the basis indicated by `transf`.
        """
        new_ops:dict[str:Qobj] = {}
        new_transf_list = self._transf_list + [transf]
        new_state:Qobj = None
        if self._state is not None:
            new_state = self._state.transform(transf)
        for key, op in self._ops.items():
            new_ops[key] = op.transform(transf)

        new_sys = Subsystem(new_ops)
        new_sys._state = new_state
        new_sys._transf_list = new_transf_list
        new_sys._base_space = self._base_space
        new_sys._tol = self._tol
        return new_sys
    
    def truncate(self, truncate_to:int)->Subsystem:
        """Generates a Subsystem identical to `self`, but truncated to `nlev`\
        energy levels.

        Args:
            truncate_to (int): Number of energy levels to truncate to
        Returns:
            Subsystem: Subsystem identical to `self`, but truncated to `nlev`\
            energy levels.
        """
        assert truncate_to<=self._nlev,\
        f'Desired truncation length {truncate_to} specifies a system with higher'\
        f' number of energy levels than self: {self._nlev}'
        if truncate_to == self._nlev:
            return self
        new_ops:dict[str:Qobj] = {}
        new_transf_list = []
        new_state = None
        keep_states = list(range(truncate_to))
        if self._state is not None:
            new_state = new_state.extract_states(keep_states)
        for key, op in self._ops.items():
            new_ops[key] = op.extract_states(keep_states)
        for transf in self._transf_list:
            if isinstance(transf, Qobj):
                new_transf_list += [transf.extract_states(keep_states)]
            else:
                trunc_basis = np.empty(len(transf), dtype=object)
                for i, basis_vect in enumerate(transf):
                    trunc_basis[i] = basis_vect.extract_states(keep_states)
                new_transf_list += [trunc_basis]
        new_sys = Subsystem(new_ops)
        new_sys._transf_list = new_transf_list
        new_sys._state = new_state
        new_sys._base_space = self._base_space
        new_sys._tol = self._tol 
        return new_sys

        

    

def build_fluxonium_operators(nlev:int,
                              E_C:float,
                              E_J:float,
                              E_L:float,
                              phi_ext:float) -> Subsystem:
    """Method defines operators for a fluxonium circuit as defined in
    Setiawan et. al. 2022.

    Args:
        nlev (int): Number of energy levels to consider in the system
        E_C (float): Circuit parameter, capacitance energy coefficient
        E_J (float): Circuit parameter, josephenson energy coefficient
        E_L (float): Circuit parameter, linear inductive energy coefficient
        phi_ext (float): External flux threading  loop formed by josephenson
         junction and linear inductor.

    Returns:
        dict[str,Qobj]: _Dictionary of operators acting on fluxonium state vector.
           Includes Hamiltonian operator
    """
    a:Qobj = qt.destroy(nlev)
    n_zpf:float = (E_L/(32*E_C))**(1/4)
    phi_zpf:float = (2*E_C/E_L)**(1/4)
    n:Qobj = n_zpf*(a + a.dag())
    phi:Qobj = phi_zpf*j*(a-a.dag())
    H:Qobj = build_fluxonium_hamiltonian(n,phi,E_C, E_J, E_L, phi_ext)
    ops = {'n':n, 'phi':phi, 'H':H}
    fluxonium = Subsystem(ops)
    return fluxonium

def build_fluxonium_hamiltonian(n:Qobj, 
                                phi:Qobj, 
                                E_C:float, 
                                E_J:float,
                                E_L:float,
                                phi_ext:float) -> Qobj:
    """Method defines the hamiltonian operator for a fluxonium circuit as described
    in Setiawan et. al. 2022.

    Args:
        n (Qobj): Charge operator for the fluxonium circuit.
        phi (Qobj): Phase operator for the fluxonium circuit.
        E_C (float): Circuit parameter. Capacitive Energy Coefficient.
        E_J (float): Circuit parameter. Josephenson Energy Coefficient.
        E_L (float): Circuit parameter. Inductive Energy Coefficient.
        phi_ext (float): External flux threading  loop formed by josephenson
         junction and linear inductor.

    Returns:
        Qobj: Hamiltonian operator for fluxonium circuit with charge operator `n` and 
        phase operator `phi`.
    """
    nlev:int = np.array(n.dims).ravel()[0]
    Id:Qobj = qt.qeye(nlev)
    capacitor_energy:Qobj = 4*E_C*n**2 # type: ignore
    josephenson_energy:Qobj = -E_J*phi.cosm()
    inductor_energy:Qobj = 0.5*E_L*(phi - 2*np.pi*phi_ext*Id)**2
    H = capacitor_energy + josephenson_energy + inductor_energy
    return H

def build_transmon_operators(nlev:int, w:float, U:float) -> Subsystem:
    """Method defines the hamiltonian operator for a transmon circuit as described
    in Setiawan et. al. 2022

    Args:
        nlev (int): Number of energy levels to consider for transmon circuit.
        w (float): Circuit parameter, frequency of 0 -> 1 transition.
        U (float): Circuit parameter, transmon anharmonicity

    Returns:
        dict[str,Qobj]: Dictionary of operators acting on transmon state vector. 
        Includes hamiltonian.
    """
    a:Qobj = qt.destroy(nlev)
    H:Qobj = w*a.dag()*a - U*a.dag()**2 * a**2
    ops = {'a':a, 'H':H}
    transmon = Subsystem(ops)
    return transmon