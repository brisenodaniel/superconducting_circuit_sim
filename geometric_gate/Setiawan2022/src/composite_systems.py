"""File defines a container class for composite systems, made up of 
multiple instances of Subsystem class from subsystems.py, tensored
into a joint Hilbert space
"""
from __future__ import annotations
import qutip as qt
import numpy as np
import warnings
from subsystems import Subsystem
from typing import TypeAlias, Callable, TypeVar


Qobj: TypeAlias = qt.Qobj
j: complex = complex(0,1)

class CompositeSystem:
   
    def __init__(self, 
                 subsystems:dict[str:Subsystem],
                 idxs:dict[str:int], 
                 H_int:Qobj|None=None,
                 tol:float = 1e-9)->None:
      self.subsystems:dict[str:Subsystem] = subsystems
      self._H_int:Qobj = H_int
      self.lbl_to_idx:dict[str:int] = idxs 
      self.idx_to_lbl:list = self.build_idx_to_lbl()
      self._tol = tol
      self.H:Qobj = self.build_hamiltonian()

    def build_hamiltonian(self)->None:
        H:int|Qobj = 0
        for sys_lbl in self.subsystems:
            H += self.get_raised_op(sys_lbl, 'H')
        if self._H_int is not None:
            H += self._H_int
        return H
    
    def build_idx_to_lbl(self)->np.ndarray[str]:
        nsys = max(self.lbl_to_idx.values()) + 1
        idx_to_lbl = np.empty(nsys,dtype=str)
        for lbl, idx in self.lbl_to_idx.items():
            idx_to_lbl[idx] = lbl
        return idx_to_lbl

    def add_interaction_term(self, H_int:Qobj)->None:
        """No return value, function adds interaction term `H_int` to self.H\
        as a side-effect

        Args:
            H_int (Qobj): Interaction term to be added to `H_int`.
        """
        assert H_int.dims == self.H.dims,\
        f'Operator of dims {self.H.dims} expected, recieved {H_int.dims}'
        self.H += H_int

    def get_bare_hamiltonian(self)->Qobj:
        """ Method to obtain bare hamiltonian from subsystems

        Returns:
            Qobj: Bare hamiltonian
        """
        nsys = max(self.lbl_to_idx.values()) + 1
        subsys_list = np.empty(nsys, dtype=object)
        for sys_lbl, idx in self.lbl_to_idx.items():
            subsys_list[idx] = self.subsystems[sys_lbl].H
        return  qt.tensor(subsys_list)

    def plus_interaction_term(self, H_int:Qobj)->CompositeSystem:
        """Returns a `CompositeSystem` equal to `self.add_interaction_term(H_int)`.\
        No side effect.

        Args:
            H_int (Qobj): Interaction term to be added to `self.H` in new `CompositeSystem`
        Returns:
            CompositeSystem: Equal to `self.add_interaction_term(H_int)`
        """
        assert H_int.dims == self.H.dims, \
        f'Operator of dims {self.H.dims} expected, recieved {H_int.dims}'
        new_sys = CompositeSystem(self.subsystems, self.lbl_to_idx, H_int)
        if self._H_int is not None:
            new_sys.add_interaction_term(self.H_int)
        return new_sys
        

    def get_raised_op(self, subsys:str, op:str|list[str], op_spec:Callable['...',Qobj]|None=None)->Qobj:
        """Method raises the desired subsystem operator `self.subsystems[subsys][op]`\
            to the product space acting on `self.H`

        Args:
            subsys (str): Label for the subsystem on which the raised operator should act
            op (str | list[str]): Label for the operator acting on the desired subsystem, or list of labels for \
                operators to combine using `op_spec`.
            op_spec (Callable[...,Qobj], optional): Function taking in the same number of operators as `op`, in the same order,\
             and returning a new Qobj. If `op` is not a list, this parameter will be ignored.

        Returns:
            Qobj: An operator of the form Id_1 x Id_2 x ... x op_j x ... Id_k, where Id_i is the identity \
                matrix acting on the i'th subsystem in self.lbl_to_idx, and op_j acts on the j'th subsystem in self.lbl_to_idx.
        """
        nsys = max(self.lbl_to_idx.values()) + 1 # number of subsystems
        oper_list:np.ndarray[Qobj] = np.empty(nsys,dtype=object)# list of subspace operators to be tensored together to form full hilbert space operator
        subsys_idx = self.lbl_to_idx[subsys] # index of each subsystem in tensor operation
        # form array of identity operators acting on the corresponding subsystem (correspondence by tensor operation order)
        for key, sys in self.subsystems.items():
            idx = self.lbl_to_idx[key]
            nlev = sys.nlev
            oper_list[idx] = qt.qeye(nlev)
        # replace only the operator at subsys_idx with operator we would like to promote
        if isinstance(op, list):
           arg_ops:list[Qobj] = [self.subsystems[subsys][lbl] for lbl in op]
           subsys_op = op_spec(*arg_ops)
        else:
            subsys_op = self.subsystems[subsys][op] # subsystem operator we would like to promote to full hilbert space
        oper_list[subsys_idx] = subsys_op 
        return qt.tensor(oper_list)
    
    def is_diag(self)->bool:
       diags = self.H.diag()
       H_diag = qt.qdiags(diags,0,self.H.dims)
       return qt.isequal(self.H, H_diag, self._tol)
    
    def __eq__(self, other:CompositeSystem)->bool:
       if self._tol != other._tol:
          warnings.warn(
             'The two CompositeSystem objects being compared do not have the same _tol attribute.\n'\
             +f' Leftmost has _tol={self._tol}, rightmost _tol={other._tol}\n'+\
             'Equality operation may be assymetric'
          )
       if not qt.isequal(self.H, other.H): return False
       if self.lbl_to_idx != other.lbl_to_idx: return False
       if self.subsystems != other.subsystems: return False
       return True
       

    


