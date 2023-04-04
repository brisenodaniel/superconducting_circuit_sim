"""File defines a container class for composite systems, made up of 
multiple instances of Subsystem class from subsystems.py, tensored
into a joint Hilbert space
"""
from __future__ import annotations
import qutip as qt
import numpy as np
import warnings
from subsystems import Subsystem
from typing import TypeAlias


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
      self.idxs:dict[str:int] = idxs 
      self.idxs_to_lbl:list = self.build_idxs_to_lbl()
      self._tol = tol
      self.H:Qobj = self.build_hamiltonian()

    def build_hamiltonian(self)->None:
      nsys = max(self.idxs.values()) + 1
      subsys_list = np.empty(nsys, dtype=object)
      for sys_lbl, idx in self.idxs.items():
        subsys_list[idx] = self.subsystems[sys_lbl].H
      H = qt.tensor(subsys_list) #bare hamiltonian
      if self._H_int is not None:
         H += self._H_int
      return H
    
    def build_idxs_to_lbl(self)->np.ndarray[str]:
        nsys = max(self.idxs.values()) + 1
        idxs_to_lbl = np.empty(nsys,dtype=str)
        for lbl, idx in self.idxs.items():
            idxs_to_lbl[idx] = lbl
        return idxs_to_lbl

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
        nsys = max(self.idxs.values()) + 1
        subsys_list = np.empty(nsys, dtype=object)
        for sys_lbl, idx in self.idxs.items():
            subsys_list[idx] = self.subsystems[sys_lbl]
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
        new_sys = CompositeSystem(self.subsystems, self.idxs, H_int)
        if self._H_int is not None:
            new_sys.add_interaction_term(self.H_int)
        return new_sys
        

    def get_raised_op(self, subsys:str, op:str)->Qobj:
        """Method raises the desired subsystem operator `self.subsystems[subsys][op]`\
            to the product space acting on `self.H`

        Args:
            subsys (str): Label for the subsystem on which the raised operator should act
            op (str): Label for the operator acting on the desired subsystem

        Returns:
            Qobj: An operator of the form Id_1 x Id_2 x ... x op_j x ... Id_k, where Id_i is the identity \
                matrix acting on the i'th subsystem in self.idxs, and op_j acts on the j'th subsystem in self.idxs.
        """
        nsys = max(self.idxs.values()) + 1 # number of subsystems
        oper_list:np.ndarray[Qobj] = np.empty(nsys,dtype=object)# list of subspace operators to be tensored together to form full hilbert space operator
        subsys_idx = self.idxs[subsys] # index of each subsystem in tensor operation
        subsys_op = self.subsystems[subsys][op] # subsystem operator we would like to promote to full hilbert space
        # form array of identity operators acting on the corresponding subsystem (correspondence by tensor operation order)
        for key, sys in self.subsystems.items():
            idx = self.idxs[key]
            nlev = sys.nlev
            oper_list[idx] = qt.qeye(nlev)
        # replace only the operator at subsys_idx with operator we would like to promote
        oper_list[subsys_idx] = subsys_op 
        return qt.tensor(oper_list)
    
    def __eq__(self, other:CompositeSystem)->bool:
       if self._tol != other._tol:
          warnings.warn(
             'The two CompositeSystem objects being compared do not have the same _tol attribute.\n'\
             +f' Leftmost has _tol={self._tol}, rightmost _tol={other._tol}\n'+\
             'Equality operation may be assymetric'
          )
       if not qt.isequal(self.H, other.H): return False
       if self.idxs != other.idxs: return False
       if self.subsystems != other.subsystems: return False
       return True
       

    


