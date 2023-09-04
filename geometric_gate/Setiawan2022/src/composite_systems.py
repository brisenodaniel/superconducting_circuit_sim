"""File defines a container class for composite systems, made up of
multiple instances of Subsystem class from subsystems.py, tensored
into a joint Hilbert space
"""
from __future__ import annotations
import qutip as qt
import numpy as np
import warnings
from functools import cached_property
from subsystems import Subsystem
from typing import TypeAlias, Callable, TypeVar


Qobj: TypeAlias = qt.Qobj
j: complex = complex(0, 1)


class CompositeSystem:
    class_idx: int = 0

    def __init__(self,
                 subsystems: dict[str:Subsystem],
                 idxs: dict[str:int],
                 H_int: Qobj | None = None,
                 tol: float = 1e-9,
                 nlev: int | None = None,
                 frozen_basis: bool = False,
                 immutable: bool = False) -> CompositeSystem:
        self.nlev = nlev
        # eigenbasis vectors written in the bare basis
        self.basis: np.ndarray[Qobj] | None = None
        self.frozen_basis = frozen_basis
        self.frozen_basis = True  # REMOVE
        self._immutable = immutable  # note, self._immutable and self._class_idx must
        if immutable:             # be first instance variables set in constructor
            self._class_idx = CompositeSystem.class_idx
            CompositeSystem.class_idx += 1
        self.subsystems: dict[str:Subsystem] = subsystems
        self.H_int: Qobj | None = H_int
        self.lbl_to_idx: dict[str:int] = idxs
        self.idx_to_lbl: list = self.build_idx_to_lbl()
        self._tol = tol
        self.eigenstates: np.ndarray[Qobj] | None = None
        # dummy hamiltionian so __hash__ doesn't throw a fit during initialization
        self.H: Qobj = qt.qeye(1)
        self.H: Qobj = self.build_hamiltonian(H_int=H_int, nlev=nlev)

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def immutable(self) -> bool:
        return self._immutable

    @property
    def object_id(self) -> int:
        return self.__hash__()

    def freeze(self) -> None:
        self._immutable = True
        CompositeSystem.class_idx += 1
        self._class_idx = CompositeSystem.class_idx

    def __diagonalize_hamiltonian(self) -> None:
        assert self.H_int is not None,\
            'Off-diagonal operator H_int must be \
            added to hamiltonian before diagonalizing\
            total hamiltonian'
        self.basis = self.H.eigenstates()[1]
        self.H = self.H.transform(self.basis)

    def build_hamiltonian(self, H_int=None, nlev=None,
                          no_side_effects=False) -> None:
        if H_int is None:
            H_int = self.H_int
        if nlev is None:
            nlev = self.nlev
        # build hamiltonian of full dims
        H: int | Qobj = 0
        for sys_lbl in self.subsystems:
            H += self.get_raised_op(sys_lbl,
                                    'H',
                                    use_dressed_eigenbasis=False,
                                    match_dressed_H_dims=False)
        if H_int is not None:
            assert H_int.dims == H.dims,\
                f'Interaction hamiltonian must have dimensions equal\
                to the bare product hamiltonian of all subsystems. \
                Expected {H.dims}, got {H_int.dims}'
            H += H_int
        # set basis
        basis = H.eigenstates()[1]
        H = self.align_basis(H, basis, nlev)
        # set zero of energy
        H = H - H[0, 0]
        if not no_side_effects:
            self.basis = basis
            self.nlev = nlev
            self.H_int = H_int
            self.H = H
            self.eigenstates = self.H.eigenstates()[1]
        return H

    def align_basis(self, op: Qobj, basis=None, nlev=None,
                    tidy: bool = True):
        if self.frozen_basis:
            return op
        # set basis and nlev variable default values
        if basis is None:
            basis = self.basis
        if nlev is None:
            nlev = self.nlev
        # Transform operator
        if basis is not None:
            op = op.transform(basis)
        if nlev is not None:
            op = op.extract_states(range(nlev))
        if tidy:
            op.tidyup(self._tol)
        return op

    def build_idx_to_lbl(self) -> np.ndarray[str]:
        nsys = max(self.lbl_to_idx.values()) + 1
        idx_to_lbl = np.empty(nsys, dtype=str)
        for lbl, idx in self.lbl_to_idx.items():
            idx_to_lbl[idx] = lbl
        return idx_to_lbl

    def add_interaction_term(self, H_int: Qobj) -> None:
        """No return value, function adds interaction term `H_int` to self.H\
        as a side-effect

        Args:
            H_int (Qobj): Interaction term to be added to `H_int`.
        """
        self.H_int = H_int
        self.H = self.build_hamiltonian()

    def get_bare_hamiltonian(self,
                             use_dressed_eigenbasis=True,
                             match_dressed_H_dims=True) -> Qobj:
        """ Method to obtain bare hamiltonian from subsystems

        Returns:
            Qobj: Bare hamiltonian
        """
        nsys = max(self.lbl_to_idx.values()) + 1
        subsys_list = np.empty(nsys, dtype=object)
        for sys_lbl, idx in self.lbl_to_idx.items():
            subsys_list[idx] = self.subsystems[sys_lbl].H
        H_bare = qt.tensor(subsys_list)
        if use_dressed_eigenbasis:
            H_bare = self.align_basis(H_bare)
        elif match_dressed_H_dims and self.nlev is not None:
            H_bare = H_bare.extract_states(range(self.nlev))
        return H_bare

    def plus_interaction_term(self, H_int: Qobj) -> CompositeSystem:
        """Returns a `CompositeSystem` equal to `self.add_interaction_term(H_int)`.\
        No side effect.

        Args:
            H_int (Qobj): Interaction term to be added to `self.H` in new `CompositeSystem`
        Returns:
            CompositeSystem: Equal to `self.add_interaction_term(H_int)`
        """
        assert H_int.dims == self.H.dims, \
            f'Operator of dims {self.H.dims} expected, recieved {H_int.dims}'
        return self.build_hamiltonian(H_int=H_int, no_side_effects=True)

    def get_raised_op(self,
                      subsys: str, op: str | list[str],
                      op_spec: Callable['...', Qobj] | None = None,
                      use_dressed_eigenbasis=True,
                      match_dressed_H_dims=True) -> Qobj:
        """Method raises the desired subsystem operator `self.subsystems[subsys][op]`\
            to the product space acting on `self.H`. This is a wrapper for self.__get_raised_ops which \
            casts all parameters to hashable types.

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
        if isinstance(op, list):
            op = tuple(op)
        raised_op = self.__get_raised_op(subsys, op, op_spec)
        if use_dressed_eigenbasis:
            raised_op = self.align_basis(raised_op)
        elif match_dressed_H_dims and self.nlev is not None:
            raised_op = raised_op.extract_states(range(self.nlev))
        return raised_op

    def __get_raised_op(self,
                        subsys: str, op: str | tuple[str],
                        op_spec: Callable['...', Qobj] | None = None
                        ) -> Qobj:
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
        nsys = max(self.lbl_to_idx.values()) + 1  # number of subsystems
        # list of subspace operators to be tensored together to form full hilbert space operator
        oper_list: np.ndarray[Qobj] = np.empty(nsys, dtype=object)
        # index of each subsystem in tensor operation
        subsys_idx = self.lbl_to_idx[subsys]
        # form array of identity operators acting on the corresponding subsystem (correspondence by tensor operation order)
        for key, sys in self.subsystems.items():
            idx = self.lbl_to_idx[key]
            nlev = sys.nlev
            oper_list[idx] = qt.qeye(nlev)
        # replace only the operator at subsys_idx with operator we would like to promote
        if isinstance(op, tuple):
            arg_ops: list[Qobj] = [self.subsystems[subsys][lbl] for lbl in op]
            subsys_op = op_spec(*arg_ops)
        else:
            # subsystem operator we would like to promote to full hilbert space
            subsys_op = self.subsystems[subsys][op]
        oper_list[subsys_idx] = subsys_op
        raised_op = qt.tensor(oper_list)
        return raised_op

    def is_diag(self) -> bool:
        diags = self.H.diag()
        H_diag = qt.qdiags(diags, 0, self.H.dims)
        return qt.isequal(self.H, H_diag, self._tol)

    def bare_comp_state(self,
                        i, j, k):
        """Function to obtain product computational states from subsystems in\
        `subsystems`. """""
        subsys_dims = [self.subsystems[lbl].nlev
                       for lbl in ['A', 'B', 'C']]
        idxs = [i, j, k]
        subsys_states = [qt.basis(dim, x)
                         for dim, x in zip(subsys_dims, idxs)]
        prod_state = qt.tensor(subsys_states)
        prod_state = self.align_basis(prod_state)
        return prod_state

    def comp_state_idx(self,
                       i, j, k):
        bare_state = self.bare_comp_state(i, j, k)
        overlap = np.ndarray(
            [abs((bare_state.dag() * dressed_state).tr())
                for dressed_state in self.eigenstates]
        )
        idx = np.argmax(overlap)
        return idx

    def comp_state_vector(self,
                          i, j, k):
        state_idx = self.comp_state_idx(i, j, k)
        return self.eigenstates[state_idx]

    def __getitem__(self,
                    i, j, k):
        return self.comp_state_vector(i, j, k)

    def __eq__(self, other: CompositeSystem) -> bool:
        if self._tol != other._tol:
            warnings.warn(
                'The two CompositeSystem objects being compared do not have the same _tol attribute.\n'
                + f' Leftmost has _tol={self._tol}, rightmost _tol={other._tol}\n' +
                'Equality operation may be assymetric'
            )
        if not qt.isequal(self.H, other.H):
            return False
        if self.lbl_to_idx != other.lbl_to_idx:
            return False
        if self.subsystems != other.subsystems:
            return False
        return True

    def _hashable_subsys(self) -> tuple[tuple[int]]:
        subsys_hashes = []
        for lbl, sys in self.subsystems.items():
            subsys_hashes.append(hash(sys))
        return tuple(subsys_hashes)

    def __hash__(self):
        if self._immutable:
            return self._class_idx
        return hash(
            ((x for x in self.H.full()),
             (hash(sys) for sys in self.subsystems.values()))
        )
