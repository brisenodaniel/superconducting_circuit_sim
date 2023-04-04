"""File contains functions used to identify hamiltonian eigenstates
with computational states.
"""


import qutip as qt
import numpy as np
from typing import TypeAlias
from composite_systems import CompositeSystem
from subsystems import Subsystem

Qobj:TypeAlias = qt.Qobj
CompTensor:TypeAlias = np.ndarray[np.ndarray[np.ndarray[Qobj]]]
GramTensor:TypeAlias = np.ndarray[np.ndarray[np.ndarray[np.ndarray[float]]]]


def get_product_comp_states(sys:CompositeSystem, 
                            comp_states:int=5)->CompTensor:
    """Function to obtain product computational states from subsystems in\
    `sys`.

    Args:
        sys (CompositeSystem): Quantum System for which to obtain product \
         computational states. Computational states will be computed from subsystems\
         in `sys`.
        comp_states (int, optional): Computational states to extract. Function will extract\
         product states made up of the lowest `comp_states` eigenstates. Defaults to 5.

    Returns:
        CompTensor: 3-dimensional array with CompTensor[n,m,k] containing the product state\
        qt.tensor(sys_0[n], sys_1[m], sys_2[k]), where sys_i refers to the eigenstates of the \
        ith system labeled in sys.idxs_to_lbl
    """
    
    ordered_subsystems:list[Subsystem] = list([sys.subsystems[sys.idxs_to_lbl[i]]\
                                                for i in range(3)])
    ordered_bases:list[np.ndarray[Qobj]] = list([ord_sys.H.eigenstates()[1]\
                                                 for ord_sys in ordered_subsystems])

    bare_comp_state_tensor = np.empty((comp_states,comp_states,comp_states), dtype=object)
    for n in range(comp_states):
        for m in range(comp_states):
            for k in range(comp_states):
                s0 = ordered_bases[0][n]
                s1 = ordered_bases[1][m]
                s2 = ordered_bases[2][k]
                bare_comp_state_tensor[n,m,k] = qt.tensor(s0, s1, s2)
    return bare_comp_state_tensor

def get_state_overlap(comp_state_tensor:CompTensor,
                     eigenstates:np.ndarray[Qobj],
                     comp_states:int=5)->GramTensor:
    """Function computes the overlap between each state in `comp_state` and states in `eigenstates`.\
    Overlap between states |n> and |m> is defined as |<n|m>|.

    Args:
        comp_state_tensor (CompTensor): Nested array with 3 coordinates, containing an \
            approximation of the computational state |n,m,k> in `comp_state_tensor[n,m,k]`.
        eigenstates (np.ndarray[Qobj]): Eigenstates which we would like to correlate with \
        approximate computational states in `comp_state_tensor`. Must have length grater or equal to \
        `comp_states**3`
        comp_states (int, optional): Number of states in each subsystem to be considered part of the \
            computational subspace. Defaults to 5.

    Returns:
        GramTensor: A multidimensional float array containing at coordinate [n,m,k,l] the overlap between\
        computational state |n,m,k> and eigenstate |l>.
    """
    gram_tensor:GramTensor = np.empty((comp_states, comp_states, comp_states, len(eigenstates)),
                                      dtype=float)
    for n in range(comp_states):
        for m in range(comp_states):
            for k in range(comp_states):
                for l, eigenstate in enumerate(eigenstates):
                    overlap = abs((comp_state_tensor[n,m,k].dag()*eigenstate).tr())
                    gram_tensor[n,m,k,l] = overlap
    return gram_tensor

def match_states(overlap_tensor:GramTensor,
                 eigenstates:np.ndarray[Qobj])->CompTensor:
    """Function to map an eigenstate to a computational state, given a GramTensor with the \
    overlap between each eigenstate and each computational state.

    Args:
        overlap_tensor (GramTensor): 4-dimensional array where `overlap_tensor[n,m,k,l]` contains\
         the overlap of theorized computational state |n,m,k> and eigenstate `eigenstates[l]`. Overlap\
         between states |x>, |y> is defined as |<x|y>|.
        eigenstates (np.ndarray[Qobj]): List of eigenstates to label as computational states.

    Returns:
        CompTensor: 3-d array where index [n,m,k] corresponds to the eigenstate in `eigenstates`\
         with largest overlap to theorized to theorized state |n,m,k>.
    """
    n_states:int = overlap_tensor.shape[0]
    comp_states:CompTensor = np.empty((n_states,n_states, n_states), dtype=object)
    for n in range(n_states):
        for m in range(n_states):
            for k in range(n_states):
                overlap_row:np.ndarray[float] = overlap_tensor[n,m,k,:]
                eigenstate_idx:int = np.argmax(overlap_row)
                comp_states[n,m,k] = eigenstates[eigenstate_idx]
    return comp_states 


def get_bare_comp_states(sys:CompositeSystem,
                         comp_states:int=5)->CompTensor:
    """Function to obtain bare computational states 

    Args:
        sys (CompositeSystem): Quantum system for which to obtain bare computational states
        comp_states (int, optional): Function will extract computational states corresponding\
            to the product state of the lowest `comp_states` eigenstates of the subystems in `sys. \
            Defaults to 5.

    Returns:
        CompTensor: 3-dimensional array with CompTensor[n,m,k] containing the bare eigenstate corresponding\
        to the product state created from the nth, mth, and kth lowest energy levels of the corresponding \
        subsystem
    """
    product_states:CompTensor = get_product_comp_states(sys, comp_states)
    bare_H:Qobj = sys.get_bare_hamiltonian()
    bare_states:np.ndarray[Qobj] = bare_H.eigenstates()[1]
    gram_tensor:GramTensor = get_state_overlap(product_states, bare_states, comp_states)
    return match_states(gram_tensor, bare_states)

def get_dressed_comp_states(sys:CompositeSystem,
                            comp_states:int=5)->CompTensor:
    """Function to obtain dressed computational states

    Args:
        sys (CompositeSystem): Quantum system for which to obtain dressed computational states.
        comp_states (int, optional): Function will extract the computational states corresponding to\
            product states of the lowest `comp_states` eigenstates of the subsystem hamiltonians in `sys`\
            . Defaults to 5.

    Returns:
        CompTensor: 3-dimensional array with CompTensor[n,m,k] containing the dressed eigenstate corresponding\
        to the product state created from the nth, mth, and kth lowest energy level of the corresponding subsystem.
    """
    bare_comp_states:CompTensor = get_bare_comp_states(sys, comp_states)
    dressed_eigenstates:np.ndarray[Qobj] = sys.H.eigenstates()[1]
    gram_tensor:GramTensor = get_state_overlap(bare_comp_states, dressed_eigenstates, comp_states)
    return match_states(gram_tensor, dressed_eigenstates)


def get_evolved_comp_state(sys_0:CompositeSystem|CompTensor,
                           sys_1:CompositeSystem|Qobj,
                           comp_states:int=5)->CompTensor:
    """Function to obtain computational states of a dynamical hamiltonian at time T, given the hamiltonian at \
    time 0 or the computational states at time 0. 

    Conditions: Function assumes that the hamiltonian at time t is a small pertubation of the hamiltonian at\
        time 0.

    Args:
        sys_0 (CompositeSystem | CompTensor): Either the entire composite system at time 0\
            time or the computational states at time 0.\
        sys_1 (CompositeSystem | Qobj): Either the entire composite system evolved to time t, or the hamiltonian\
            of the system at time t.
        comp_states (int, optional): Function will extract the computational states corresponding to\
            product states of the lowest `comp_states` eigenstates of the subystem hamiltonians in `sys`. Defaults to 5.

    Returns:
        CompTensor: 3-dimensional array with CompTensor[n,m,k] containing the eigenstate corresponding to the\
            computational state |n,m,k> at time 0.
    """
    if isinstance(sys_0, CompositeSystem):
        comp_states_0:CompTensor = get_dressed_comp_states(sys_0, comp_states)
    else:
        comp_states_0:CompTensor = sys_0
    if isinstance(sys_1, CompositeSystem):
        evolved_eigenstates:np.ndarray[Qobj] = sys_1.H.eigenstates()[1]
    else:
        evolved_eigenstates:np.ndarray[Qobj] = sys_1.eigenstates()[1]
    overlap_tensor:GramTensor = get_state_overlap(comp_states_0, evolved_eigenstates, comp_states)
    return match_states(overlap_tensor, evolved_eigenstates)