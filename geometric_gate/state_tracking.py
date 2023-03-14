"""
File generates computational states assuming
computational states are fock states of excitations
in individual fluxonia and the transmon coupler, and 
not eigenstates of the dressed hamiltonian
"""
import qutip as qt
from typing import Collection

str_to_idx = {'g': 0, 'e': 1, 'f': 2}
idx_to_str = ['g','e','f']

def str_to_idx_list(state_label:str):
    assert all([char in 'gef' for char in state_label]), \
                'Invalid state label'
    return list([str_to_idx[char] for char in state_label])

def idx_list_to_str(state_idx:Collection):
    assert all([idx in (0,1,2) for idx in state_idx]),\
    'Invalid state index'
    return ''.join([idx_to_str[idx] for idx in state_idx])

def comp_state(nlev, state):
    if isinstance(state, str):
        state = str_to_idx_list(state)
    return qt.basis([3,3,3], state)
           
