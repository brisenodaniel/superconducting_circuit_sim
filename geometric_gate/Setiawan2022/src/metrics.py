# Python file to implement fidelity measurements of gates
import qutip as qt
import numpy as np
import os 
import sys
from functools import singledispatch, reduce
from typing import TypeAlias, Any, TypeVar
from composite_systems import CompositeSystem
from geometric_gate import run_pulse
from file_io import build_qu_fname, is_file
from state_labeling import get_dressed_comp_states, CompCoord, CompTensor

Qobj:TypeAlias = qt.Qobj
T = TypeVar('T')
char_to_idx:dict[str,int] = {'g': 0,
                            '0': 0,
                            'e': 1,
                            '1': 1,
                            'f': 2,
                            '2': 2}
str_to_qubit_idx:dict[str,tuple[int,int,int]] = {
    'ggg': (0,0),
    'egg': (1,0),
    'geg': (0,1),
    'eeg': (1,1)
}

qubit_idx_to_str:np.ndarray[np.ndarray[str]]= np.array(
    [['ggg', 'egg'],
     ['egg', 'eeg']], dtype=str)

qubit_idx_to_product_coord = np.array(
    [[[0,0,0], [1,0,0],
      [1,0,0], [1,1,0]]]
)

def get_state_idx(state_lbl:str)->tuple[int]:
    state_idx = [char_to_idx[char] for char in state_lbl]
    return tuple(state_idx)


def get_two_qubit_subspace_basis(H:Qobj)\
    ->tuple[np.ndarray[np.ndarray[int]], np.ndarray[np.ndarray[Qobj]]]:
    # H must be in bare product basis
    comp_states:CompTensor
    comp_idx: CompCoord
    comp_states, comp_idx = get_dressed_comp_states(H)
    basis_idxs:np.ndarray[np.ndarray[int]] = np.zeros(shape=[2,2], dtype=int)
    basis_vecs:np.ndarray[np.ndarray[Qobj]] = np.zeros(shape=[2,2], dtype=object)
    for i,j in np.ndindex(basis_vecs.shape):
        vec_product_coord = qubit_idx_to_product_coord[i,j]
        basis_vecs[i,j] = comp_states[vec_product_coord]
        basis_idxs[i,j] = comp_idx[vec_product_coord]
    return basis_idxs, basis_vecs 

def get_evolved_state(s0:tuple[int,int], geo_phase:float, gate_lbl:float, tg:float,
                      omega_0:float, dt:float, pulse_kwargs:dict[str,Any]={})\
                      ->tuple[Qobj,CompositeSystem]:
    args = locals() #make copy of arguments for recursive call
    s0 = qubit_idx_to_str[s0[0], s0[1]]
    fname = build_qu_fname(gate_lbl+f'{s0}-s0', tg, omega_0, dt)
    if is_file(fname):
        evolved_sys:T
        circuit:CompositeSystem
        evolved_sys, circuit = qt.qload(fname)
        return evolved_sys.states[-1], circuit
    else:
        evolved_sys:T = run_pulse(geo_phase, gate_lbl, s0, tg, omega_0, dt, **pulse_kwargs)
        qt.qsave(evolved_sys, fname)
        return get_evolved_state(**args)

def get_unitary_elem(vect1:Qobj, vect2:Qobj,
                     geo_phase:float, gate_lbl:str, tg:float, omega_0:float,
                     dt:float, basis_matrix:np.ndarray[np.ndarray[Qobj]], pulse_kwargs:dict[str,Any]={})->\
                        tuple[complex,CompositeSystem]:
    if isinstance(vect1, tuple[int,int]):
        vect1 = basis_matrix[vect1]
    if isinstance(vect2, tuple[int,int]):
        vect2 = basis_matrix[vect2]
    Uvect2:Qobj
    circuit:CompositeSystem
    Uvect2, circuit = get_evolved_state(vect2, geo_phase, gate_lbl, tg, omega_0, dt, pulse_kwargs)
    return vect1.dag()*Uvect2, circuit

# @get_unitary_elem.register 
# def get_unitary_elem(vect1:tuple[int,int], vect2:tuple[int,int],geo_phase:float,
#                      gate_lbl:str, tg:float, omega_0:float, dt:float, 
#                      basis_matrix:np.ndarray[np.ndarray[Qobj]], pulse_kwargs:dict[str,Any]|None=None)\
#                      ->tuple[complex,CompositeSystem]:
#     vect1 = basis_matrix[vect1]
#     vect2 = basis_matrix[vect2]
#     return get_unitary_elem(vect1, vect2, geo_phase,
#                             gate_lbl, tg, omega_0, dt, 
#                             pulse_kwargs)


def build_unitary(geo_phase:float, gate_lbl:str, tg:float, omega_0:float,
                  dt:float, pulse_kwargs:dict[str,Any]={})->Qobj:
    #run pulse on ground state to obtain circuit info
    circuit:CompositeSystem
    _, circuit = get_evolved_state([0,0], geo_phase, 
                                   gate_lbl, tg, 
                                   omega_0, dt, 
                                   pulse_kwargs)
    # get hilbert space dim
    H = circuit.H 
    hilbert_dim = reduce(lambda x,y: x*y, H.dims[0])
    #build empty unitary matrix
    U = np.zeros(shape=[hilbert_dim,hilbert_dim], dtype=complex)
    #get dressed eigenvectors corresponding to two qubit subspace
    q_subspace_basis_idxs:CompCoord
    q_subspace_basis_states:CompTensor
    q_subspace_basis_idxs, q_subspace_basis_states = get_two_qubit_subspace_basis(H)
    #populate elements of U corresponding to two qubit subspace
    subspace_dims = q_subspace_basis_idxs.shape
    for i,j in np.ndindex(subspace_dims):
        for k,l in np.ndindex(subspace_dims):
            u_idx_ij = q_subspace_basis_idxs[i,j]
            u_idx_kl = q_subspace_basis_idxs[k,l]
            U_ij_kl = get_unitary_elem((i,j),(k,l), geo_phase,
                                       gate_lbl, tg, omega_0, dt,
                                       q_subspace_basis_states, pulse_kwargs)
            U[u_idx_ij,u_idx_kl] = U_ij_kl 
    return U

if __name__=='__main__':
    build_unitary(np.pi, 'CZ', 130, 1.135, 0.01)

    






        


