#!/usr/bin/env python3
from geo_gate_fluxonium_sim import *   
#check if matrix is diagonal
static_system = build_static_system(nlev=9)
H = static_system['H_dressed']
H_bare = static_system['H_bare']
def is_diag(H):
    H_diag = qt.qdiags(H.diag(),offsets=0)
    H_diff = H - H_diag
    count_diff = np.count_nonzero(H_diff.full())
    assert count_diff!=0, 'Non_diagonal Hamiltonian'
qutrit_lev = [0,1,2]
qutrit_names = ['g','e','f']
idxs, _ = find_bare_states(H_bare)
for lev_f1, name_f1 in zip(qutrit_lev, qutrit_names):
    for lev_f2, name_f2 in zip(qutrit_lev, qutrit_names):
        for lev_c, name_c in zip(qutrit_lev, qutrit_names):
            lev_name = '{}{}{}'.format(name_f1, name_f2, name_c)
            lev_idx = idxs[lev_f1, lev_f2, lev_c]
            print('state {} index: {}'.format(lev_name, lev_idx))


def vary_nlev(nlev_min, nlev_max, ncomp=3):
    ncol = nlev_max - nlev_min + 1
    nrow = ncomp**3
    state_idx_matrix = np.empty((nrow,ncol), dtype=int)
    nlev_list = range(nlev_min, nlev_max+1)
    clev_list = list(range(ncomp))
    for i, nlev in enumerate(nlev_list):
        static_system = build_static_system(nlev)
        H_bare = static_system['H_bare']
        idxs, _ = find_bare_states(H_bare)
        col_index = 0
        for m in clev_list:
            for l in clev_list:
                for k in clev_list:
                    state_idx_matrix[col_index,i] = idxs[m,l,k]
                    col_index += 1
    return state_idx_matrix
        
arr = vary_nlev(3,28)
print(arr)

#qubit_states = [ggg,egg,geg,gge,gee,ege,eeg]
#qubit_names = ['ggg', 'egg', 'geg', 'gge', 'gee', 'ege', 'eeg']
#unpack = lambda arr, idx: arr[idx[0], idx[1], idx[2]]
#
#for state, name in zip(qubit_states, qubit_names):
    #print('state {} index:'.format(name), unpack(idxs,state))
#ee0 = states[1,1,0]
#ge1 = states[0,1,1]
#gf0 = states[0,2,0]
#print(ee0.dims)
#a = ops_C['a']
#denom_1 = ee0.dag()*a.dag()*a*ge1
#denom_2 = gf0.dag()*a.dag()*a*ge1
#print(denom_1)
#print(denom_2)
#print(denom_1 - denom_2)
#print('eigenenergies', H_0.eigenenergies()[:5])
