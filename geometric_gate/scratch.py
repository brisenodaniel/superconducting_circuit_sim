#!/usr/bin/env python3
from geo_gate_fluxonium_sim import *   
import pandas as pd
#check if matrix is diagonal
# static_system = build_static_system(nlev=9)
# H = static_system['H_dressed']
# H_bare = static_system['H_bare']
# def is_diag(H):
#     H_diag = qt.qdiags(H.diag(),offsets=0)
#     H_diff = H - H_diag
#     count_diff = np.count_nonzero(H_diff.full())
#     assert count_diff!=0, 'Non_diagonal Hamiltonian'
# qutrit_lev = [0,1,2]
# qutrit_names = ['g','e','f']
# idxs, _ = find_bare_states(H_bare)
# for lev_f1, name_f1 in zip(qutrit_lev, qutrit_names):
#     for lev_f2, name_f2 in zip(qutrit_lev, qutrit_names):
#         for lev_c, name_c in zip(qutrit_lev, qutrit_names):
#             lev_name = '{}{}{}'.format(name_f1, name_f2, name_c)
#             lev_idx = idxs[lev_f1, lev_f2, lev_c]
#             print('state {} index: {}'.format(lev_name, lev_idx))


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

def select_system_traits(system, traits):
    assert isinstance(traits, Collection) and not isinstance(traits,dict), \
        '`traits` must be a string, or a collection of strings'
    if traits=='all':
        return system
    elif isinstance(traits,str):
        return system[traits]
    else:
        return {trait: system[trait] for trait in traits}
        
def track_states_over_nlev(nlev_min, nlev_max, states:list, return_system_trait=None):
    nrow = nlev_max - nlev_min + 1
    ncol = len(states) 
    nlevs = range(nlev_min, nlev_max + 1)
    state_idx_matrix = np.empty((nrow,ncol), dtype=int)
    if return_system_trait is not None:
        systems = np.empty(nrow, dtype=object)
    for n, nlev in enumerate(nlevs):
        print('%'*40)
        print('\trunning {} levels'.format(nlev))
        print('\tconfig {}/{}'.format(n+1, nrow))
        static_system = build_static_system(nlev)
        H_bare = static_system['H_bare']
        if return_system_trait is not None:
            systems[n] = select_system_traits(static_system,
                                              return_system_trait)
        for m, state in enumerate(states):
            state_idx, _ = find_state(H_bare, state)
            state_idx_matrix[n,m] = state_idx
        print('\tDone.')
        print('@'*40)
    state_names = states_tostr_list(states)
    state_idx_df = pd.DataFrame(state_idx_matrix,
                                index = nlevs,
                                columns = state_names)
    if return_system_trait is not None:
        return state_idx_df, systems
    else:
        return state_idx_df

def track_eigenenergies_over_nlev(nlev_min, nlev_max, states:list, memorize_hamiltonians=False):
    if memorize_hamiltonians:
        system_trait = 'H_dressed'
        state_idx_df, H_list = track_states_over_nlev(nlev_min,
                                              nlev_max,
                                              states,
                                              system_trait)
    else:
        state_idx_df = track_states_over_nlev(nlev_min,
                                              nlev_max,
                                              states)
        H_list = list([build_static_system(n)['H_dressed']
                       for n in range(nlev_min, nlev_max+1)])
    energy_df = energy_df_from_idxs(state_idx_df, H_list)
    return energy_df
    
def energy_df_from_idxs(idx_df, H_list):
    energy_arr = np.empty(idx_df.to_numpy().shape)
    states = idx_df.columns
    for n, nlev in enumerate(idx_df.index):
        eigenenergies = H_list[n].eigenenergies()
        for m, state in enumerate(states):
            idx = idx_df.loc[nlev, state]
            en = eigenenergies[idx]
            energy_arr[n,m] = en
    energy_df = pd.DataFrame(energy_arr,
                             index=idx_df.index,
                             columns=idx_df.columns)
    return energy_df


def states_tostr_list(states):
    return list([state_tostr(state) for state in states])

def str_tostate_list(labels):
    return list([str_tostate_list(label) for label in labels])

def state_tostr(state):
    tostr_dict = {0: 'g', 1: 'e', 2: 'f'}
    return ''.join([tostr_dict[x] for x in state])
def str_tostate(label):
    tostate_dict = {'g':0, 'e':1, 'f':2}
    label_itemized = list(label)
    return tuple((tostate_dict[sub_label] for sub_label in label_itemized))
    
comp_states = [
    (1,1,0),
    (0,1,1),
    (0,2,0),
    (1,0,0),
    (0,1,0),
    (0,0,1)
]
#states_df = track_states_over_nlev(3,20,comp_states)
energy_df = track_eigenenergies_over_nlev(3,20,comp_states)

#display(states_df)
display(energy_df)
plt.figure()
#states_df.plot(title='state index')
# plt.show()
energy_df.plot(title='eigenenergies')
plt.show()
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
