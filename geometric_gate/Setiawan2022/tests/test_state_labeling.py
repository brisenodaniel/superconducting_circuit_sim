# Unit tests for state_labeling.py
DEBUG = False
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable
import yaml
from typing import TypeAlias
import vary_system_params as vparam

if DEBUG:
    import sys
    sys.path.append('./src')
    yaml_path = './config/circuit_parameters.yaml'
elif __name__== '__main__':
    import sys
    sys.path.append('../src')
    yaml_path = '../config/circuit_parameters.yaml'
else:
     yaml_path = './config/circuit_parameters.yaml'    

from composite_systems import CompositeSystem
import subsystems
import static_system
import state_labeling

Qobj: TypeAlias = qt.Qobj
j: complex = complex(0,1)

#extract circuit parameters
with open(yaml_path,'r') as stream:
    try:
        ct_params:dict[str,dict] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Subsystem Constructor Parameters
flux_param_lbls:list[str] = ['E_C', 'E_J', 'E_L', 'phi_ext']
transmon_param_lbls:list[str] = ['w','U']

flux_A_params:dict[str,float] = {lbl:ct_params['A'][lbl] for lbl in flux_param_lbls}
flux_B_params:dict[str,float] = {lbl:ct_params['B'][lbl] for lbl in flux_param_lbls}
transmon_params:dict[str,float] = {lbl:ct_params['C'][lbl] for lbl in transmon_param_lbls}

# subsystem constructors
flux_constr = subsystems.build_fluxonium_operators
transmon_constr = subsystems.build_transmon_operators

# build static system
sys = static_system.build_static_system(ct_params)

def interactive(stable_nlev):
    sys = static_system.build_static_system(ct_params, 
                                           stable_nlev=stable_nlev)
    comp_states, comp_coords = state_labeling.get_dressed_comp_states(sys,comp_states=3)
    ee0 = comp_states[1,1,0]
    ge1 = comp_states[0,1,1]
    gf0 = comp_states[0,2,0]
    a = sys.get_raised_op('C','a')

    ggg = comp_states[0,0,0]

    eigenen = sys.H.eigenenergies()
    ee0_en = eigenen[comp_coords[1,1,0]]
    ge1_en = eigenen[comp_coords[0,1,1]]
    gf0_en = eigenen[comp_coords[0,2,0]]
    ggg_en = eigenen[comp_coords[0,0,0]]

    energies = {'ee0':ee0_en,
                'ge1':ge1_en,
                'gf0':gf0_en,
                'ggg':ggg_en}

    matrix_elem_1 = abs((ee0.dag()*a.dag()*a*ge1).tr())
    matrix_elem_2 = abs((gf0.dag()*a.dag()*a*ge1).tr())
    return matrix_elem_1, matrix_elem_2, comp_coords, energies

def make_plots():
    matrix_elem_1:list[float] = []
    matrix_elem_2:list[float] = []
    computational_state_idxs = []
    ggg = []
    gf0 = []
    ge1 = []
    ee0 = []
    for i in range(3,16):
        m1, m2, c, e = interactive(i)
        matrix_elem_1.append(m1)
        matrix_elem_2.append(m2) 
        computational_state_idxs.append(c) 
        ggg.append(e['ggg'])
        gf0.append(e['gf0'])
        ge1.append(e['ge1'])
        ee0.append(e['ee0'])
    fig1, ax1 = plt.subplots()
    ax1.plot(range(3,16), matrix_elem_1, label='|<ee0|a*a|ge1>|')
    ax1.plot(range(3,16), matrix_elem_2, label='|<gf0|a*a|ge1>|')
    ax1.legend()
    fig1.savefig('./plots/unstable_matrix_elem.pdf')

    fig2, ax2 = plt.subplots()
    ax2.plot(range(3,16),ggg,label='ggg')
    ax2.plot(range(3,16), gf0, label='gf0')
    # ax2.plot(range(3,16), ge1, label='ge1')
    ax2.plot(range(3,16), ee0, label='ee0')
    ax2.legend()
    fig2.savefig('./plots/unstable_eigenen.pdf')



# begin tests

def test_get_product_comp_states():
    nstate=5
    comp_states = state_labeling.get_product_comp_states(sys,nstate)
    assert comp_states.shape == (nstate,nstate,nstate), 'Computational state tensor'+\
    f'has wrong dimensions, expected {(nstate,nstate,nstate)}, got {comp_states.shape}'
    assert isinstance(comp_states[0,0,0], Qobj),\
    f'Computational states should be Qobj, but has type {type(comp_states[0,0,0])}'


def test_get_dressed_comp_states():
    nstate = 5
    comp_states, _ = state_labeling.get_dressed_comp_states(sys, nstate)
    assert comp_states.shape == (nstate,nstate,nstate), 'Computational state tensor'+\
    f'has wrong dimensions, expected {(nstate,nstate,nstate)}, got {comp_states.shape}'
    assert isinstance(comp_states[0,0,0], Qobj),\
    f'Computational states should be Qobj, but has type {type(comp_states[0,0,0])}'
    assert all([isinstance(comp_states[n,m,k],Qobj)\
                for n in range(nstate)\
                    for m in range(nstate)\
                        for k in range(nstate)]),\
            'Computational states not all of same type'
    assert all([comp_states[n,m,k].isket \
                for n in range(nstate)\
                    for m in range(nstate)\
                        for k in range(nstate)]),\
            'Computational states not kets'
    
def test_transmon_annihilation_op_matrix_elems():
    comp_states, _ = state_labeling.get_dressed_comp_states(sys)
    a = sys.get_raised_op('C','a')
    ee0 = comp_states[1,1,0]
    ge1 = comp_states[0,1,1]
    gf0 = comp_states[0,2,0]

    assert abs( (ee0.dag()*a.dag()*a*ge1).tr() ) - 0.194 <= 2e-3,\
    f'Expected <ee0|a.dag()*a|ge1> = 0.194, got {abs((ee0.dag()*a.dag()*a*ge1).tr())}'

    assert abs( (gf0.dag()*a.dag()*a*ge1).tr() ) - 0.111 <=2e-3,\
    f'Expected <gf0|a.dag()*a|ge1> = 0.111, got {abs((gf0.dag()*a.dag()*a*ge1).tr())}'



def plot_labels_over_param_ranges(kwargs:dict[str,list[int], dict[str,Any]],
                           constr:Callable['...',CompositeSystem],
                           constr_lbl:str)->None:
    idx:dict[str,tuple[int,int,int]] = {
            'ggg': (0,0,0),
            'gf0': (0,2,0),
            'ge1': (0,1,1),
            'ee0': (1,1,0)
        }
    idx_lbls = ['ggg','gf0','ge1','ee0']
    data_dict:dict[str, dict[str, list[int] | list[CompositeSystem]]] =\
          vparam.vary_mod_params(kwargs, constr)
    print(data_dict.keys())
    for varied_param, plot_data in data_dict.items():
        param_range:list[int] = plot_data['param_range']
        systems:list[CompositeSystem] = plot_data['systems']
        comp_idxs:np.ndarray[np.ndarray[int]] = np.empty((len(param_range),4), dtype=int)
        for i, system in enumerate(systems):
            cidx = state_labeling.get_dressed_comp_states(system)[1]
            comp_idxs[i,:] = np.array([cidx[idx[lbl]] for lbl in idx_lbls])
        fname = f'{constr_lbl}_comp_state_idx_{varied_param}_from_{param_range[0]}_to_{param_range[-1]}'
        pname = f'{constr_lbl} Comp State Index vs. {varied_param}'
        fig, ax = plt.subplots()
        ax.set_title(pname)
        for i, state_lbl in enumerate(idx_lbls):
            cstate_idxs = comp_idxs[:,i]
            ax.plot(param_range, cstate_idxs, label=state_lbl)
        ax.legend()
        fig.savefig(f"../plots/test_labeling/{fname}.pdf")
    return None

def plot_eigenen_over_param_ranges(kwargs:dict[str,list[int], dict[str,Any]],
                           constr:Callable['...',CompositeSystem],
                           constr_lbl:str)->None:
    idx:dict[str,tuple[int,int,int]] = {
            'ggg': (0,0,0),
            'gf0': (0,2,0),
            'ge1': (0,1,1),
            'ee0': (1,1,0)
        }
    idx_lbls = ['ggg','gf0','ge1','ee0']
    data_dict:dict[str, dict[str, list[int] | list[CompositeSystem]]] =\
          vparam.vary_mod_params(kwargs, constr)
    print(data_dict.keys())
    for varied_param, plot_data in data_dict.items():
        param_range:list[int] = plot_data['param_range']
        systems:list[CompositeSystem] = plot_data['systems']
        comp_ens:np.ndarray[np.ndarray[float]] = np.empty((len(param_range),4), dtype=float)
        for i, system in enumerate(systems):
            cidx = state_labeling.get_dressed_comp_states(system)[1]
            eigenen = system.H.eigenenergies()
            comp_ens[i,:] = np.array([eigenen[cidx[idx[lbl]]] for lbl in idx_lbls])
        fname = f'{constr_lbl}_comp_state_eigenen_{varied_param}_from_{param_range[0]}_to_{param_range[-1]}'
        pname = f'{constr_lbl} Comp State Eigenenergies vs. {varied_param}'
        fig, ax = plt.subplots()
        ax.set_title(pname)
        for i, state_lbl in enumerate(idx_lbls):
            cstate_idxs = comp_ens[:,i]
            ax.plot(param_range, cstate_idxs, label=state_lbl)
        ax.legend()
        ax.set_xlabel(f'{varied_param} value')
        ax.set_ylabel('Eigenenergy')
        fig.savefig(f"../plots/test_labeling/{fname}.pdf")
    return None



trunc_kwarg = {
    'trunc': (list(range(5,16)),
              {'stable_nlev': 18})
}

constr_trunc = static_system.build_bare_system
constr_lbl_trunc = 'Static_system_with_18_stable_nlev'

def run_plots()->None:
    stable_nlev = list(range(5,16))
    trunc = list(range(5,16))
    nlev = list(range(5,60))
    max_level=5

    kwargs = {
        'stable_nlev': (stable_nlev,
                        {'trunc':max_level,
                         'min_nlev':max_level}),
        'trunc': (trunc,
                  {'stable_nlev': max(trunc)+2}),
        'nlev': (nlev,
                 {'stabilize': False})        
            }
    constr = static_system.build_static_system
    constr_lbl = 'Static_System'
    plot_labels_over_param_ranges(kwargs, constr, constr_lbl)
    plot_eigenen_over_param_ranges(kwargs,constr,constr_lbl)
   

if __name__=='__main__':
    run_plots()


if DEBUG:
    make_plots()
