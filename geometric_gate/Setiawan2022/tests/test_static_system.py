# Unit tests for static_system.py
DEBUG = False
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from typing import TypeAlias, Callable, Any

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
import optimize_subsystem as opt
import subsystems
from subsystems import Subsystem
from composite_systems import CompositeSystem
import static_system

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

#helper function
def is_diag(op:Qobj, tol:float = 1e-9)->bool:
    diags = op.diag()
    diag_op = qt.qdiags(diags,offsets=0, dims=op.dims)
    return qt.isequal(op, diag_op, tol)

#begin tests
def test_get_params():
    module_params = static_system.get_params(yaml_path)
    assert module_params == ct_params, 'Parameter import failure'

def test_build_bare_system():
    module_bare_sys = static_system.build_bare_system(ct_params)
    #handroll composite system
    stable_levels:int = 5
    flux_A:Subsystem = opt.build_optimized_system(flux_constr, flux_A_params, stable_levels)
    flux_B:Subsystem = opt.build_optimized_system(flux_constr, flux_B_params, stable_levels)
    transmon:Subsystem = opt.build_optimized_system(transmon_constr, transmon_params, stable_levels)
    subsys_dict:dict[str,Subsystem] = {'A': flux_A,
                                       'B': flux_B,
                                       'C': transmon}
    subsys_idxs:dict[str,Subsystem] = {'A':0,
                                       'B':1,
                                       'C':2}

    handroll_bare_sys = CompositeSystem(subsys_dict,
                                       subsys_idxs)
    assert handroll_bare_sys == module_bare_sys,\
    'Handrolled bare systems do not match module-built bare system'

def test_build_interaction_H():
    gs = ct_params['interaction']
    bare_sys = static_system.build_bare_system(ct_params)
    interaction_H = static_system.build_interaction_H(gs, bare_sys)
    assert not is_diag(interaction_H), 'Interaction Hamiltonian must not be diagonal'

def test_build_static_system():
    module_static_sys = static_system.build_static_system(ct_params)
    #test is_diag
    assert module_static_sys.H.isherm, 'Static System Hamiltonian not Hermitian'
    assert not is_diag(module_static_sys.H), 'Static System Hamiltonian should not be diagonal'

########### Interactive Section #################

def plot_H():
    module_static_sys = static_system.build_statidc_system(ct_params)
    qt.hinton(abs(module_static_sys.H.full()))

def plot_eigenen():
    module_static_sys = static_system.build_static_system(ct_params)
    eigenen = module_static_sys.H.eigenenergies()
    plt.scatter(range(5), eigenen[:5])
    plt.show()

def plot_eigenen_varying_params(kwargs:dict[str,tuple[list[int],dict[str,Any]]],
                                constr:Callable['...',CompositeSystem],
                                constr_lbl:str,
                                max_level:int=5)->None:
    subsys_lbls = ['A','B','C']
    for key in kwargs:
        print(f'making plot for {key}')
        vary_params:list[int] = kwargs[key][0]
        const_params:dict[str,Any] = kwargs[key][1]
        fig, ax = plt.subplots()
        fig_subs, ax_subs = plt.subplots(3)
        min_arg = vary_params[0]
        max_arg = vary_params[-1]
        ftitle = f'{constr_lbl}_eigenen_varying_{key}_from_{min_arg}_to_{max_arg}'
        title = f'{constr_lbl} Eigenenergy Varying {key} From {min_arg} to {max_arg}'
        ftitle_subs = f'{constr_lbl}_Subsystems_eigenen_varying_{key}_from_{min_arg}_to{max_arg}'
        title_subs = f'Eigenen Varying {key} from {min_arg} to {max_arg}'
        ax.set_title(title)
        eigenen_list = np.empty((len(vary_params),max_level), 
                                dtype=float)
        subsys_eigenen_list = np.empty( [3] + list(eigenen_list.shape),
                                       dtype=float)
        for i, value in enumerate(vary_params):
            kwarg:dict[str,Any] = {key:value}
            kwarg.update(const_params)
            if key=='nlev':
                kwarg['stabilize'] = False
            sys:CompositeSystem = constr(ct_params, **kwarg)
            eigenen:np.ndarray[float] = sys.H.eigenenergies()[:max_level]
            eigenen -= eigenen[0] # make sure zero of energy is equal to energy of the ground state for all models
            eigenen_list[i,:] = eigenen
            for n, subsys_lbl in enumerate(subsys_lbls):
                subsys:Qobj = sys.subsystems[subsys_lbl]
                subsys_eig:np.ndarray[float] = subsys.H.eigenenergies()[:max_level]
                subsys_eig -= subsys_eig[0]
                subsys_eigenen_list[n,i,:] = subsys_eig 

        for i in range(eigenen_list.shape[1]):
            eigen_over_n = eigenen_list[:,i]
            ax.plot(vary_params, eigen_over_n, label=f'energy level {i}')
        for n, lbl in enumerate(subsys_lbls):
            ax_subs[n].set_title(f'{constr_lbl} Subsys {lbl} '+title_subs)
            for i in range(subsys_eigenen_list.shape[2]):
                eigen_over_n = subsys_eigenen_list[n,:,i]
                ax_subs[n].plot(vary_params, eigen_over_n, label=f'energy level {i}')
        ax.legend()
        fig.savefig(f'../plots/{ftitle}')
        fig_subs.tight_layout()
        fig_subs.savefig(f'../plots/{ftitle_subs}')

def run_plots(stable_nlev:list[int], trunc:list[int], nlev:list[int], max_level:int=5)->None:

    kwargs = {
        'stable_nlev': (stable_nlev,
                        {'trunc':max_level,
                         'min_nlev':max_level}),
        'trunc': (trunc,
                  {'stable_nlev': 25}),
        'nlev': (nlev,
                 {'stabilize': False})        
            }
    print('Making plots for bare system')
    plot_eigenen_varying_params(kwargs,static_system.build_bare_system, 'bare_system')
    print('Making plots for dressed static system')
    plot_eigenen_varying_params(kwargs, static_system.build_static_system, 'static_system')
    print('Done.')

stable_nlev = list(range(5,16))
trunc = list(range(5,20))
nlev = list(range(5,60))

    
if __name__=='__main__':
    run_plots(stable_nlev, trunc, nlev)
if DEBUG:
    run_plots(stable_nlev, trunc, nlev)


    