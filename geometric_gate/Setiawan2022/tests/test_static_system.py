# Unit tests for static_system.py
DEBUG = False
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from typing import TypeAlias, Callable

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

def test_build_bare_systems():
    module_bare_sys = static_system.build_bare_systems(yaml_path)
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
    bare_sys = static_system.build_bare_systems(yaml_path)
    interaction_H = static_system.build_interaction_H(gs, bare_sys)
    assert not is_diag(interaction_H), 'Interaction Hamiltonian must not be diagonal'

def test_build_static_system():
    module_static_sys = static_system.build_static_system(yaml_path)
    #test is_diag
    assert module_static_sys.H.isherm, 'Static System Hamiltonian not Hermitian'
    assert not is_diag(module_static_sys.H), 'Static System Hamiltonian should not be diagonal'


def plot_H():
    module_static_sys = static_system.build_static_system(yaml_path)
    qt.hinton(abs(module_static_sys.H.full()))

def plot_eigenen():
    module_static_sys = static_system.build_static_system(yaml_path)
    eigenen = module_static_sys.H.eigenenergies()
    plt.scatter(range(5), eigenen[:5])
    plt.show()

if DEBUG:
    test_build_bare_systems()


    