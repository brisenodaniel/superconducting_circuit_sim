# Unit tests for composite_systems.py
DEBUG = False
import qutip as qt
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
import static_system
from subsystems import Subsystem
from composite_systems import CompositeSystem


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

#begin tests

def test_constructor():
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

    composite_system = CompositeSystem(subsys_dict,
                                       subsys_idxs)
    
    # hand-roll composite hamiltonian
    IA, IB, IC = qt.qeye(flux_A.nlev), qt.qeye(flux_B.nlev), qt.qeye(transmon.nlev)
    H_A = qt.tensor(flux_A.H, IB, IC)
    H_B = qt.tensor(IA, flux_B.H, IC)
    H_C = qt.tensor(IA, IB, transmon.H)

    H = H_A + H_B + H_C

    assert composite_system.H.isherm,\
    'Creation of composite system yielded a non-hermitian hamiltonian'

    assert all([subsys_dict[lbl] == composite_system.subsystems[lbl]\
                           for lbl in ['A','B','C']]), \
                           'Composite system subsystems changed during object creation'
    assert qt.isequal(H, composite_system.H, 1e-9),\
    'Creation of composite system did not yield correct hamiltonian'

def test_get_raised_op():
    nlev=50
    flux_A:Subsystem = opt.build_optimized_system(flux_constr, flux_A_params, nlev=nlev)
    flux_B:Subsystem = opt.build_optimized_system(flux_constr, flux_B_params, nlev=nlev)
    transmon:Subsystem = opt.build_optimized_system(transmon_constr, transmon_params, nlev=nlev)
    subsys_dict:dict[str,Subsystem] = {'A': flux_A,
                                       'B': flux_B,
                                       'C': transmon}
    subsys_idxs:dict[str,Subsystem] = {'A':0,
                                       'B':1,
                                       'C':2}
    sys:CompositeSystem = CompositeSystem(subsys_dict, subsys_idxs)
    n_A = sys.get_raised_op('A','n')
    n_B = sys.get_raised_op('B','n')
    a_C = sys.get_raised_op('C','a')
    n_C = sys.get_raised_op('C',['a'], lambda x: x.dag()+x)

    IA = qt.qeye(flux_A.nlev)
    IB = qt.qeye(flux_B.nlev)
    IC = qt.qeye(transmon.nlev)

    assert (qt.isequal(n_A, qt.tensor(flux_A['n'], IB, IC), 1e-9)),\
    'Flux A n operator incorrectly contructed'
    assert(qt.isequal(n_B, qt.tensor(IA, flux_B['n'],IC), 1e-9)),\
    'Flux B n operator incorrectly constructed'
    assert(qt.isequal(a_C, qt.tensor(IA, IB, transmon['a']),1e-9)),\
    'Transmon a operator incorrectly constructed'
    assert(qt.isequal(n_C, qt.tensor(IA, IB, transmon['a'].dag() + transmon['a']),1e-9)),\
    'Transmon n operator incorrectly constructed'




if DEBUG:
    test_constructor()
