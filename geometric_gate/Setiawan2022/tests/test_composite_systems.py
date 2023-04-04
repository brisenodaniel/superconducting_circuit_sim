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
    H_A = flux_A.H
    H_B = flux_B.H
    H_C = transmon.H
    H = qt.tensor([H_A,H_B,H_C])

    assert composite_system.H.isherm,\
    'Creation of composite system yielded a non-hermitian hamiltonian'

    assert all([subsys_dict[lbl] == composite_system.subsystems[lbl]\
                           for lbl in ['A','B','C']]), \
                           'Composite system subsystems changed during object creation'
    assert qt.isequal(H, composite_system.H, 1e-9),\
    'Creation of composite system did not yield correct hamiltonian'

if DEBUG:
    test_constructor()
