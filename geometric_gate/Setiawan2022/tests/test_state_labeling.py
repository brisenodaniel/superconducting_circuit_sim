# Unit tests for state_labeling.py
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

import subsystems
from subsystems import Subsystem
from composite_systems import CompositeSystem
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
    comp_states = state_labeling.get_dressed_comp_states(sys, nstate)
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

