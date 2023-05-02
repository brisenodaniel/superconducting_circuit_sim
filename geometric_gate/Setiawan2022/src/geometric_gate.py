from __future__ import annotations
from typing import Callable, TypeAlias, TypeVar
import qutip as qt
import numpy as np
import pulse
import yaml
from functools import singledispatch
from composite_systems import CompositeSystem
from state_labeling import get_dressed_comp_states, CompCoord, CompTensor
import static_system
Qobj:TypeAlias = qt.Qobj
T = TypeVar('T')
def get_params(path:str)->dict[str,float|dict[str,float]]:
    with open(path,'r') as stream:
        try:
            params:dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

#define constants
ct_params:dict[str,float|dict[str,float]] = get_params('../config/circuit_parameters.yaml')
pl_params:dict[str,float|dict[str,float]] = get_params('../config/pulse_parameters.yaml')
circuit:CompositeSystem = static_system.build_static_system(ct_params)
comp_states:CompTensor
comp_state_idx:CompCoord
comp_states, comp_state_idx = get_dressed_comp_states(circuit)

# define funcs
@singledispatch
def get_state(state_lbl:tuple[int,int,int])->Qobj:
    return comp_states[state_lbl]
@get_state.register
def __(state_lbl:str)->Qobj:
    state_idx = str_to_tuple(state_lbl)
    return get_state(state_idx)
@get_state.register
def __(state:Qobj)->Qobj:
    return state

def str_to_tuple(state_lbl:str)->tuple[int,int,int]:
    char_to_int:dict[str,int] = {
        'g': 0,
        '0': 0,
        'e': 1,
        '1': 1,
        'f': 2,
        '2': 2
    }
    state_idx = tuple([char_to_int[char] for char in state_lbl])
    return state_idx


def get_mesolve_formatted_H(circuit:CompositeSystem, 
                            pulse_array:np.ndarray[float])->list[Qobj|list[Qobj, np.ndarray[float]]]:
    H0:Qobj = circuit.H 
    #build H_mod, eq 31
    H_mod:Qobj = circuit.get_raised_op('C', ['a'], lambda a: a.dag()*a)
    return [H0, [H_mod, pulse_array]]
    

def run_pulse(geo_phase:float,
              gate_lbl:str, 
              s0:Qobj|tuple[int,int,int]|str='ggg',
              tg:float=pl_params['tg'],
              omega_0:float=pl_params['omega_0'],
              dt:float=1e-2,
              **kwargs)->T:
    if not isinstance(s0, Qobj):
        s0 = get_state(s0)
    pulse_array:np.array[float] = pulse.build_pulse(gate_label=gate_lbl, 
                                                    geo_phase=geo_phase, 
                                                    circuit=circuit,
                                                    tg=tg, 
                                                    omega_0=omega_0, 
                                                    dt=dt,
                                                    **kwargs)
    H_mesolve:list[Qobj|list[Qobj,np.ndarray[float]]] = get_mesolve_formatted_H(circuit, pulse_array)
    tlist = np.arange(0, tg, dt)
    result:T = qt.mesolve(H_mesolve, s0, tlist)
    return result

def run_pulses_in_config(default_s0:str='ggg')->dict[str,tuple[T,dict[str,float]]]:
    s0:Qobj = get_state('ggg')
    default_config:dict[str,float|dict[str,float]] = get_params('../config/pulse_parameters.yaml')
    pulse_configs:dict[str,dict[str,float]] = get_params('../config/pulses.yaml')
    results:dict[str,T] = {}
    for gate_lbl, set_params in pulse_configs.items():
        if 's0' not in set_params:
            init_states = [default_s0]
        else:
            init_states = set_params.pop('s0')
            if not isinstance(init_states, list):
                init_states = [init_states]
        gate_params = default_config
        gate_params.update(set_params)
        for s0 in init_states:
            gate_spec = {spec_lbl: gate_params[spec_lbl]\
                        for spec_lbl in ['tg','omega_0','dt']}
            phi0 = get_state(s0)
            results[f'{gate_lbl}_{s0}-s0'] =(run_pulse(gate_lbl=gate_lbl, s0=phi0, save_config=False, **gate_params),
                                gate_spec)
    return results 

