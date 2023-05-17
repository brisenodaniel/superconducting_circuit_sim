import yaml
from os import mkdir
from qutip import Qobj, qsave, qload
from os.path import isdir, isfile
import numpy as np
from typing import Any, TypeVar, TypeAlias
from gate import GateProfile, PulseProfile, PulseConfig

T = TypeVar('T')
PulseDict = TypeVar('PulseDict')
GateDict = TypeVar('GateDict')
DataObj = TypeVar('DataObj', PulseConfig, PulseProfile, GateProfile)

#filename generators

def build_pulse_fname(pulse_config: PulseConfig | PulseProfile,
                      pulse_array: np.ndarray[float] | None = None) -> str:
    # if pulse_array is not provided, pulse_config must be of type PulseProfile
    if pulse_array is None:
        pulse_array = pulse_config.pulse
    return  f'{pulse_config.name}-{hash(pulse_config)}'

def build_pulse_component_fname(pulse_desc: PulseConfig | PulseProfile,
                                component_name: str) -> str:
    return f'{pulse_desc.name}-{component_name}-{hash(pulse_desc)}'

def build_gate_fname(gate: Gate) -> str:
    return f'{gate.name}-{hash(gate)}'

######## File Writing

def cache_pulse(pulse: PulseConfig | PulseProfile,
                pulse_array: np.ndarray[float] | None = None) -> None:
    # if pulse_array is None, pulse must be of type PulseProfile
    fname = build_pulse_fname(pulse, pulse_array)
    save_desc(pulse, fname, 'pulse')
    np.save(fname, pulse_array)

def cache_pulse_component(pulse: PulseConfig | PulseProfile,
                          comp_name: str,
                          component_array: np.ndarray[float] | None = None) -> None:
    if component_array is None:
        component_array = pulse.profiled_components[comp_name]
    fname = build_pulse_component_fname(pulse, comp_name)
    save_component_desc(pulse.name, comp_name, fname)
    np.save(fname, component_array)

def cache_gate(gate: GateProfile) -> None:
    fname = build_gate_fname(gate)
    save_desc(gate, fname, mode='Gate')
    unitary = gate.unitary
    trajectories = gate.trajectories
    data: dict[str, Qobj | dict[str, np.ndarray[complex]]] = \
        {'unitary': unitary, 'trajectories': trajectories}
    qsave(data, f'../output/sim_results/{fname}')

############# File logging

def save_component_desc(pulse_name: str, component_name: str, fname: str)\
    -> None:
    cache = get_cache_description('pulse')
    cache[pulse_name]['save_components'][component_name]['file'] = fname
    with open('../output/pulses/cache_desc.yaml', 'w') as yaml_file:
        yaml.dump(cache, yaml_file)

def save_desc(target: DataObj, fname: str, mode: str = 'pulse') -> None:
    assert mode in ['pulse', 'Pulse', 'gate', 'Gate', 'sim_results'],\
        f'{mode} not a valid mode parameter'
    directory = ''
    if mode in ['pulse', 'Pulse']:
        directory = '../output/pulses/'
    elif mode in ['gate', 'Gate', 'sim_results']:
        directory = '../output/sim_results/'
    desc = target.as_dict()
    desc['file'] = fname
    cache = get_cache_description(mode)
    name = desc.pop['name']
    cache[name] = desc
    with open(directory+'cache_desc.yaml', 'w') as yaml_file:
        yaml.dump(cache, yaml_file)


############ File Reading



def load_pulse(pulse_config: PulseConfig) -> np.ndarray[float]:
    cache = get_cache_description('pulse')
    pulse_name: str = pulse_config.name
    assert pulse_name in cache, \
        f'pulse {pulse_name} has not been saved in ../output/pulses'
    fname:str = cache[pulse_name]['fname']
    return np.load(f'../output/pulses/{fname}')

def load_pulse_component(pulse_config: PulseConfig, component_name: str)\
    -> np.ndarray[complex | float]:
    cache: dict[str, PulseDict] = get_cache_description('pulse')
    pulse_name: str = pulse_config.name
    assert pulse_name in cache, \
        f'pulse {pulse_name} has not been saved in ../output/pulses'
    pulse_components: dict[str, str] = cache[pulse_name]
    assert component_name in pulse_components, \
        f'component {component_name} has not been profiled for {pulse_name}'
    fname: str = pulse_components[component_name]['fname']
    return np.load(f'../output/pulses/{fname}')

def load_gate(gate_name) -> dict[str, Qobj | dict[str, np.ndarray[complex]]]:
    cache = get_cache_description('Gate')
    assert gate_name in cache, \
        f'{gate_name} has not been cached'
    fname = '../output/sim_results/' + cache[gate_name]['file']
    return qload(fname)

#def load_pulse(gate_lbl:str,
               #tg:float,
               #omega_0:float,
               #dt:float,
               #dir:str='../output/pulses',
               #**other_spec)->np.ndarray[float]:
    #fname = build_pulse_path(gate_lbl, tg, omega_0, dt, dir,**other_spec)
    #return np.load(fname)


def get_params(path:str)->dict[str,float|dict[str,float]]:
    with open(path, 'r') as stream:
        try:
            params: dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def get_ct_params()->dict[str,float|dict[str,float]]:
    return get_params('../config/circuit_parameters.yaml')


def get_pulse_defaults()->dict[str,float|dict[str,float]]:
    return get_params('../config/pulse_parameters.yaml')


def get_pulse_specs()->dict[str,float|dict[str,float]]:
    return get_params('../config/pulses.yaml')


def get_cache_description(cache_lbl: str | None = None) -> dict[str, Pulse] | None:
    directory: str = ''
    if cache_lbl in ['pulse', 'Pulse', None]:
        directory = '../output/pulses'
    elif cache_lbl in ['gate', 'Gate', 'sim_results']:
        directory = '../output/sim_results'
    else:
        return None
    fname = directory + '/cache_desc.yaml'
    if is_file(fname):
        return get_params(fname)
    else:
        return None

# Coupler params g vary
# implement relaxation and dissapation
# Ej  may vary in transmon coupler by 5%
# T1, T2 80ms, vary by up to 50%, 40ms - 80ms

def is_file(path:str)->bool:
    return isfile(path)
