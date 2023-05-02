"""File to pre-build pulse profiles ahead of simulating dynamics with\
 qutip.

 A pulse with geometric phase `w`, gate time `tg` and max amplitude\
 `omega_0` (omega_0 as defined in ../config/pulse_parameters.yaml) \
 should be saved as `f'{w}_{tg}ns_{omega_0}hz_pulse.npy'`
"""
import sys
sys.path.append('../src')
from pulse import Pulse 
from composite_systems import CompositeSystem
import static_system
import os 
import yaml
import qutip as qt 
import numpy as np 
from typing import TypeAlias 

Qobj:TypeAlias = qt.Qobj

# extract configuration parameters
def get_params(path:str)->dict[str,float|dict[str,float]]:
    with open(path,'r') as stream:
        try:
            params:dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

#define constants
ct_params:dict[str,float|dict[str,float]] = \
    get_params('../config/circuit_parameters.yaml')
p_params:dict[str,float|dict[str,float]] = \
    get_params('../config/pulse_parameters.yaml')
default_tg:float = p_params['tg']
default_amp:float = p_params['omega_0']
prebuilt_pulses = os.listdir('../pulses')
ct:CompositeSystem = static_system.build_static_system(ct_params)
# utility funcs

def is_prebuilt(geo_phase:str|int,
                 tg:str|int=default_tg, 
                 amp:float|int=default_amp)->bool:
    fname:str = f'{geo_phase}_{tg}ns_{amp}hz_pulse.npy'
    return fname in prebuilt_pulses 

def build_pulse(geo_phase:float,
                phase_label:str=None,
                dt:float=0.01,
                tg:float=default_tg,
                amp:float=default_amp)->None:
    if phase_label is None:
        phase_label = str(geo_phase)
    if is_prebuilt(phase_label, tg, amp):
        return None
    pulse_params:dict[str,float|dict[str,float]] = {
        **p_params,
        'tg':tg,
        'amp':amp
    }
    fname = f'../pulses/{phase_label}_{tg}ns_{amp}hz_pulse.npy'
    tlist = np.arange(0,tg,dt)
    pulse_generator:Pulse = Pulse(pulse_params=pulse_params,
                  circuit=ct,
                  dt=dt)
    pulse_generator.build_pulse(tlist, geo_phase,fname)
    
def build_pulses()->None:
    with open('../pulses/pulses.yaml','r') as stream:
        try:
            pulse_specs = yaml.safe_load_all(stream)
        except yaml.YAMLError as exc:
            print(exc)
        for spec in pulse_specs:
            build_pulse(**spec)

if __name__ == '__main__':
    build_pulses()