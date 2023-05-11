
import yaml
from os import mkdir
from qutip import Qobj, qsave, qload
from os.path import isdir, isfile
import numpy as np
from typing import Any, TypeVar, TypeAlias

T = TypeVar('T')

def build_pulse_fname(gate_lbl:str,
                      tg:float,
                      omega_0:float,
                      dt:float,
                      other_spec:list[tuple[str,Any]]=[])->str:
    fname = f'{gate_lbl}_{tg}ns_{omega_0}GHz_{dt}dt'
    for lbl, val in other_spec:
        if lbl=='':
            fname += f'_{val}'
        else:
            fname += f'_{val}-{lbl}'
    return fname

def build_pulse_path(gate_lbl:str,
                     tg:float,
                     omega_0:float,
                     dt:float,
                     dir:str='../output/pulses',
                     other_spec:list[tuple[str,Any]]=[])->str:
    if not dir[-1] == '/':
        dir += '/'
    fname = build_pulse_fname(gate_lbl, tg, omega_0, dt, other_spec)
    return dir + fname

def save_pulse(pulse:np.ndarray[float],
               gate_lbl:str,
               tg:float,
               omega_0:float,
               dt:float,
               dir:str='../output/pulses',
               other_spec:list[tuple[str,Any]]=[])->None:
    fname = build_pulse_path(gate_lbl, tg, omega_0, dt, dir, other_spec)
    np.save(fname, pulse)

def load_pulse(gate_lbl:str,
               tg:float,
               omega_0:float,
               dt:float,
               dir:str='../output/pulses',
               other_spec:list[tuple[str,Any]]=[])->np.ndarray[float]:
    fname = build_pulse_path(gate_lbl, tg, omega_0, dt, dir, other_spec)
    return np.load(fname)

def is_pulse_cached(gate_lbl:str,
                      tg:float,
                      omega_0:float,
                      dt:float,
                      dir:str='../output/pulses',
                      other_spec:list[tuple[str,Any]]=[])->bool:
    fname = build_pulse_path(gate_lbl, tg, omega_0, dt, dir, other_spec)
    return isfile(fname)

def build_qu_fname(gate_lbl:str, 
                   init_state:str,
                   tg:float, 
                   omega_0:float, 
                   dt:float,
                   other_spec:tuple[tuple[str,Any]]={})->str:
    fname = f'{gate_lbl}_{init_state}-s0_{tg}ns_{omega_0}GHz_{dt}dt'
    for lbl, val in other_spec:
        if lbl=='':
            fname += f'_{val}'
        else:
            fname += f'_{val}-{lbl}'
    return fname

def build_qu_path(gate_lbl:str,
                  init_state:str,
                  tg:float,
                  omega_0:float,
                  dt:float,
                  dir:str='../output/trajectories/',
                  other_spec:list[tuple[str,Any]]=[])->str:
    if not dir[-1]=='/':
        dir += '/'
    fname:str = build_qu_fname(gate_lbl,
                               init_state,
                               tg,
                               omega_0,
                               dt,
                               other_spec)
    return dir + fname


def build_qu_folder_name(gate_lbl:str,
                         tg:float,
                         omega_0:float,
                         dt:float,
                         path_prefix:str='../output/trajectories/',
                         other_spec:list[tuple[str,Any]]=[])->str:
    if path_prefix[-1] != '/':
        path_prefix += '/'
    folname = f'{gate_lbl}_{tg}ns_{omega_0}GHz_{dt}dt'
    for lbl, val in other_spec:
        if lbl=='':
            folname+= f'_{val}'
        else:
            folname += f'_{val}-{lbl}'
    return path_prefix+folname 

def build_unitary_fname(gate_lbl:str,
                            tg:float,
                            omega_0:float,
                            dt:float,
                            other_spec:list[tuple[str,Any]]=[])->str:
    return build_qu_folder_name(gate_lbl+'_Unitary', tg, omega_0,
                                dt, path_prefix='', other_spec=other_spec)

def save_gate(sim_result:tuple[dict[str,T], Qobj],
              gate_lbl:float,
              tg:float,
              omega_0:float,
              dt:float,
              other_spec:list[tuple[str,Any]])->None:
    folname = build_qu_folder_name(gate_lbl, tg, omega_0, 
                                   dt, other_spec=other_spec)
    if not isdir(folname):
        mkdir(folname)
    trajectories, U = sim_result 
    #save unitary
    Ufname = build_unitary_fname(gate_lbl, tg, omega_0, dt, other_spec)
    qsave(folname+'/'+Ufname)
    #save trajectories
    for s0, traj in trajectories.items():
        save_trajectory(s0,traj, gate_lbl,
                        tg, omega_0, dt, path_prefix=folname,
                        other_spec=other_spec)
        
def save_trajectory(s0:str,
                    traj:T,
                    gate_lbl:float,
                    tg:float,
                    omega_0:float,
                    dt:float,
                    path_prefix:str='',
                    other_spec:list[tuple[str,Any]]=[])->None:
    fpath = build_qu_path(gate_lbl, s0, tg, omega_0, 
                          dt, dir=path_prefix, other_spec=other_spec)
    qsave(fpath, traj)

def is_trajectory_cached(init_state:str, gate_lbl:float, tg:float,
                         omega_0:float, dt:float, folname:str|None=None,
                         other_spec:list[tuple[str,Any]]=[])->bool:
    if folname is None:
        folname = build_qu_folder_name(gate_lbl, tg, omega_0, dt,
                                       other_spec=other_spec)
    fpath = build_qu_path(gate_lbl, init_state, tg, omega_0, dt, folname,
                          other_spec=other_spec)
    return isfile(fpath)

def is_unitary_cached(gate_lbl:float, tg:float, omega_0:float, dt:float,
                      folname:str|None=None, other_spec:list[tuple[str,Any]]=[]):
    if folname is None:
        folname = build_qu_folder_name(gate_lbl, tg, omega_0, dt,
                                       other_spec=other_spec)
    fname = build_unitary_fname(gate_lbl, tg, omega_0, dt, other_spec)
    fpath = folname + '/' + fname 
    return isfile(fpath)
    
def get_params(path:str)->dict[str,float|dict[str,float]]:
    with open(path,'r') as stream:
        try:
            params:dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def get_ct_params()->dict[str,float|dict[str,float]]:
    return get_params('../config/circuit_parameters.yaml')

def get_pulse_defaults()->dict[str,float|dict[str,float]]:
    return get_params('../config/pulse_parameters.yaml')

def get_pulse_specs()->dict[str,float|dict[str,float]]:
    return get_params('../config/pulses.yaml')

def is_file(path:str)->bool:
    return isfile(path)