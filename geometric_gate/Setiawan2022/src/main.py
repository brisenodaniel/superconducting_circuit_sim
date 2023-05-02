import geometric_gate
import qutip as qt
import numpy as np
from typing import TypeVar
T = TypeVar('T')


def build_qu_fname(gate_lbl:str, 
                   tg:float, 
                   omega_0:float, 
                   dt:float,
                   include_dir:bool=True)->str:
    fname = f'{gate_lbl}_{tg}ns_{omega_0}GHz_{dt}dt'
    if include_dir:
        fname = '../output/trajectories/' + fname 
    return fname

def save_sim_result(gate_lbl:str, tg:float, omega_0:float, dt:float, result, **kwargs):
    fname = build_qu_fname(gate_lbl, tg, omega_0, dt)
    qt.qsave(result, fname)

def save_sim_results(results:dict[str,tuple[T,dict[str,float]]])->None:
    for gate_lbl in results:
        mesolve_result, gate_spec = results[gate_lbl]
        save_sim_result(gate_lbl=gate_lbl, result=mesolve_result, **gate_spec)

def run_all_sims()->None:
    results = geometric_gate.run_pulses_in_config()
    save_sim_results(results)

if __name__=='__main__':
    run_all_sims()
        
    