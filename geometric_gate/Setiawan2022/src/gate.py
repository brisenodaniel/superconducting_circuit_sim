# Given a circuit and a pulse config, generates the pulse
# and simulates the action of the pulse on a circuit. From
# simulated trajectories constructs the gate unitary in the 
# qubit subspace
import qutip as qt
import numpy as np
import file_io 
from numpy import sqrt
from composite_systems import CompositeSystem
from pulse import Pulse, StaticPulseAttr
from state_labeling import CompCoord, CompTensor
from typing import TypeAlias, TypeVar, Any 

dummy_res = qt.mesolve(qt.qeye(2), qt.basis(2,0),[0,1])
Qobj:TypeAlias = qt.Qobj
result:TypeAlias = qt.solver.Result

def profile_gate(gate_lbl:str,
                 geo_phase:float,
                 tg:float,
                 omega_0:float,
                 dt:float,
                 circuit:CompositeSystem,
                 psi_0:Qobj,
                 init_states: list[str|list[tuple[str,float]]]=[],
                 profile_pulse_components:dict[str,dict[str,complex]]|None=None,
                 **kwargs)->tuple[result,Qobj]:
    #add initial states needed to compute unitary and see phase changes to init_states
    mandatory_init_states = ['ggg','egg','geg','eeg',
                             [('ggg',1/sqrt(2)),('egg', 1/sqrt(2))], # +1 eigenstate of sigmax on q1, q2=g
                             [('geg', 1/sqrt(3)),('eeg', 1/sqrt(2))], # +1 eigenstate of sigmax on q1, q2=e
                             [('ggg',1/sqrt(2)), ('egg',1.j/sqrt(2))], # +1 eigenstate of sigmay on q1, q2=g
                             [('geg', 1/sqrt(2)),('eeg', 1.j/sqrt(2))] # +1 eigenstate of sigmay on q1, q2=e
                             ]
    init_states += mandatory_init_states
    #create pulse object and obtain pulse
    pulse_generator = Pulse(kwargs,)
    

