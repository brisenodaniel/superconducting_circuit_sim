# Unit tests for pulse.py
DEBUG = False
#package imports 
import qutip as qt
import numpy as np
# module imports
#set up path
if DEBUG:
    import sys
    sys.path.append('./src')
    ct_path = './config/circuit_parameters.yaml'
    pulse_path = './config/pulse_parameters.yaml'
elif __name__== '__main__':
    import sys
    sys.path.append('../src')
    ct_path = '../config/circuit_parameters.yaml'
    pulse_path = '../config/pulse_parameters.yaml'
else:
     ct_path = './config/circuit_parameters.yaml'    
     pulse_path = './config/pulse_parameters.yaml'
import static_system
from state_labeling import CompCoord, CompTensor, get_dressed_comp_states
from composite_systems import CompositeSystem, Qobj
from subsystems import Subsystem
from pulse import Pulse, StaticPulseAttr

# import params
ct_params:dict[str, dict[str,float]] = static_system.get_params(ct_path)
pulse_params:dict[str,float|dict[str,float]] = static_system.get_params(pulse_path)

# Set up circuit hilbert space 
ct:CompositeSystem = static_system.build_static_system(ct_params)
c_coord:CompCoord
c_states:CompTensor
c_states, c_coord = get_dressed_comp_states(ct)

#numerical error tolerance
tol = 1e-9

#begin test defs
def test_static_pulse_attr_constr()->None:
    static_attr:StaticPulseAttr = StaticPulseAttr(pulse_params, ct)
    nlev:int = ct.H.eigenenergies().shape[0]
    assert static_attr.nlev == nlev, 'Wrong number of eigenstates\
        initialized'
    
def test_static_pulse_attr_state()->None:
    static_attr:StaticPulseAttr = StaticPulseAttr(pulse_params, ct)
    states:dict[str,tuple[int,3]] = {
        'ge0': (0,1,0),
        'eg0': (1,0,0),
        'ee0': (1,1,0),
        'ge1': (0,1,1),
        'gf0': (0,2,0),
        'gg1': (0,0,1)
    }
   
    for state_lbl, state_idx in states.items():
        #check indexing by tuple
        s_test:Qobj = static_attr.state(state_idx)
        s_correct:Qobj = c_states[state_idx]
        assert qt.isequal(s_test, s_correct, 1e-9),\
        'State indexing by tuple in StaticPulseAttr object is broken'

        #check indexing by string
        s_test:Qobj = static_attr.state(state_lbl)
        assert qt.isequal(s_test, s_correct, 1e-9),\
        'State indexing by string in StaticPulseAttr object is broken'

    #check indexing by int
    eigenstates:np.ndarray[Qobj] = ct.H.eigenstates()[1]
    for i in range(20):
        s_test:Qobj = static_attr.state(i)
        s_correct:Qobj = eigenstates[i]
        assert qt.isequal(s_test, s_correct, 1e-9),\
        'State indexing by int in StaticPulseAttr object is broken'

def test_static_attr_eigenen()->None:
    static_attr:StaticPulseAttr = StaticPulseAttr(pulse_params, ct)
    eigenen:np.ndarray[float] = ct.H.eigenenergies()
    eigenen -= eigenen[0]
    states:dict[str,tuple[int,3]] = {
        'ge0': (0,1,0),
        'eg0': (1,0,0),
        'ee0': (1,1,0),
        'ge1': (0,1,1),
        'gf0': (0,2,0),
        'gg1': (0,0,1)
    }
    for state_lbl, state_idx in states.items():
        #check indexing by tuple
        s_coord:int = c_coord[state_idx]
        e_ans:float = eigenen[s_coord]
        e_test:float = static_attr.eigenen(state_idx)
        assert abs(e_ans-e_test) <=tol,\
        f'Eigenen indexing by tuple in StaticPulseAttr is broken. '\
        +f'For {state_idx} expected {e_ans}, got {e_test}'

        #check indexing by string
        e_test = static_attr.eigenen(state_lbl)
        assert abs(e_ans-e_test) <=tol,\
        f'Eigenen indexing by string in StaticPulseAttr is broken. '\
        +f'for {state_lbl} expected {e_ans}, got {e_test}'
    #check indexing by int
    for i in range(20):
        e_test:float = static_attr.eigenen(i)
        e_ans:float = eigenen[i]
        assert abs(e_ans-e_test) <=tol,\
        f'Eigenen indexing by int in StaticPulseAttr is broken. '\
        +f'for {i} expected {e_ans}, got {e_test}'

def test_static_attr_trans_freq()->None:
    static_attr:StaticPulseAttr = StaticPulseAttr(pulse_params, ct)
    eigenen:np.ndarray[float] = ct.H.eigenenergies()
    eigenen -= eigenen[0]
    states:dict[str,tuple[int,3]] = {
        'ge0': (0,1,0),
        'eg0': (1,0,0),
        'ee0': (1,1,0),
        'ge1': (0,1,1),
        'gf0': (0,2,0),
        'gg1': (0,0,1)
    }
    for state_lbl, state_idx in states.items():
        for state2_lbl, state2_idx in states.items():
            s_coord:int = c_coord[state_idx]
            s2_coord:int = c_coord[state2_idx] 
            freq_ans:float = eigenen[s_coord] - eigenen[s2_coord]
            #check indexing by tuple
            freq_test:float = static_attr.trans_freq(state_idx, state2_idx)
            assert abs(freq_ans - freq_test)<=tol,\
            'transition frequency indexed by tuples broken in StaticPulseAttr. '+ \
            f'For {state2_lbl} to {state_lbl} transition, expected freq {freq_ans}, got {freq_test}'

            #check indexing by string
            freq_test:float = static_attr.trans_freq(state_lbl, state2_lbl)
            assert abs(freq_ans - freq_test)<=tol,\
            'transition frequency indexed by strings broken in StaticPulseAttr. '+ \
            f'For {state2_lbl} to {state_lbl} transition, expected freq {freq_ans}, got {freq_test}'

        #check indexing by int
        for i in range(20):
            for j in range(20):
                freq_ans:float = eigenen[i] - eigenen[j]
                freq_test:float = static_attr.trans_freq(i,j)
                assert abs(freq_ans - freq_test)<=tol,\
            'transition frequency indexed by ints broken in StaticPulseAttr. '+ \
            f'For {i} to {j} transition, expected freq {freq_ans}, got {freq_test}'
    
def test_mod_trans_detunings()->None:
    static_attr:StaticPulseAttr = StaticPulseAttr(pulse_params, ct)
    flux_lbls:list[str] = ['A', 'B']
    pm:list[int] = [-1, 1]
    states:dict[str,tuple[int,3]] = {
        'ge0': (0,1,0),
        'eg0': (1,0,0),
        'ee0': (1,1,0),
        'ge1': (0,1,1),
        'gf0': (0,2,0),
        'gg1': (0,0,1)
    }
    #test that function runs for all combinations of values
    for sys_lbl in flux_lbls:
        for sgn in pm:
            for s1 in states:
                for s2 in states:
                    static_attr.mod_trans_detunings(sys_lbl, s1, s2, sgn)
    # test against handrolled values of delta
    for sys_lbl in ['A','B']:
        s1:str = 'ge1'
        s2:str = 'ee0'
        s3:str = 'gg1'
        trans_freq_s1_s2:float = static_attr.trans_freq(s1, s2)
        trans_freq_s1_s3:float = static_attr.trans_freq(s1, s3)
        trans_freq_s2_s3:float = static_attr.trans_freq(s2, s3)

        delta_s1_s2_m:float = trans_freq_s1_s2 - static_attr.w_mod[sys_lbl]
        delta_s1_s2_p:float = trans_freq_s1_s2 + static_attr.w_mod[sys_lbl]
        delta_s1_s3_m:float = trans_freq_s1_s3 - static_attr.w_mod[sys_lbl]
        delta_s1_s3_p:float = trans_freq_s1_s3 + static_attr.w_mod[sys_lbl]
        delta_s2_s3_m:float = trans_freq_s2_s3 - static_attr.w_mod[sys_lbl]
        delta_s2_s3_p:float = trans_freq_s2_s3 + static_attr.w_mod[sys_lbl]

        assert delta_s1_s2_m == static_attr.mod_trans_detunings(sys_lbl,
                                                                s1,
                                                                s2,
                                                                -1)
        assert delta_s1_s2_p == static_attr.mod_trans_detunings(sys_lbl,
                                                                s1,
                                                                s2,
                                                                1)
        assert delta_s1_s3_m == static_attr.mod_trans_detunings(sys_lbl,
                                                                s1,
                                                                s3,
                                                                -1)
        assert delta_s1_s3_p == static_attr.mod_trans_detunings(sys_lbl,
                                                                s1,
                                                                s3,
                                                                1)
        assert delta_s2_s3_m == static_attr.mod_trans_detunings(sys_lbl,
                                                                s2,
                                                                s3,
                                                                -1)
        assert delta_s2_s3_p == static_attr.mod_trans_detunings(sys_lbl,
                                                                s2,
                                                                s3,
                                                                1)
        

# Begin Pulse Test

def test_pulse_constr()->None:
    pulse:Pulse = Pulse(pulse_params, ct)

#test only public methods in pulse
def test_pulse_delta_wC()->None:
    pulse:Pulse = Pulse(pulse_params, ct)
    tg:float = pulse_params['tg']
    tlist:np.ndarray[float] = np.linspace(0, tg, 10)
    for t in tlist:
        pulse.delta_wC(t, np.pi/2)


def test_pulse_vectorized_delta_wC()->None:
    pulse:Pulse = Pulse(pulse_params, ct)
    tg:float = pulse_params['tg']
    tlist:np.ndarray[float] = np.linspace(0, tg, 10)
    results = pulse.delta_wC(tlist, np.pi/2)
        

if DEBUG:
    # pulse:Pulse = Pulse(pulse_params, ct)
    # tg:float = pulse_params['tg']
    # pulse.delta_wC(1, np.pi/2)
    test_pulse_vectorized_delta_wC()


