""" File contains functions and objects used to generate geometric pulse
 as described in Method 2 of Setiawan et. al. 2022 in the literature folder
"""
from __future__ import annotations
from functools import cache, cached_property, singledispatchmethod, singledispatch
import numpy as np
from typing import TypeAlias
from collections import abc
import qutip as qt
import yaml
import os
import static_system
from composite_systems import CompositeSystem
from state_labeling import CompTensor, CompCoord, get_dressed_comp_states
from dataclasses import dataclass
from scipy.integrate import quad
DeltaDict:TypeAlias = dict[str,dict[tuple[int,3], dict[tuple[int,3], dict[int,float]]]]
Qobj:TypeAlias = qt.Qobj

# the following object populates with static coefficients in pulse
@dataclass
class StaticPulseAttr:
    class_idx = 0
    @staticmethod
    def __get_unique_id()->int:
        StaticPulseAttr.class_idx +=1
        return StaticPulseAttr.class_idx
    
    def __init__(self, 
                 pulse_params:dict[dict[float]], 
                 circuit:CompositeSystem, 
                 n_comp_state:int=5)->StaticPulseAttr:
        # give current object it's id
        self._id = StaticPulseAttr.__get_unique_id()
        # Set up circuit and state indexing
        self._ct:CompositeSystem = circuit
        self._ct.freeze() # Markes CompositeSystem as immutable object for fast-hashing
        self._params:dict[dict[float]] = pulse_params
        self._n_comp_state:int = n_comp_state
        self._eigenstates:np.ndarray[Qobj]
        self._eigenenergies:np.ndarray[float]
        self._eigenenergies, self._eigenstates= self._ct.H.eigenstates()
        # set ground state to 0 energy for consistency
        self._eigenenergies -= self._eigenenergies[0]
        self._nlev = len(self._eigenenergies)
        self._comp_coord:CompCoord
        self._comp_states:CompTensor
        self._comp_states, self._comp_coord  = \
            get_dressed_comp_states(self._ct, n_comp_state)
        self.w_mod:dict[str,float] = {'A': self.trans_freq('ee0','ge1'),
                 'B': self.trans_freq('gf0','ge1')}
        
    def __hash__(self):
        return self._id
       
    @property
    def nlev(self)->int:
        return self._nlev 
    
    @property
    def comp_coord(self)->CompCoord:
        return self._comp_coord 
    
    @property
    def comp_states(self)->CompTensor:
        return self._comp_states
    
    @property
    def id(self)->int:
        return self._id

    def trans_freq(self, k:int|str|tuple[int,3], l:int|str|tuple[int,3])->float:
        e_k = self.eigenen(k)
        e_l = self.eigenen(l)
        return e_k - e_l


    @cache
    def __str_to_tuple(self, state_lbl:str)->tuple[int,3]:
        assert len(state_lbl)==3, f'Expected state label length 3, got {len(state_lbl)}'
        idx = [0,0,0]
        char_to_int:dict[str,int] = {'g':0, 'e':1, 'f':2,'0':0, '1':1, '2':2}
        for i, char in enumerate(state_lbl):
            idx[i] = char_to_int[char]
        return tuple(idx)
    
    def eigenen(self, state:str|int|tuple[int,3])->float:
        if isinstance(state, str):
            state = self.__str_to_tuple(state)
        if isinstance(state, tuple):
            state = self._comp_coord[state]
        return self._eigenenergies[state]
    
    def state(self, state_idx:str|tuple[int,3]|int)->Qobj:
        if isinstance(state_idx, str):
            state_idx = self.__str_to_tuple(state_idx)
        if isinstance(state_idx, int):
            return self._eigenstates[state_idx]
        else:
            return self._comp_states[state_idx]
    
    
    def state_idx(self, state:str|tuple[int,3]|Qobj|int)->int:
        idx = state # idx dummy variable to ensure no side effects on state
        if isinstance(idx, str):
            idx:tuple[int,3] = self.__str_to_tuple(state)
        if isinstance(state, abc.Iterable):
            idx:int = self._comp_coord[idx]
        elif isinstance(idx, Qobj):
            idx:int = np.array([qt.isequal(idx,eig) \
                            for eig in self._eigenstates]).nonzero()[0][0]
        return idx 
    
    
    @cache
    def mod_trans_detunings(self, 
                             sys:str, 
                             s1:tuple[int,3]|str|int, 
                             s2:tuple[int,3]|str|int, 
                             pm:int)->float:
        """Class method generates the detuning of the transition |s1> <--> |s2> from\
        the modulation tone applied to fluxonium sys, as seen in the capital delta term
        in (C2). So, for parameters `(sys, s1, s2, pm)`, function will return\
              Delta^sys_{s1,s2,pm} in equation (C2).

        Args:
            sys (str): A or B. Corresponds to fluxonium A or B.
            s1 (tuple[int,3]): Computational state 1 involved in transition |s2> -> |s1>, indexed\
             in the bare product basis
            s2 (tuple[int,3]): Computational state 2 involved in transition |s2> -> |s1>, indexed\
             in the bare product basis
            pm (int): +1 or -1. Corresponds to sigma index in equation C2

        Returns:
            float: Detuning of transition |s1> <--> |s2> from modulation tone applied to fluxonium sys.
        """
        assert pm in [-1,1], f'parameter `pm` must be 1 or -1, got {pm}'
        assert sys in ['A','B'], f'parameter sys must be A or B, got {sys}'
        if isinstance(s1,str):
            s1 = self.__str_to_tuple(s1)
        if isinstance(s2, str):
            s2 = self.__str_to_tuple(s2)
        w_mod:float = self.w_mod[sys]
        return self.trans_freq(s1,s2) + pm*w_mod
    

################### begin pulse def
class Pulse:
    
    def __init__(self,
                 pulse_params:dict[dict[float]],
                 circuit:CompositeSystem,
                 dt:float=0.01,
                 n_comp_states:int=5):
        self.static_attr:StaticPulseAttr = StaticPulseAttr(pulse_params, circuit, n_comp_states)
        self._ct:CompositeSystem = circuit
        if 'dt' in pulse_params:
            self._dt:float = pulse_params['dt']
        else:
            self._dt:float = dt
        self._params:dict[str,dict[str,float]|float] = pulse_params
        if 'n_comp_states' in pulse_params:
            self.n_comp_states:int = pulse_params['n_comp_states']
        else:
            self.n_comp_states:int = n_comp_states
        self._omega_0:float = 2*np.pi*self._params['omega_0']/pulse_params['tg']


    def __omega_A(self, t:float)->float:
        theta = self.__theta(t,0)
        dtheta = self.__theta(t,1)
        d2theta = self.__theta(t,2)
        pulse_env = np.sin(theta) + 4*np.cos(theta)*d2theta\
            /(self._omega_0**2 + 4*dtheta**2)
        # pulse_env = np.sin(self.__theta(t,0)) +\
        #                 4*np.cos(self.__theta(t,0))*self.__theta(t,2)\
        #                 /(self._omega_0**2 + 4*self.__theta(t,1)**2)
        return self._omega_0*pulse_env
    
    def __omega_B(self,t:float, geo_phase:float)->complex:
        i = complex(0,1)
        theta = self.__theta(t,0)
        dtheta = self.__theta(t,1)
        d2theta = self.__theta(t,2)
        phase = np.exp(i*self.__phase_arg(t, geo_phase))
        # pulse_env = np.cos(self.__theta(t,0)) - \
        #   4*np.sin(self.__theta(t,0))*self.__theta(t,2)\
        #   /(self._omega_0**2 + 4*self.__theta(t,1)**2)
        pulse_env = np.cos(theta) - \
            4*np.sin(theta)*d2theta/(self._omega_0**2 + 4*dtheta**2)
        return self._omega_0*phase*pulse_env

    def __phase_arg(self,t:float, geo_phase:float)->float:
        tg = self._params['tg']
        return geo_phase*np.heaviside(t-tg/2,1)
    
    def __theta(self, t:float, deriv:int=0)->float:
        #implements eq A2
        tg = self._params['tg']
        if 0<=t<=tg/2:
            return self.__theta_interval_1(t,deriv)
        if tg/2 < t<=tg:
            return self.__theta_interval_2(t,deriv)
        else:
            return 0 
   
    def __theta_interval_1(self, t:float, d:int)->float:
        tg = self._params['tg']
        derivs:list[float] = [
            np.pi/2,
            np.pi/(2*tg),
            np.pi/(2*tg**2)
        ]
        return derivs[d]*self.__polynom(t/tg, d=d)
        
    def __theta_interval_2(self, t:float, d:int)->float:
        tg = self._params['tg']
        derivs:list[float] = [
            (np.pi/2),
            (np.pi/(2*tg)),
            (np.pi/(2*tg**2))
        ]
        if d==0:
            return derivs[d]*(1-self.__polynom(t/tg-1/2, d=d))
        else:
            return -1*derivs[d]*self.__polynom(t/tg - 1/2, d=d)
        
    
    def __polynom(self, t:float, d:int=0)->float:
        #implements eq A3
        derivs:list[tuple[float,6]] = [
            (6*2**5, -15*2**4, 10*2**3, 5, 4, 3),
            (960, -960, 240, 4, 3, 2),
            (3840, -2280, 480, 3, 2, 1)
        ]
        return self.__p(t, *derivs[d])
    
    def __p(self, x:float, c1:float, c2:float, c3:float, e1:float, e2:float, e3:float)->float:
        return c1*x**e1 + c2*x**e2 + c3*x**e3
        
    @cache
    def __g_ac(self, t:float, geo_phase:float)->dict[str,float]:
        ge1_idx:int = self.static_attr.state_idx('ge1')
        ee0_idx:int = self.static_attr.state_idx('ee0')
        gf0_idx:int = self.static_attr.state_idx('gf0')
        return {
            'A': self.__omega_A(t)/(self.__adag_a[ge1_idx,ee0_idx]),
            'B':self.__omega_B(t,geo_phase)/(self.__adag_a[ge1_idx,gf0_idx])
        }
       
    
    @cached_property
    def highest_leakage_st(self)->int:
        leakage_states_idxes = [
            self.static_attr.comp_coord[i,j,k]\
            for i in range(self.n_comp_states)
            for j in range(self.n_comp_states)
            for k in range(self.n_comp_states)
        ]
        return max(leakage_states_idxes)

    #@cache
    def __delta_ek(self, t:float, state:tuple[int,3]|str, geo_phase:float)->float:
        #implements eq C2b
        leakage_states:list[int] = list(range(self.highest_leakage_st+1))
        sum_c2b = 0
        for l in leakage_states:
            for flux_lbl in ['A','B']:
                for sgn in [-1,1]:
                    sum_c2b += self.__C2b__summand(t, geo_phase, state, l, flux_lbl, sgn)
        return sum_c2b

    def __C2b__summand(self, 
                       t:float, 
                       geo_phase,
                       k:tuple[int,3]|str|int, 
                       l:int,
                       flux_lbl:str,
                       sgn:int )->float:
        """Function implements summand in equation C2b

        Args:
            t (float): current timestep
            k (tuple[int,3] | str | int): Label for state k in eq C2b
            l (int): Label for leakeage state l of total hamiltonian iterated over in eq C2b
            flux_lbl (str): A or B, label for state j in eq C2b
            sgn (int): +1 or -1, sigma in eq C2b
            
        Returns:
            float: Summand term index (j,sigma,l) in C2b
        """
        denominator:float = 4*self.static_attr.mod_trans_detunings(flux_lbl,
                                                         k,
                                                         l,
                                                         sgn)
        if abs(denominator)<=1e-3:
            return 0
        k_idx:int = self.static_attr.state_idx(k)
        adag_a:np.ndarray = self._adag_a_as_matrix #see next function definition for explanation of adag_a
        k_adaga_l:complex = adag_a[k_idx][l]
        g_ac:float = self.__g_ac(t, geo_phase)[flux_lbl]
        numerator:float = (g_ac*k_adaga_l).conjugate()*(g_ac*k_adaga_l)
        return numerator/denominator
    
    @cached_property
    def __adag_a(self)->Qobj:
        """this function is used only for optimization of bottleneck observed in
        profiling. Returns operator a.dag()*a in eigenbasis where a is the QHO destruction operator
        acting on the transmon coupler
        """
        adag_a:Qobj = self._ct.get_raised_op('C', ['a'], lambda a: a.dag()*a)
        eigenbasis:np.array[Qobj] = self._ct.H.eigenstates()[1]
        adag_a = adag_a.transform(eigenbasis)
        return adag_a
    
    @cached_property
    def _adag_a_as_matrix(self)->np.ndarray:
        """this function is used only for optimization of bottleneck observed in
        profiling. Returns full numpy matrix for operator a.dag()*a in eigenbasis where a is the QHO destruction operator
        acting on the transmon coupler
        """
        return self.__adag_a.full()

    
    def __delta_wmod(self, t:float, geo_phase:float)->dict[str,float]:
        delta_ge1:float = self.__delta_ek(t, 'ge1', geo_phase)
        delta_ee0:float = self.__delta_ek(t, 'ee0', geo_phase)
        delta_gf0:float = self.__delta_ek(t, 'gf0', geo_phase)
        return{
            'A': delta_ge1 - delta_ee0,
            'B': delta_ge1 - delta_gf0
        }
    
    @singledispatchmethod
    @cache
    def __w_mod(self, t:float, geo_phase:float, sys:str)->float:
        assert sys in ['A','B'], f'`sys` param must be A or B, got {sys}'
        deltas:dict[str,float] = self.__delta_wmod(t, geo_phase)
        w_a = self.static_attr.w_mod['A']
        w_b = self.static_attr.w_mod['B']
        if sys=='A':
            return w_a + deltas['A']
        else:
            return w_b + deltas['B']
        
    @__w_mod.register
    def __(self, tlist:abc.Iterable, geo_phase:float, sys:str)->np.ndarray[float]:
        return np.array([self.__w_mod(t, geo_phase, sys) for t in tlist])

    @singledispatchmethod 
    def delta_wC(self, t:float, geo_phase:float)->float:
        i = complex(0,1)
        sum_terms:list[float] = []
        gs:dict[str,float] = self.__g_ac(t, geo_phase)
        for flux in ['A','B']:
            ts:np.ndarray[float] = np.arange(0, t, self._dt)
            w_mods:np.ndarray[float] = self.__w_mod(ts, geo_phase, flux)
            phase_arg:float = np.trapz(w_mods, ts, self._dt)
            phase:complex = np.exp(-i*phase_arg)
            g_j:float = gs[flux]
            summand:float = (g_j*phase).real
            sum_terms.append(summand)
        return sum(sum_terms)
    
    @delta_wC.register
    def __(self, tlist:abc.Iterable, geo_phase:float)->np.ndarray[float]:
        return np.array([self.delta_wC(t, geo_phase) for t in tlist])
    

    def build_pulse(self, 
                    tlist:abc.Iterable[float], 
                    geo_phase:float, 
                    fname=None,
                    as_txt=False)->np.ndarray[float]:
        pulse = self.delta_wC(tlist, geo_phase)
        if fname is not None:
            if as_txt:
                np.savetxt(fname,pulse)
            else:
                np.save(fname, pulse)
        return pulse


############ Diagnostic functions
    def get_integrand_func(self,
                           tlist:np.ndarray[int], 
                           state:str,
                           geo_phase:float)->np.ndarray[float]:
        res_lst = [
            self.__delta_ek(t, state, geo_phase) for t in tlist
        ]
        return np.array(res_lst)
    
########### The following section is used to add pulses to the 
###########  `../pulses/` directory. To add a new pulse, write the 
###########  parameters of the new pulse to `../pulses/pulses.yaml`
###########  as specified in that file's documentation. Then, run 
###########  this file as a script from this directory `$python pulse.py`


# extract configuration parameters
def get_params(path:str)->dict[str,float|dict[str,float]]:
    with open(path,'r') as stream:
        try:
            params:dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

#define default constants
default_p_params:dict[str,float|dict[str,float]] = \
    get_params('../config/pulse_parameters.yaml')
default_tg:float = default_p_params['tg']
default_amp:float = default_p_params['omega_0']


def build_npy_fname(gate_lbl:str,
                    tg:float=default_tg,
                    amp:float=default_amp,
                    dt:float=1e-2,
                    include_dir:bool=True)->str:
    fname = f'{gate_lbl}_{tg}ns_{amp}GHz_{dt}dt.npy'
    if include_dir:
        fname = '../output/pulses/' + fname
    return fname

#define funcs
def is_prebuilt(gate_label:str|int,
                 tg:str|int=default_tg, 
                 amp:float|int=default_amp,
                 dt=1e-2)->bool:
    prebuilt_pulses = os.listdir('../output/pulses')
    fname:str = build_npy_fname(gate_label, tg, amp, dt, include_dir=False)
    #fname:str = f'{geo_phase}_{tg}ns_{dt}dt_{amp}GHz_pulse.npy'
    return fname in prebuilt_pulses 


def build_pulse(gate_label:str,
                geo_phase:float,
                circuit:CompositeSystem,
                tg:float=default_tg,
                amp:float=default_amp,
                dt:float=0.01)->np.ndarray[float]:
    fname = build_npy_fname(gate_label, tg, amp, dt)

    if is_prebuilt(gate_label, tg, amp, dt):
        return np.load(fname)
    else:
        pulse_params:dict[str,float|dict[str,float]] = {
            **default_p_params,
            'tg':tg,
            'omega_0':amp
        }
        tlist = np.arange(0,tg,dt)
        pulse_generator:Pulse = Pulse(pulse_params=pulse_params,
                    circuit=circuit,
                    dt=dt)
        pulse_spec = {gate_label:{
            'geo_phase': geo_phase,
            'tg': tg,
            'omega_0': amp,
            'dt': dt}}
        write_pulse_spec(pulse_spec)
        return pulse_generator.build_pulse(tlist, geo_phase,fname)
    
def write_pulse_spec(pulse_spec:dict[str,dict[str,float]])->None:
    current_specs = get_params('../config/pulses.yaml')
    updated_specs = {**current_specs, **pulse_spec}
    with open('../config/pulses.yaml', 'w') as yfile:
        try:
            yaml.safe_dump(updated_specs, yfile)
        except yaml.YAMLError as exc:
            print(exc)

def build_pulses(pulses_params:list[dict[str,float|str]]|None=None)->None:
    """function checks that all pulses specified in \
    `../config/pulses.yaml` are written to the `../output/pulses`
    directory as a .npy file. If an entry in pulses.yaml is
    found that does not have a corresponding .npy file, that 
    pulse will be generated and it's .npy file written.
    """
    if pulses_params is None:
        with open('../config/pulses.yaml','r') as stream:
            try:
                pulse_specs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        for gate_lbl, spec in pulse_specs.items():
            build_pulse(gate_label=gate_lbl, **spec)
    else:
        return list(
            [build_pulse(**pulse_params) for pulse_params in pulses_params]
        )

if __name__ == '__main__':
    pass
   # build_pulses()