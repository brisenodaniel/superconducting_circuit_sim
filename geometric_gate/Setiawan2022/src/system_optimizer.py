"""Quantum subsystem diagonalization and truncation

File contains methods which:

    Given a quantum system constructor `constr_qsys` and a dictionary of 
    constructor arguments `constr_args`, determines the minimum number of 
    energy levels to include in simulation to obtain stable bottom n
    eigenenergies of the quantum system.

    Given a dictionary of operators as returned by `constr_qsys`, diagonalizes
    system hamiltonian and re-expresses all dictionary operators in hamiltonian 
    eigenbasis. Then, truncates the diagonalized hamiltonian and other operators
    to a given number of energy levels.

Requirements:
    `constr_qsys` accepts argument `nlev` which determines the number of energy
    levels to consider for the quantum system.

    `constr_qsys` returns a dictionary of type {str:Qobj} of operators defined
    for the quantum system state vector. Dictionary contains entry with label 'H' 
    corresponding to the hamitonian of the system. Dictionaries of this type are
    referred to in this file as datatype `Qsys`.
"""

from typing import TypeAlias
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import warnings

Qobj: TypeAlias = qt.Qobj
Qsys: TypeAlias = dict[str,Qobj]
j: complex = complex(0,1)

def stablize_nlev(constr_qsys:function, 
                  constr_args:dict,
                  stable_levels:int=5,
                  min_nlev:int=5,
                  max_nlev:int=None,
                  n_stable_reps:int=5)->tuple[Qsys, int]:
    assert stable_levels<=min_nlev, \
    'Lower bound of simulated levels `min_nlev`:{} is smaller than desired \
        number of stable levels `stable_levels`:{}'.format(min_nlev, stable_levels)
    #set up loop variables
    nlev:int = min_nlev
    constr_args['nlev'] = min_nlev
    qsys:Qsys = constr_qsys(**constr_args)
    stable_reps:int = 0
    #loop until eigenenergies don't change over `n_stable_reps`
    while stable_reps<n_stable_reps:
        constr_args['nlev'] += 1
        new_qsys:Qsys = constr_qsys(**constr_args)
        energies_changed:bool = compare_energies(qsys, new_qsys, stable_levels)
        if energies_changed:
            stable_reps = 0
            nlev = constr_args['nlev']
            qsys = new_qsys
            if max_nlev is not None and nlev>max_nlev:
                msg = "Maximum number of energy levels (nlev={}) exeeded.".format(max_nlev)
                msg += "\nOptimization will terminate before eigenenergy of bottom {} levels is achieved.".format(stable_levels)
                msg += "\nTo fix, increase max_nlev parameter or set to None."
                warnings.warn(msg)
                nlev = max_nlev
                constr_args['nlev'] = nlev
                qsys = constr_qsys(**constr_args)
                break
    return qsys, nlev

def get_energies_over_nlev(constr_qsys:function,
                           constr_args:dict,
                           track_bottom_n_levels:int=5,
                           min_nlev:int=5,
                           max_nlev:int=30)->list[np.ndarray]:
    assert track_bottom_n_levels<=min_nlev, \
    'Lower bound of simulated levels `min_nlev`:{} is smaller than desired \
        number of tracked levels:{}'.format(min_nlev, track_bottom_n_levels)
    #set up loop variables
    nlev:int = min_nlev
    eigenenergies:list = []
    while nlev<=max_nlev:
        constr_args['nlev'] = nlev
        qsys:Qsys = constr_qsys(**constr_args)
        energies:np.ndarray = qsys['H'].eigenenergies()[:track_bottom_n_levels]
        eigenenergies.append(energies)
        nlev+=1
    return eigenenergies

def plot_energies_over_nlev(constr_qsys:function,
                            constr_args:dict,
                            track_bottom_n_levels:int=5,
                            min_nlev:int=5,
                            max_nlev:int=30)->None:
    params:dict = locals() # make copy of function parameters
    eigenen:list[np.ndarray] = get_energies_over_nlev(**params)
    eigenen:np.ndarray = np.array(eigenen)
    for level in range(track_bottom_n_levels):
        en:np.array = eigenen[:,level]
        lbl = "{}".format(level)
        plt.plot(range(min_nlev,max_nlev+1), en, label=lbl)
    plt.show()
    return None

def compare_energies(qsys_1:Qsys, 
                     qsys_2:Qsys,
                     stable_levels:int) -> bool:
    H1:Qobj = qsys_1['H']
    H2:Qobj = qsys_2['H']
    e1:np.ndarray = H1.eigenenergies()[:stable_levels]
    e2:np.ndarray = H2.eigenenergies()[:stable_levels]
    return (e1 == e2).all()

def diagonalize_Qsys(qsys:Qsys)->tuple[Qsys,np.array[Qobj]]:
    _, eigenbasis = qsys['H'].eigenstates()
    for key, operator in qsys:
        qsys[key] = operator.transform(eigenbasis)
    return qsys, eigenbasis

def truncate_Qsys(qsys:Qsys,
                  nlev:int,
                  eigenbasis=None)->tuple[Qsys,np.array(Qobj)]:
    for key, operator in qsys:
        qsys[key] = operator.extract_states(range(nlev))
    if eigenbasis is not None:
        for idx, eigenstate in enumerate(eigenbasis):
            eigenbasis[idx] = eigenstate.extract_states(range(nlev))
    else:
        _, eigenbasis = qsys['H'].eigenstates()
    return qsys, eigenbasis

def build_optimized_system(constr_qsys:function,
                           constr_args:dict,
                           stable_levels:int=5,
                           min_nlev:int=5,
                           max_nlev:int=None,
                           truncate_to:int=None,
                           n_stable_reps:int=5
                           )->tuple[Qsys,np.array[Qobj]]:
    if truncate_to is None:
        truncate_to = stable_levels
    qsys, _ = stablize_nlev(constr_qsys,
                            constr_args,
                            stable_levels,
                            min_nlev,
                            max_nlev,
                            n_stable_reps)
    qsys, eigenbasis = diagonalize_Qsys(qsys)
    qsys, eigenbasis = truncate_Qsys(qsys, truncate_to, eigenbasis)
    return qsys, eigenbasis


