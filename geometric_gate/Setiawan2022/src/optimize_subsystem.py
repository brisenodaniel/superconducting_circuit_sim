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

Custom Datatypes:
    Qobj: Alias for qt.Qobj
    Qsys: Dictionary with values of type Qobj corresponding to operators acting on
     some quantum system. Must include key 'H' with value corresponding to the system
     hamiltonian.

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
    """Determines the minimum number of energy levels to consider to simulate\
    a given quantum system such that the relevant energy levels do not change\
    thier eigenenergies as the Hamiltonian considers higher energy states.

    Args:
        constr_qsys (function): Constructor which returns dictionary of quantum operators (Qsys)\
         corresponding to operators acting on a quantum system, including the Hamiltonian which \
         must have key 'H'. Must accept a parameter `nlev` which determines the number of energy \
         levels modeled in Qsys.
        constr_args (dict): Named arguments to constr_qsys. Need not include `nlev`.
        stable_levels (int, optional): Determines which energy levels are stablized. 
         For a given value n, [0,1,2,...,n-1] levels will be stablized. Defaults\
         to 5.
        min_nlev (int, optional): Minimum number of simulated levels to consider. Must be\
         greater than `stable_levels`. Defaults to 5.
        max_nlev (int, optional): Maximum number of simulated levels to consider. If not \
         provided, function will run until stabilization is achieved. Defaults to None.
        n_stable_reps (int, optional): Number of additional energy states of the Hamiltionian \
         to consider when determining stability of the eigenstates. If function detemines that \
         the model is stable with n energy levels, then the hamiltonians with [n+1,...,n+n_stable_reps] \
         yielded the same eigenenergies (for eigenstates indicated by `stable_levels`) as the hamiltonian \
         with n eigenstates. Defaults to 5.

    Returns:
        tuple[Qsys, int]: (qsys, nlev), where nlev is the minimum number of energy levels modeled to achieve \
         stability of the eigenenergies, and qsys is the return value of `constr_qsys` with nlev energy levels.
    """
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
                           track_bottom_n_levels:int=None,
                           min_nlev:int=5,
                           max_nlev:int=30)->list[np.ndarray]:
    """Function tracks eigenenergies of a system's hamiltonian as the number\
    of energy levels is varied between [min_nlev,max_nlev].

    Args:
        constr_qsys (function): Constructor which returns a `Qsys` (defined in file docstring).\
         Must accept a parameter `nlev` which determines the number of energy levels modeled in `Qsys`.
        constr_args (dict): Named arguments to `constr_qsys`. Need not include `nlev`.
        track_bottom_n_levels (int, optional): For a given value n, function will track the eigenenergies of the bottom\
         n energy levels. If not provided, all eigenenergies of the given hamiltonian will be returned. Defaults to None.
        min_nlev (int, optional): Minimum number of simulated levels to consider. Must be\
         greater than `stable_levels`. Defaults to 5.
        max_nlev (int, optional): Maximum number of simulated levels to consider. Defaults to 30

    Returns:
        list[np.ndarray]: 2d iterable with indices [nlev-1][eigenen], where nlev is the number of modeled energy\
         levels and eigenen is the list of eigenenergies for hamiltonian eigenstates as set by `track_bottom_n_levels`
    """
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
    """Function plots eigenenergies of a system's hamiltonian as the number\
    of energy levels is varied between [min_nlev, max_nlev].
    Args:
        constr_qsys (function): Constructor which returns a `Qsys` (defined in file docstring).\
         Must accept a parameter `nlev` which determines the number of energy levels modeled in `Qsys`.
        constr_args (dict): Named arguments to `constr_qsys`. Need not include `nlev`.
        track_bottom_n_levels (int, optional): For a given value n, function will plot the eigenenergies of the bottom\
             n energy levels. Defaults to 5.
        min_nlev (int, optional): Minimum number of simulated levels to consider. Must be\
         greater than `stable_levels`. Defaults to 5.
        max_nlev (int, optional): Maximum number of simulated levels to consider. Defaults to 30

    Returns:
       None
    """
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
    """Function determines if two Qsys have hamiltonians with\
      equal eigenenergies for bottom `stable_levels` eigenstates.

    Args:
        qsys_1 (Qsys): First system to compare. Type `Qsys` defined in\
         file docstring.
        qsys_2 (Qsys): Second system to compare.
        stable_levels (int): For a given value n, function will determine if \
         bottom n eigenstates of the hamiltonians in (qsys_1, qsys_2) have identical
         eigenenergies.

    Returns:
        bool: True if first two parameters have identical eigenenergies for bottom `stable_levels`\
         eigenstates.
    """
    H1:Qobj = qsys_1['H']
    H2:Qobj = qsys_2['H']
    e1:np.ndarray = H1.eigenenergies()[:stable_levels]
    e2:np.ndarray = H2.eigenenergies()[:stable_levels]
    return (e1 == e2).all()

def diagonalize_Qsys(qsys:Qsys)->tuple[Qsys,np.ndarray[Qobj]]:
    """Function finds the eigenbasis for the hamiltonian in qsys, and \
    transforms all operators in qsys to that eigenbasis

    Args:
        qsys (Qsys): Quantum system with operators to be transformed to its hamiltonian's\
         eigenbasis.

    Returns:
        tuple[Qsys,np.ndarray[Qobj]]: (sys, basis) where sys is qsys transformed to \
         it's hamiltonian's eigenbasis and basis is the eigenbasis.
    """
    _, eigenbasis = qsys['H'].eigenstates()
    for key, operator in qsys:
        qsys[key] = operator.transform(eigenbasis)
    return qsys, eigenbasis

def truncate_Qsys(qsys:Qsys,
                  nlev:int,
                  eigenbasis:np.ndarray[Qobj]=None)->tuple[Qsys,np.ndarray[Qobj]]:
    """Function truncates all operators in qsys to an operator acting on a state with `nlev`
    energy levels. Truncates basis vectors in `eigenbasis` if provided.

    Args:
        qsys (Qsys): Quantum system with operators to be truncated.
        nlev (int): Number of energy levels to truncate to.
        eigenbasis (np.ndarray[Qobj], optional): If provided, eigenbasis of qsys relative to some \
            basis. If not provided, will be set to the eigenstates of the hamiltonian in qsys.\
            Defaults to None.

    Returns:
        tuple[Qsys,np.ndarray[Qobj]]: (sys, basis) where sys is `qsys` with all Qobj truncated to `nlev`\
         energy levels, and basis is `eigenbasis` identically truncated.
    """
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
    """Function first builds the quantum system defined by `constr_qsys` with enough energy levels\
    to model dynamics of the bottom `stable_levels` eigenstates (ordered by increasing eigenenergy)\
    with minimal truncation error. Then, the stable system model is transformed into it's hamiltonian's\
    eigenbasis, and truncated to `truncate_to` energy levels.

    Args:
        constr_qsys (function): Constructor which returns dictionary of quantum operators (Qsys)\
         corresponding to operators acting on a quantum system, including the Hamiltonian which \
         must have key 'H'. Must accept a parameter `nlev` which determines the number of energy \
         levels modeled in Qsys.
        constr_args (dict): Named arguments to constr_qsys. Need not include `nlev`.
        stable_levels (int, optional): Determines which energy levels are stablized. 
         For a given value n, [0,1,2,...,n-1] levels will be stablized. Defaults\
         to 5.    
         min_nlev (int, optional): Minimum number of simulated levels to consider. Must be\
         greater than `stable_levels`. Defaults to 5.
        max_nlev (int, optional): Maximum number of simulated levels to consider. If not \
         provided, function will run until stabilization is achieved. Defaults to None.
        truncate_to (int, optional): Number of energy levels to truncate to after all operators\
            have been transformed to hamiltonian eigenbasis (hamiltonian given by `constr_qsys`).\
            If not provided, is set to the value of `stable_levels`. Defaults to None.
        n_stable_reps (int, optional): Number of additional energy states of the Hamiltionian \
         to consider when determining stability of the eigenstates. If function detemines that \
         the model is stable with n energy levels, then the hamiltonians with [n+1,...,n+n_stable_reps] \
         yielded the same eigenenergies (for eigenstates indicated by `stable_levels`) as the hamiltonian \
         with n eigenstates. Defaults to 5.

    Returns:
        tuple[Qsys,np.array[Qobj]]: tuple(sys, basis) where sys is the quantum system returned by `constr_sys`\
         transformed to it's hamiltonian's eigenbasis and truncated to `truncate_to` energy levels with minimal\
         truncation error; and `basis` is the truncated hamiltonian eigenbasis. `basis` is given relative to the\
         basis in which `constr_sys` defines the system operators.
    """
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