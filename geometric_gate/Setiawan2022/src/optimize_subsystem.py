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
   

Requirements:
    `constr_qsys` accepts argument `nlev` which determines the number of energy
    levels to consider for the quantum system.

    `constr_qsys` returns a Subsystem with hamiltonian operator defined. Subystem\
    class defined in subsystems module
"""

from typing import TypeAlias, Callable
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from subsystems import Subsystem

Qobj: TypeAlias = qt.Qobj
Qsys: TypeAlias = dict[str,Qobj]

j: complex = complex(0,1)

def stabilize_nlev(constr_qsys:Callable['...', Subsystem], 
                  constr_args:dict,
                  stable_levels:int=3,
                  tol:float=1e-9,
                  min_nlev:int=None,
                  max_nlev:int=None,
                  n_stable_reps:int=5)->tuple[Subsystem, int]:
    """Determines the minimum number of energy levels to consider to simulate\
    a given quantum system such that the relevant energy levels do not change\
    thier eigenenergies as the Hamiltonian considers higher energy states.

    Args:
        constr_qsys (function): Constructor which returns dictionary of quantum operators (Qsys)\
         corresponding to operators acting on a quantum system, including the Hamiltonian which \
         must have key 'H'. Must accept a parameter `nlev` which determines the number of energy \
         levels modeled in Qsys.
        constr_args (dict): Named arguments to constr_qsys. Need not include `nlev`.
        stable_levels (int, optional): Determines which energy levels are stabilized. 
         For a given value n, [0,1,2,...,n-1] levels will be stabilized. Defaults\
         to 5.
        tol (float, optional): Numerical error tolerance. If two eigenenergies e1 and e2 are such that |e1-e2|<=tol,\
         then e1,e2 will be considered equal.
        min_nlev (int, optional): Minimum number of simulated levels to consider. Must be\
         greater than `stable_levels`. If not provided, will be set to the value of `stable levels. Defaults to None.
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
    if min_nlev is None:
        min_nlev = stable_levels
    assert stable_levels <= min_nlev, \
    'Lower bound of simulated levels `min_nlev`:{} is smaller than desired \
        number of stable levels `stable_levels`:{}'.format(min_nlev, stable_levels)
    assert max_nlev is None  or min_nlev<max_nlev,\
    'Upper bound of simulated levels `max_nlev`:{} is smaller than lower bound\
        min_nlev:{}'.format(max_nlev,min_nlev)
    #set up loop variables
    nlev:int = min_nlev
    constr_args['nlev'] = min_nlev
    qsys:Qsys = constr_qsys(**constr_args)
    stable_reps:int = 0
    #loop until eigenenergies don't change over `n_stable_reps`
    while stable_reps<n_stable_reps:
        constr_args['nlev'] += 1
        new_qsys:Qsys = constr_qsys(**constr_args)
        energies_changed:bool = not has_equal_energies(qsys,
                                                       new_qsys, 
                                                       stable_levels,
                                                       tol)
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
        else:
            stable_reps += 1
    return qsys, nlev

def get_energies_over_nlev(constr_qsys:Callable['...',Qsys],
                           constr_args:dict,
                           track_bottom_n_levels:int=3,
                           min_nlev:int=None,
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
         greater than `stable_levels`. If not provided, will be set to the value of `track_bottom_n_levels`.
        max_nlev (int, optional): Maximum number of simulated levels to consider. Defaults to 30

    Returns:
        list[np.ndarray]: 2d iterable with indices [nlev-1][eigenen], where nlev is the number of modeled energy\
         levels and eigenen is the list of eigenenergies for hamiltonian eigenstates as set by `track_bottom_n_levels`
    """
    if min_nlev is None:
        min_nlev = track_bottom_n_levels
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

def plot_energies_over_nlev(constr_qsys:Callable['...', Qsys],
                            constr_args:dict,
                            track_bottom_n_levels:int=3,
                            min_nlev:int=None,
                            max_nlev:int=30,
                            print_to:str=None)->None:
    """Function plots eigenenergies of a system's hamiltonian as the number\
    of energy levels is varied between [min_nlev, max_nlev].
    Args:
        constr_qsys (function): Constructor which returns a `Qsys` (defined in file docstring).\
         Must accept a parameter `nlev` which determines the number of energy levels modeled in `Qsys`.
        constr_args (dict): Named arguments to `constr_qsys`. Need not include `nlev`.
        track_bottom_n_levels (int, optional): For a given value n, function will plot the eigenenergies of the bottom\
             n energy levels. Defaults to 5.
        min_nlev (int, optional): Minimum number of simulated levels to consider. Must be\
         greater than `stable_levels`. If not provided, will be set to the value of `track_bottom_n_levels`.
        max_nlev (int, optional): Maximum number of simulated levels to consider. Defaults to 30\
        print_to (str, optional): File name of eigenenergy plot. File will be placed in `../plots` directory.\
         If not provided, function will print plot to standard out.

    Returns:
       None
    """
    if min_nlev is None:
        min_nlev = track_bottom_n_levels
    params:dict = locals() # make copy of function parameters
    del params['print_to']
    eigenen:list[np.ndarray] = get_energies_over_nlev(**params)
    eigenen:np.ndarray = np.array(eigenen)
    for level in range(track_bottom_n_levels):
        en:np.array = eigenen[:,level]
        lbl = "{}".format(level)
        plt.plot(range(min_nlev,max_nlev+1), en, label=lbl)
        plt.legend()
        plt.title('Eigenenergies of Bottom {} Levels Vs Tot.Simulated Levels'\
                  .format(track_bottom_n_levels))
        plt.ylabel('Eigenenergy')
        plt.xlabel('Total Simulated Levels')
    if print_to is not None:
        if '../plots/' not in print_to:
            print_to = '../plots/'+ print_to
        plt.savefig(print_to)
    else:
        plt.show()
    return None

def is_diag(op:Qobj, tol:float=1e-9)->bool:
    """Function determines if Qobj is diagonal

    Args:
        op (Qobj): Quantum operator
        tol (int, optional): Numerical error tolerance. If \
         a matrix element has value with magnitude less than tol,\
         element will be considered 0.

    Returns:
        bool: True if `op` is diagonal, false otherwise
    """
    diags = op.diag()
    diag_op = qt.qdiags(diags,0)
    return qt.isequal(diag_op, op, tol)

def has_equal_energies(qsys_1:Qsys, 
                       qsys_2:Qsys,
                       stable_levels:int=None,
                       tol:float=1e-9) -> bool:
    """Function determines if two Qsys have hamiltonians with\
      equal eigenenergies for bottom `stable_levels` eigenstates.

    Args:
        qsys_1 (Qsys): First system to compare. Type `Qsys` defined in\
         file docstring.
        qsys_2 (Qsys): Second system to compare.
        stable_levels (int): For a given value n, function will determine if \
         bottom n eigenstates of the hamiltonians in (qsys_1, qsys_2) have identical
         eigenenergies.
        tol (float, optional): Numerical error tolerance. If two eigenenergies e1 and e2 are such that |e1-e2|<=tol,\
         then e1,e2 will be considered equal.
    Returns:
        bool: True if first two parameters have identical eigenenergies for bottom `stable_levels`\
         eigenstates.
    """
    if stable_levels is None:
        stable_levels = qsys_1['H'].dims[0][0]
    H1:Qobj = qsys_1['H']
    H2:Qobj = qsys_2['H']
    e1:np.ndarray = H1.eigenenergies()[:stable_levels]
    e2:np.ndarray = H2.eigenenergies()[:stable_levels]
    if tol==0:
        return all(e1==e2)
    else:
        return all(abs(e1-e2)<=tol)

def diagonalize_Qsys(qsys:Subsystem)->Subsystem:
    """Function finds the eigenbasis for the hamiltonian in qsys, and \
    transforms all operators in qsys to that eigenbasis

    Args:
        qsys (Subsystem): Quantum system with operators to be transformed to its hamiltonian's\
         eigenbasis.

    Returns:
        Subsystem: Subsystem identical to `qsys` but in hamiltonian eigenbasis.
    """
    if is_diag(qsys['H']):
        return qsys
    _, eigenbasis = qsys['H'].eigenstates()
    diag_sys = qsys.transform(eigenbasis)

    return diag_sys

def truncate_Qsys(qsys:Subsystem,
                  truncate_to:int)->Subsystem:
    """Function truncates all operators in qsys to an operator acting on a state with `nlev`
    energy levels. Truncates basis vectors in `eigenbasis` if provided.

    Args:
        qsys (Subsystem): Quantum system with operators to be truncated.
        truncate_to (int): Number of energy levels to truncate to.
        eigenbasis (np.ndarray[Qobj], optional): If provided, eigenbasis of qsys relative to some \
            basis. If not provided, will be set to the eigenstates of the hamiltonian in qsys.\
            Defaults to None.

    Returns:
        tuple[Subsystem,np.ndarray[Qobj]]: (sys, basis) where sys is `qsys` with all Qobj truncated to `truncate_to`\
         energy levels, and basis is `eigenbasis` identically truncated.
    """
    return qsys.truncate(truncate_to)
    # # check dimensions of qsys are greater than or equal to desired truncation length
    # for key, operator in qsys.items():
    #     nlev = operator.full().shape[0]
    #     assert truncate_to<=nlev, \
    #     f'Desired truncation length {truncate_to} is greater than number of energy levels in operator'\
    #     +f' {key}:{nlev}.'
    # # check dimensions of the basis are greater than or equal to desired truncation length
    # assert truncate_to<=len(eigenbasis),\
    # f'Desired truncation length {truncate_to} indicates a vector space of higher dimension than'\
    # + f' number of basis vectors in `eigenbasis`:{len(eigenbasis)}'
    # for eigenstate in eigenbasis:
    #     nlev = eigenstate.full().shape[0]
    #     assert truncate_to<=nlev,\
    #     f'Desired truncation length {truncate_to} is greater than length of basis vectors {nlev}'
    # # truncate operators
    # for key, operator in qsys.items():
    #     qsys[key] = operator.extract_states(range(truncate_to))
    # # truncate eigenbasis
    # if eigenbasis is not None:
    #     eigenbasis = eigenbasis[:truncate_to]
    #     for idx, eigenstate in enumerate(eigenbasis):
    #         eigenbasis[idx] = eigenstate.extract_states(range(truncate_to))
    # else:
    #     _, eigenbasis = qsys['H'].eigenstates()
    # return qsys, eigenbasis

def build_optimized_system(constr_qsys:Callable['...', Subsystem],
                           constr_args:dict,
                           stable_levels:int=3,
                           tol:float=1e-9,
                           min_nlev:int=None,
                           max_nlev:int=None,
                           truncate_to:int=None,
                           n_stable_reps:int=5
                           )->tuple[Subsystem,np.ndarray[Qobj]]:
    """Function first builds the quantum system defined by `constr_qsys` with enough energy levels\
    to model dynamics of the bottom `stable_levels` eigenstates (ordered by increasing eigenenergy)\
    with minimal truncation error. Then, the stable system model is transformed into it's hamiltonian's\
    eigenbasis, and truncated to `truncate_to` energy levels.

    Args:
        constr_qsys (function): Constructor which returns dictionary of quantum operators (Subsystem)\
         corresponding to operators acting on a quantum system, including the Hamiltonian which \
         must have key 'H'. Must accept a parameter `nlev` which determines the number of energy \
         levels modeled in Subsystem.
        constr_args (dict): Named arguments to constr_qsys. Need not include `nlev`.
        stable_levels (int, optional): Determines which energy levels are stabilized. 
         For a given value n, [0,1,2,...,n-1] levels will be stabilized. Defaults\
         to 5.    
         min_nlev (int, optional): Minimum number of simulated levels to consider. Must be\
         greater than or equal to `stable_levels`. If not provided, will be set to the value of `stable levels`.
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
        tuple[Subsystem,np.array[Qobj]]: tuple(sys, basis) where sys is the quantum system returned by `constr_sys`\
         transformed to it's hamiltonian's eigenbasis and truncated to `truncate_to` energy levels with minimal\
         truncation error; and `basis` is the truncated hamiltonian eigenbasis. `basis` is given relative to the\
         basis in which `constr_sys` defines the system operators.
    """
    if truncate_to is None:
        truncate_to = stable_levels
    if min_nlev is None:
        min_nlev = stable_levels
    assert truncate_to<=min_nlev,\
    f'Parameter `truncate_to` must have value less than or equal to `min_nlev`.'+\
    f' `truncate_to` has value {truncate_to}, `min_nlev` has {min_nlev}.' +\
    f' If you did not provide `min_nlev` as a parameter, it was set to the value of parameter `stable_levels`.'
    qsys, _ = stabilize_nlev(constr_qsys,
                             constr_args,
                             stable_levels,
                             tol,
                             min_nlev,
                             max_nlev,
                             n_stable_reps)
    qsys = diagonalize_Qsys(qsys)
    qsys = qsys.truncate(truncate_to)
   # qsys, eigenbasis = truncate_Qsys(qsys, truncate_to, eigenbasis)
    return qsys