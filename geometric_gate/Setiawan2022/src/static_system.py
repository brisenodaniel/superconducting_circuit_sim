"""File contains functions for constructing the static hamiltonian of the NISQ system as described
in Setiawan et. al. 2022, in the literature folder. 
"""
import qutip as qt
import subsystems
import yaml
import optimize_subsystem as opt
from typing import TypeAlias, Callable
from subsystems import Subsystem
from composite_systems import CompositeSystem


Qobj: TypeAlias = qt.Qobj
#constants
j: complex = complex(0,1)

def get_params(path:str='../config/circuit_parameters.yaml')->dict:
    with open(path,'r') as stream:
        try:
            ct_params:dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return ct_params

def build_interaction_H(gs:dict[str:float],
                        bare_sys:CompositeSystem)-> Qobj:
    """ Builds the interaction hamiltonian as described in eq (23) of Setiawan et. al. 2022\
    , given the coupling strengths g_ij and bare system.

    Args:
        gs (dict[str:float]): Dictionary with labels (g_AC, g_BC, g_AB), corresponding to coupling strengths\
         between circuit components
        bare_sys (CompositeSystem): System composed of Subsystem objects for fluxonia A,B, and transmon\
         coupler C.

    Returns:
        Qobj: Quantum Object corresponding to interaction term in the composite system hamiltonian as\
         described in eq (23) of Setiawan et. al. 2022.
    """
    H_int:Qobj = qt.Qobj(dims=bare_sys.H.dims)
    for g_lbl in ('g_AC', 'g_BC'):
        # extract coupling strengths and relevant operators
        g:float = gs[g_lbl]
        sys_lbl:str = g_lbl[2]
        n:Qobj = bare_sys.get_raised_op(sys_lbl, 'n')
        a:Qobj = bare_sys.get_raised_op('C', 'a')
        #build interaction hamiltonian
        H_int += g*n*(a.dag()+a)
    n_A:Qobj = bare_sys.get_raised_op('A','n')
    n_B:Qobj = bare_sys.get_raised_op('B','n')
    H_int += gs['g_AB']*n_A*n_B
    return H_int

def build_bare_systems(ct_params:dict[dict[float]],
                       stable_levels:int=5,
                       flux_param_lbls:list[str]=['E_C','E_L','E_J','phi_ext'],
                       transmon_param_lbls:list[str]=['w','U'])->CompositeSystem:
    """Function builds the composite system with hamiltonian H = H_A + H_B + H_C, as described in\
     eq(22) in Setiawan et. al. 2022 (without the terms H_int, H_mod)

    Args:
        param_path (str, optional): Path to yaml file containing circuit parameters. Defaults to '../config/circuit_parameters.yaml'.
        stable_levels (int, optional): Number of desired energy levels per subsystem which should have minimal truncation error. Defaults to 5.
        flux_param_lbls (list[str], optional): List of parameters in the yaml file specified by `param_path` which should \
            be given to fluxonium constructor. Defaults to ['E_C','E_L','E_J','phi_ext'].
        transmon_param_lbls (list[str], optional): List of parameters in the yaml file specified by `param_path` which should \
            be given to transmon constructor.. Defaults to ['w','U'].

    Returns:
        CompositeSystem: The system with subsystems flux_A, flux_B, transmon. The hamiltonian corresponds to the tensor product of these 
        three subsystems' hamiltonians.
    """
    flux_constr:Callable['...',Subsystem] = subsystems.build_fluxonium_operators
    transmon_constr:Callable['...',Subsystem] = subsystems.build_transmon_operators

    flux_A_params:dict[str,float] = {lbl:ct_params['A'][lbl] for lbl in flux_param_lbls}
    flux_B_params:dict[str,float] = {lbl:ct_params['B'][lbl] for lbl in flux_param_lbls}
    transmon_params:dict[str,float] = {lbl:ct_params['C'][lbl] for lbl in transmon_param_lbls}
    
    flux_A = opt.build_optimized_system(flux_constr, flux_A_params, stable_levels)
    flux_B = opt.build_optimized_system(flux_constr, flux_B_params, stable_levels)
    transmon = opt.build_optimized_system(transmon_constr, transmon_params, stable_levels)
    subsystems_dict = {
        'A': flux_A,
        'B': flux_B,
        'C': transmon
    }
    subsystems_idx = {
        'A': 0,
        'B': 1,
        'C': 2
    }
    bare_system = CompositeSystem(subsystems_dict, subsystems_idx)
    return bare_system

def build_static_system(ct_params:dict[dict[float]],
                        stable_levels:int=5,
                        flux_param_lbls:list[str]=['E_C','E_L', 'E_J', 'phi_ext'],
                        transmon_param_lbls:list[str]=['w','U'])->CompositeSystem:
    """Function builds the composite system with hamiltonian H = H_A + H_B + H_C + H_int, as described in\
        eq(22) in Setiawan et. al. 2022 (without the term H_mod)

        Args:
            param_path (str, optional): Path to yaml file containing circuit parameters. Defaults to '../config/circuit_parameters.yaml'.
            stable_levels (int, optional): Number of desired energy levels per subsystem which should have minimal truncation error. Defaults to 5.
            flux_param_lbls (list[str], optional): List of parameters in the yaml file specified by `param_path` which should \
                be given to fluxonium constructor. Defaults to ['E_C','E_L','E_J','phi_ext'].
            transmon_param_lbls (list[str], optional): List of parameters in the yaml file specified by `param_path` which should \
                be given to transmon constructor.. Defaults to ['w','U'].

        Returns:
            CompositeSystem: The system with subsystems flux_A, flux_B, and transmon. Hamiltonian includes interaction term \
            given by equation (23)
    """
    interaction_params:dict[str,float] = ct_params['interaction']
    bare_system:CompositeSystem = build_bare_systems(ct_params,
                                     stable_levels,
                                     flux_param_lbls,
                                     transmon_param_lbls)
    H_int:Qobj = build_interaction_H(interaction_params,bare_system)
    H_static:CompositeSystem = bare_system.plus_interaction_term(H_int)

    return H_static
    




