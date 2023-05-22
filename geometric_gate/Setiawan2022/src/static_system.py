"""File contains functions for constructing the static hamiltonian of the NISQ system as described
in Setiawan et. al. 2022, in the literature folder.
"""
import qutip as qt
import numpy as np
import subsystems
import yaml
import optimize_subsystem as opt
from typing import TypeAlias, Callable
from subsystems import Subsystem
from composite_systems import CompositeSystem

Qobj: TypeAlias = qt.Qobj
# constants
j: complex = complex(0, 1)


def get_params(path: str = "../config/circuit_parameters.yaml") -> dict:
    with open(path, "r") as stream:
        try:
            ct_params: dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return ct_params


def build_interaction_H(gs: dict[str:float], bare_sys: CompositeSystem) -> Qobj:
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
    H_int: Qobj | int = 0
    for g_lbl in ("g_AC", "g_BC"):
        # extract coupling strengths and relevant operators
        g: float = (
            np.pi * 2 * gs[g_lbl]
        )  # convert coupling strengths to radian frequency
        sys_lbl: str = g_lbl[2]
        n_j: Qobj = bare_sys.get_raised_op(sys_lbl, "n")
        n_C: Qobj = bare_sys.get_raised_op("C", ["a"], lambda a: a.dag() + a)
        # build interaction hamiltonian
        H_int += g * n_j * n_C
    n_A: Qobj = bare_sys.get_raised_op("A", "n")
    n_B: Qobj = bare_sys.get_raised_op("B", "n")
    H_int += gs["g_AB"] * n_A * n_B
    return H_int


def build_bare_system(
    ct_params: dict[str, dict[str, float]],
    stable_nlev: int = 5,
    trunc: int | None = None,
    min_nlev: int | None = None,
    stabilize: bool = True,
    nlev: int = 30,
    flux_param_lbls: list[str] = ["E_C", "E_L", "E_J", "phi_ext"],
    transmon_param_lbls: list[str] = ["w", "U"],
) -> CompositeSystem:
    """Function builds the composite system with hamiltonian H = H_A + H_B + H_C, as described in\
     eq(22) in Setiawan et. al. 2022 (without the terms H_int, H_mod)

    Args:
    ct_params (dict[str,dict[str,float]]): Dictionary containing circuit parameters. Must contain keys\
        'A', 'B', 'C', which correspond to fluxonia A,B, and transmon coupler C. Values must be dictionaries\
        with circuit parameter names as keys and numerical parameter values.
    stable_nlev (int, optional): Number of desired energy levels per subsystem which should have stable eigenenergy\
        values as more levels are added to the simulation in the QHO basis. If `stabilize=False`, this parameter\
            will be ignored. Defaults to 5.
    trunc (int | None, optional): Number of energy levels to keep in each subsystem after transforming to the hamiltonian\
        eigenbasis and truncating. Must be less than min_nlev. If not provided, will be set to the value of stable_nlev.
    min_nlev (int | None, optional): Minimum number of energy levels to model in each subsystem after stabilization.\
        Must be greater than `trunc`. If None, will be set to the value of `stable_levels`\
    stabilize (bool, optional): If True, subsystems will undergo a stabilization algorithm until the bottom `stable_nlev`\
        eigenstates in the QHO basis do not vary their eigenenergies as more levels are added to the simulation in the QHO\
            basis. Defaults to True.
    nlev (int | None, optional): If `stabilize == False`, subsystems will be initialized in the QHO basis with `nlev` energy\
        levels. If `stabilize==True`, parmeter will be ignored. Defaults to 30.
    flux_param_lbls (list[str], optional): List of parameter names in ct_params under keys 'A' and 'B' which should be given\
        to fluxonium constructor. Defaults to ['E_C','E_L','E_J','phi_ext']..
    transmon_param_lbls (list[str], optional): List of parameter names in ct_params under key 'C' which should be given\
        to transmon constructor. Defaults to ['w','U'].

    Returns:
        CompositeSystem: The system with subsystems flux_A, flux_B, transmon. The hamiltonian corresponds to the tensor product of these
        three subsystems' hamiltonians.
    """
    flux_constr: Callable["...", Subsystem] = subsystems.build_fluxonium_operators
    transmon_constr: Callable["...", Subsystem] = subsystems.build_transmon_operators

    flux_A_params: dict[str, float] = {
        lbl: ct_params["A"][lbl] for lbl in flux_param_lbls
    }
    flux_B_params: dict[str, float] = {
        lbl: ct_params["B"][lbl] for lbl in flux_param_lbls
    }
    transmon_params: dict[str, float] = {
        lbl: ct_params["C"][lbl] for lbl in transmon_param_lbls
    }
    opt_sys_params: dict = {
        "constr_qsys": None,
        "constr_args": None,
        "stable_nlev": stable_nlev,
        "truncate_to": trunc,
        "min_nlev": min_nlev,
        "stabilize": stabilize,
        "nlev": nlev,
    }
    opt_sys_params["constr_qsys"] = flux_constr
    opt_sys_params["constr_args"] = flux_A_params
    flux_A = opt.build_optimized_system(**opt_sys_params)
    opt_sys_params["constr_args"] = flux_B_params
    flux_B = opt.build_optimized_system(**opt_sys_params)
    opt_sys_params["constr_qsys"] = transmon_constr
    opt_sys_params["constr_args"] = transmon_params
    transmon = opt.build_optimized_system(**opt_sys_params)
    subsystems_dict = {"A": flux_A, "B": flux_B, "C": transmon}
    subsystems_idx = {"A": 0, "B": 1, "C": 2}
    bare_system = CompositeSystem(subsystems_dict, subsystems_idx)
    return bare_system


def build_static_system(
    ct_params: dict[str, dict[str, float]],
    stable_nlev: int = 5,
    trunc: int | None = None,
    stabilize: bool = True,
    min_nlev: int | None = None,
    nlev: int = 30,
    flux_param_lbls: list[str] = ["E_C", "E_L", "E_J", "phi_ext"],
    transmon_param_lbls: list[str] = ["w", "U"],
) -> CompositeSystem:
    """Function builds the composite system with hamiltonian H = H_A + H_B + H_C + H_int, as described in\
        eq(22) in Setiawan et. al. 2022 (without the term H_mod)
    Args:
        ct_params (dict[str,dict[str,float]]): Dictionary containing circuit parameters. Must contain keys\
            'A', 'B', 'C', which correspond to fluxonia A,B, and transmon coupler C. Values must be dictionaries\
            with circuit parameter names as keys and numerical parameter values.
        stable_nlev (int, optional): Number of desired energy levels per subsystem which should have stable eigenenergy\
            values as more levels are added to the simulation in the QHO basis. If `stabilize=False`, this parameter\
                will be ignored. Defaults to 5.
        trunc (int | None, optional): Number of energy levels to keep in each subsystem after transforming to the hamiltonian\
            eigenbasis and truncating. If not provided, will be set to the value of stable_nlev.
        stabilize (bool, optional): If True, subsystems will undergo a stabilization algorithm until the bottom `stable_nlev`\
            eigenstates in the QHO basis do not vary their eigenenergies as more levels are added to the simulation in the QHO\
                basis. Defaults to True.
        nlev (int | None, optional): If `stabilize == False`, subsystems will be initialized in the QHO basis with `nlev` energy\
            levels. If `stabilize==True`, parmeter will be ignored. Defaults to 30.
        flux_param_lbls (list[str], optional): List of parameter names in ct_params under keys 'A' and 'B' which should be given\
            to fluxonium constructor. Defaults to ['E_C','E_L','E_J','phi_ext']..
        transmon_param_lbls (list[str], optional): List of parameter names in ct_params under key 'C' which should be given\
            to transmon constructor. Defaults to ['w','U'].

    Returns:
        CompositeSystem: The system with subsystems flux_A, flux_B, and transmon. Hamiltonian includes interaction term \
                    given by equation (23)    """

    bare_sys_params: dict = locals()  # make copy of parameters
    bare_system: CompositeSystem = build_bare_system(
        **bare_sys_params
    )  # build_bare_systems takes same parameters as current function
    interaction_params: dict[str, float] = ct_params["interaction"]
    H_int: Qobj = build_interaction_H(interaction_params, bare_system)
    static_system: CompositeSystem = bare_system.plus_interaction_term(H_int)
    return static_system
