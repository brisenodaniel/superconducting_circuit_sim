#!/usr/bin/env python
from __future__ import annotations
import qutip as qt
import numpy as np
import file_io
import hashing
import simulation_setup as sim_setup
import multiprocessing
from math import sqrt
from warnings import warn
from simulation_setup import PulseDict, CircuitParams
from simulation_setup import PulseProfile
from simulation_setup import Config
from dataclasses import dataclass
from static_system import build_bare_system
from composite_systems import CompositeSystem, Qobj
from state_labeling import get_dressed_comp_states, CompCoord, CompTensor
from typing import TypeAlias, Callable
from functools import reduce

Qobj: TypeAlias = qt.Qobj
GateDict: TypeAlias = dict[str, str |
                           PulseDict | CircuitParams | bool | list[str]]


@dataclass
class GateProfile:
    """Simple container class for gate data."""

    def __init__(
        self,
        name: str,
        pulse_profile: PulseProfile,
        gate_unitary: Qobj | None = None,
        gate_2q_unitary: Qobj | None = None,
        two_q_unitary: Qobj | None = None,
        circuit: CompositeSystem | None = None,
        circuit_config: CircuitParams | None = None,
        trajectories: dict[str, qt.solver.Result] = {},
        fidelity: float | None = None,
    ) -> GateProfile:
        self.name: str = name
        self.pulse: np.ndarray[float] = pulse_profile.pulse
        self.pulse_profile: PulseProfile = pulse_profile
        self.target_unitary: str = pulse_profile.pulse_config.target_unitary
        self.gate_unitary: Qobj | None = gate_unitary
        self.two_q_unitary: Qobj | None = two_q_unitary
        if circuit is None:
            self.circuit: Qobj = pulse_profile.circuit
        else:
            self.circuit: Qobj = circuit
        if circuit_config is None:
            self.circuit_config: CircuitParams = (
                pulse_profile.pulse_config.circuit_config
            )
        else:
            self.circuit_config: CircuitParams = circuit_config
        self.trajectories: dict[str, qt.solver.Result] = trajectories
        self.fidelity: float = fidelity

        # send warning if initialized with circuit but no circuit config
        if self.circuit is not None and self.circuit_config is None:
            msg = f"Circuit intitialized for gate {self.name} \
            but no circuit configuration dict given, caching may fail."
            warn(msg)

    def as_dict(self) -> GateDict:
        desc: GateDict = {}  # init descriptor dictionary
        desc["name"] = self.name
        desc["pulse_config"] = self.pulse_profile.as_dict()
        desc["unitary_cached"] = self.gate_unitary is not None
        desc["two_q_unitary_cached"] = self.two_q_unitary is not None
        desc["circuit_config"] = self.circuit_config
        desc["trajectories"] = [
            init_state for init_state in self.trajectories.keys()]
        desc["fidelity"] = self.fidelity
        return desc

    def as_noNone_dict(self) -> GateDict:
        return file_io.tidy_cache(self.as_dict())

    def __hash__(self) -> int:
        return hashing.hash_dict(self.as_dict())


def drive_op(circuit: CompositeSystem) -> Qobj:
    n_C: Qobj = circuit.get_raised_op("C", ["a"], lambda a: a.dag() * a)
    return n_C


def run_sim_from_configs(
    drive_op: Callable["...", Qobj] = drive_op,
    n_comp_states: int = 2,
    cache_gates: bool = True,
    use_pulse_cache: bool = True,
    use_gate_cache: bool = True,
    multiprocess: bool = False,
) -> None:
    config: Config
    pulse_dict: dict[str, PulseProfile]
    config, pulse_dict = sim_setup.setup_sim_from_configs(
        use_pulse_cache=use_pulse_cache, multiprocess=multiprocess
    )
    if multiprocess:
        run_sim_multiprocess(
            config, pulse_dict, drive_op, n_comp_states, cache_gates, use_gate_cache
        )
    else:
        run_sim(
            config,
            pulse_dict,
            drive_op,
            n_comp_states,
            cache_gates,
            use_gate_cache=use_gate_cache,
        )


def run_sim(
    config: Config,
    pulse_profiles: dict[str, PulseProfile],
    drive_op: Callable["...", Qobj] = drive_op,
    n_comp_states: int = 2,
    cache_gates: bool = True,
    use_gate_cache: bool = True,
) -> None:
    circuit_configs: dict[str, dict[str, dict[str, float]]] = config.ct_params
    for ct_name, ct_config in circuit_configs.items():
        for pulse_name, profile in pulse_profiles.items():
            gate_name: str = f"{pulse_name}-{ct_name}"
            gate_profile = profile_gate(
                gate_name,
                profile,
                drive_op=drive_op,
                circuit_config=ct_config,
                n_comp_states=n_comp_states,
                cache_gate=True,
                use_gate_cache=use_gate_cache,
            )
            del gate_profile


def run_sim_multiprocess(
    config: Config,
    pulse_profiles: dict[str, PulseProfile],
    drive_op: Callable["...", Qobj] = drive_op,
    n_comp_states: int = 2,
    cache_gates: bool = True,
    use_gate_cache: bool = True,
) -> dict[str, GateProfile]:
    n_cpu = multiprocessing.cpu_count()
    n_workers = n_cpu - 1
    circuit_configs: dict[str, dict[str, dict[str, float]]] = config.ct_params
    param_list = list(
        [
            (
                f"{pulse_name}-{ct_name}",
                profile,
                drive_op,
                None,
                None,
                ct_config,
                n_comp_states,
                False,
                use_gate_cache,
                True,
            )
            for ct_name, ct_config in circuit_configs.items()
            for pulse_name, profile in pulse_profiles.items()
        ]
    )
    with multiprocessing.Pool(n_workers) as p:
        p.starmap(profile_gate, param_list)
        file_io.pool_desc()


def profile_gate(
    name,
    pulse_profile,
    drive_op: Qobj | Callable["...", Qobj] = drive_op,
    empty_gate_profile: GateProfile | None = None,
    circuit: CompositeSystem | None = None,
    circuit_config: CircuitParams | None = None,
    n_comp_states: int = 2,
    cache_gate: bool = True,
    use_gate_cache: bool = True,
    multiprocess: bool = True,
):
    gate_profile: GateProfile = assemble_empty_profile(
        name,
        pulse_profile,
        circuit=circuit,
        circuit_config=circuit_config,
        gate_profile=empty_gate_profile,
    )
    # check if gate has been pre-computed
    if use_gate_cache:
        if file_io.gate_cached(gate_profile):
            return file_io.load_gate(gate_profile.name)
    # else, compute gate profile
    comp_states: CompTensor
    comp_coord: CompCoord
    comp_states, comp_coord = get_dressed_comp_states(
        gate_profile.circuit, n_comp_states
    )
    gate_profile.trajectories = get_trajectories(
        pulse_profile=pulse_profile,
        circuit=gate_profile.circuit,
        comp_states=comp_states,
        drive_op=drive_op,
    )

    gate_profile.gate_unitary, gate_profile.two_q_unitary = compute_unitary(
        gate_profile.trajectories, comp_states, comp_coord, gate_profile.circuit.H.dims
    )
    ideal_unitary = file_io.load_unitary(
        pulse_profile.pulse_config.target_unitary)
    gate_profile.fidelity = compute_fidelity(
        ideal_unitary, gate_profile.two_q_unitary, gate_profile.circuit.H, comp_coord
    )
    if cache_gate:
        file_io.cache_gate(gate_profile, multiprocess)
    if multiprocess:
        del gate_profile
        del pulse_profile
        del circuit
        del circuit_config
    else:
        return gate_profile


def compute_fidelity(
    U_ideal: Qobj, U_g: Qobj, H: Qobj | None = None, comp_coord: CompCoord | None = None
) -> float:
    # implements eq (14)
    s0 = qt.qeye(2)
    sigmas: list[Qobj] = [qt.sigmaz(), qt.sigmax(), qt.sigmay(), s0]
    fidelity_sum: float = 0
    # if H is not None:
    # assert comp_coord is not None,\
    # 'If parameter `H` is provided,\
    # `comp_coord` must be provided as well'
    # U: Qobj = transform_op_to_dressed_basis(U_ideal, H, comp_coord)
    # else:
    U: Qobj = U_ideal.copy()
    for si in sigmas:
        for sj in sigmas:
            if not (si == sj == s0):
                fidelity_sum += _fidelity_summand(U, U_g, si, sj)
    return (1 / 4) + (1 / 80) * fidelity_sum


def transform_op_to_dressed_basis(U: Qobj, H: Qobj, comp_coord: CompCoord) -> Qobj:
    basis: np.ndarray[complex] = H.eigenstates()[1]
    hlbrtspc_dims: list[list[int]] = basis[0].dims
    # cast operator dimensions of the full bare basis
    U_full: Qobj = cast_op_to_space(U, "composite", comp_coord, hlbrtspc_dims)
    # transform to dressed basis
    U_full = U_full.transform(basis)
    # cast back to two qubit subspace
    U_qs = cast_op_to_space(U_full, "qubit", comp_coord)
    return U_qs


def cast_op_to_space(
    op: Qobj, space: str, comp_coord: CompCoord, new_dims: list[int] = [5, 5, 5]
) -> Qobj:
    assert space in [
        "qubit",
        "composite",
    ], f'`space` param must be "qubit" or "composite", got {space}'
    basis: dict[str, str] = {"ggg": "00",
                             "geg": "01", "egg": "10", "eeg": "11"}
    if space == "qubit":
        matrix_shape = [4, 4]
        new_dims = [2, 2]
    else:
        dims = np.array(new_dims).ravel()
        dims = np.prod(dims)
        matrix_shape = [dims, dims]
    U_arr = np.zeros(shape=matrix_shape, dtype=complex)
    for bra_lbl, bra_bin in basis.items():
        for ket_lbl, ket_bin in basis.items():
            # get computational state eigenstate idices
            ket_i, ket_j, ket_k = str_to_tuple(
                ket_lbl
            )  # cast labels to index [i,j,k] of computational state |ijk>
            bra_i, bra_j, bra_k = str_to_tuple(bra_lbl)
            ket_eig_idx: int = comp_coord[ket_i, ket_j, ket_k]
            bra_eig_idx: int = comp_coord[bra_i, bra_j, bra_k]
            # get qubit subspace indices
            ket_qs_idx: int = int(ket_bin, 2)
            bra_qs_idx: int = int(bra_bin, 2)
            # populate new array
            if space == "qubit":
                U_arr[bra_qs_idx, ket_qs_idx] = op[bra_eig_idx, ket_eig_idx]
            else:
                U_arr[bra_eig_idx, ket_eig_idx] = op[bra_qs_idx, ket_qs_idx]
    return qt.Qobj(U_arr, dims=[new_dims, new_dims])


def _fidelity_summand(U: Qobj, U_g: Qobj, si: Qobj, sj: Qobj) -> float:
    # summand in eq (14)
    two_qubit_sigma = qt.tensor(si, sj)
    summand = (U_g * two_qubit_sigma * U_g.dag() *
               U * two_qubit_sigma * U.dag()).tr()
    return (
        summand.real
    )  # this term should be real, but numerical precision error may lead to imaginary parts O(1e-15)


def compute_unitary(
    trajectories: dict[str, Qobj],
    comp_states: CompTensor,
    comp_coord: CompCoord,
    hamiltonian_dims: list[int] = [[5, 5, 5], [5, 5, 5]],
) -> tuple[Qobj, Qobj]:
    dims: list[list[int, int, int], list[int, int, int]] = hamiltonian_dims
    matrix_dims: list[int, int] = [np.prod(dims[0]), np.prod(dims[1])]
    U_arr: np.ndarray[complex] = np.zeros(shape=matrix_dims, dtype=complex)
    U_qs_arr = np.zeros(shape=[4, 4], dtype=complex)
    basis: dict[str, str] = {"ggg": "00",
                             "geg": "01", "egg": "10", "eeg": "11"}
    for bra, bra_bin in basis.items():
        for ket, ket_bin in basis.items():
            # compute matrix element indices in full hilbert space
            i_bra, j_bra, k_bra = str_to_tuple(bra)
            i_ket, j_ket, k_ket = str_to_tuple(ket)
            bra_idx = comp_coord[i_bra, j_bra, k_bra]
            ket_idx = comp_coord[i_ket, j_ket, k_ket]
            # compute matrix element indices in two qubit subspace
            bra_qs_idx = int(bra_bin, 2)  # convert binary index to int
            ket_qs_idx = int(ket_bin, 2)
            bra_vector = comp_states[i_bra, j_bra, k_bra]
            ket_vector = trajectories[ket][-1]
            matrix_elem = (bra_vector.dag() * ket_vector).tr()
            U_arr[bra_idx][ket_idx] = matrix_elem
            U_qs_arr[bra_qs_idx][ket_qs_idx] = matrix_elem
    U: Qobj = qt.Qobj(U_arr, dims=dims)
    U_qs: Qobj = qt.Qobj(U_qs_arr, dims=[[2, 2], [2, 2]])
    # correct trivial phase to (along diagonal) [0,0,0,phi]
    U = trivial_z(U, comp_coord)
    U_qs = trivial_z(U_qs)
    return U, U_qs


def trivial_z(U: Qobj, comp_state_idx: CompCoord | None = None) -> Qobj:
    if comp_state_idx is None:
        return _trivial_z_qs(U)
    else:
        return _trivial_z_full_hlbrtspc(U, comp_state_idx)


def _trivial_z_qs(U: Qobj) -> Qobj:
    # set global phase to phase of 00
    U = U * np.exp(-1.0j * np.angle(U[0, 0]))
    # get phase of middle two diagonal elements
    phi_1_1 = np.angle(U[1, 1])
    phi_2_2 = np.angle(U[2, 2])
    # assemble trivial z unitary
    diag = [
        1,
        np.exp(-1.0j * phi_1_1),
        np.exp(-1.0j * phi_2_2),
        np.exp(-1.0j * (phi_1_1 + phi_2_2)),
    ]
    triv_Z = np.diag(diag)
    triv_Z = qt.Qobj(triv_Z, dims=[[2, 2], [2, 2]])
    return triv_Z * U


def _trivial_z_full_hlbrtspc(U: Qobj, comp_state_idx: CompCoord) -> Qobj:
    # set global phase to phase of comp state 000
    i0: int = comp_state_idx[0, 0, 0]
    phi_00: complex = np.angle(U[i0, i0])
    U: Qobj = U * np.exp(-1.0j * phi_00)
    # assemble trivial z unitary
    dims: tuple[tuple[int, int, int], tuple[int, int, int]] = U.dims
    matrix_dims: tuple[int, int] = [np.prod(dims[0]), np.prod(dims[1])]
    diag = np.repeat(1 + 0.0j, matrix_dims[0])
    triv_Z: np.ndarray[complex] = np.diag(diag)
    # collect indices
    i1: int = comp_state_idx[0, 1, 0]
    i2: int = comp_state_idx[1, 0, 0]
    i3: int = comp_state_idx[1, 1, 0]
    # collect phases
    phi_11 = np.angle(U[i1, i1])
    phi_22 = np.angle(U[i2, i2])
    # assemble matrix
    triv_Z[i0, i0] = 1
    triv_Z[i1, i1] = np.exp(-1.0j * phi_11)
    triv_Z[i2, i2] = np.exp(-1.0j * phi_22)
    triv_Z[i3, i3] = np.exp(-1.0j * (phi_11 + phi_22))
    triv_Z = qt.Qobj(triv_Z, dims=dims)
    return triv_Z * U


def get_trajectories(
    pulse_profile: PulseProfile,
    circuit: CompositeSystem,
    comp_states: CompTensor,
    drive_op: Qobj | Callable["...", Qobj] = drive_op,
) -> dict[str, qt.solver.Result]:
    H: list[Qobj | list[Qobj, np.ndarray[float]]] = assemble_sim_hamiltonian(
        circuit, pulse_profile, drive_op
    )
    init_states = assemble_init_states(pulse_profile, circuit, comp_states)
    pulse_params: PulseDict = pulse_profile.pulse_config.pulse_params
    return {
        state_lbl: evolve_state(state, H, pulse_params)
        for state_lbl, state in init_states.items()
    }


def evolve_state(
    state: Qobj, H: list[Qobj | list[Qobj, np.ndarray[float]]], pulse_params: PulseDict
) -> qt.solver.Result:
    # setup evolution timesteps
    tg: float = pulse_params["tg"]
    dt: float = pulse_params["dt"]
    t_ramp: float = pulse_params["t_ramp"]
    tlist: np.ndarray[float] = np.arange(0, tg + 2 * t_ramp, dt)[1:]
    try:
        return qt.mesolve(H, state, tlist).states
    except ValueError as exc:
        print(exc)
        breakpoint()


def assemble_init_states(
    pulse_profile: PulseProfile, circuit: CompositeSystem, comp_states: CompTensor
) -> dict[str, Qobj]:
    default_states: list[str | tuple(str, dict[str, complex])] = [
        "ggg",
        "egg",
        "geg",
        "eeg",
    ]
    init_states = default_states
    if "s0" in pulse_profile.pulse_config.pulse_params:
        param_states = pulse_profile.pulse_config.pulse_params["s0"]
        if isinstance(param_states, list):
            init_states = default_states + param_states
    return spec_list_to_state_dict(init_states, circuit, comp_states)


def spec_list_to_state_dict(
    states: list[str | tuple(str, dict[str, complex])],
    circuit: CompositeSystem,
    comp_states: CompTensor,
) -> dict[str, Qobj]:
    state_vectors: dict[str, Qobj] = {}  # initialize return dict
    for spec in states:
        if isinstance(spec, str):
            state_vectors[spec] = spec_to_state(spec, comp_states)
        else:
            state_name = spec[0]
            state_recipe = spec[1]
            state_vectors[state_name] = spec_to_state(
                state_recipe, comp_states)
    return state_vectors


def spec_to_state(
    state_recipe: str | tuple[str, dict[str, complex]], comp_states: CompTensor
) -> Qobj:
    assert isinstance(
        state_recipe, str | list | tuple | dict
    ), f"first parameter must be of type str, list, tuple or dict, \
        got {type(state_recipe)}"
    if isinstance(state_recipe, str):
        return str_to_state(state_recipe, comp_states)
    elif isinstance(state_recipe, tuple | list):
        return dict_to_state(state_recipe[1], comp_states)
    else:
        return dict_to_state(state_recipe, comp_states)


def str_to_state(state_lbl: str, comp_states: CompTensor) -> Qobj:
    i, j, k = str_to_tuple(state_lbl)
    return comp_states[i][j][k]


def dict_to_state(state_recipe: dict[str, float], comp_states: CompTensor) -> Qobj:
    state = 0
    for state_lbl, coef in state_recipe.items():
        state += coef * str_to_state(state_lbl, comp_states)
    return state


def assemble_empty_profile(
    name: str,
    pulse_profile: PulseProfile,
    circuit: CompositeSystem | None = None,
    circuit_config: CircuitParams | None = None,
    gate_profile: GateProfile | None = None,
) -> GateProfile:
    if gate_profile is None:
        gate_profile: GateProfile = GateProfile(name, pulse_profile)
    assert any(
        [
            x is not None
            for x in [
                circuit,
                circuit_config,
                gate_profile.circuit,
                gate_profile.circuit_config,
            ]
        ]
    ), f"No circuit or circuit\
    configuration given for assembly\
    of {name} gate profile.\n \
    You must provide either circuit_config\
    or a gate_profile object with either\
    attribute circuit_config or circuit\
    not None."
    if circuit is not None:
        if circuit_config is None:
            warn(
                f"Circuit initialized for gate {name} but no circuit\
                 configuration dict given, caching may fail"
            )
        gate_profile.circuit: CompositeSystem = circuit
        gate_profile.circuit_config: CircuitParams = circuit_config
    elif circuit_config is not None:
        gate_profile.circuit: CompositeSystem = sim_setup.setup_circuit(
            circuit_config)
        gate_profile.circuit_config: CircuitParams = circuit_config
    else:
        gate_profile.circuit: CompositeSystem = pulse_profile.circuit
        gate_profile.circuit_config = pulse_profile.pulse_config.circuit_config
    gate_profile.pulse_profile = pulse_profile
    gate_profile.pulse = pulse_profile.pulse
    return gate_profile


def assemble_sim_hamiltonian(
    circuit: CompositeSystem,
    pulse_profile: PulseProfile,
    drive_op: Qobj | Callable[CompositeSystem, Qobj],
) -> list[Qobj | list[Qobj, np.ndarray[float]]]:
    """${1}DOCYAS--

    --NPDOCYAS--

    Parameters
    ----------
    circuit : CompositeSystem
        --NPDOCYAS--
    pulse_profile : PulseProfile
        --NPDOCYAS--
    drive_op : Qobj | Callable[CompositeSystem, Qobj]
        --NPDOCYAS--

    Returns
    -------
    list[Qobj | list[Qobj, np.ndarray[float]]]
        --NPDOCYAS--

    Examples
    --------
    --NPDOCYAS--

    """

    H_0: Qobj = circuit.H
    if isinstance(drive_op, Qobj):
        H_d: Qobj = drive_op
    else:
        H_d: Qobj = drive_op(circuit)
    drive: np.ndarray[float] = pulse_profile.pulse
    return [H_0, [H_d, drive]]


# Utility Funcs
def str_to_tuple(lbl: str) -> tuple[int]:
    str_to_int_map = {"g": 0, "0": 0, "e": 1, "1": 1, "f": 2, "2": 2}
    return tuple([str_to_int_map[char] for char in lbl])
