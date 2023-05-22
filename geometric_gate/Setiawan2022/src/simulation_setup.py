#!/usr/bin/env python
"""Simulation setup and pulse cache management.

Collect simulation parameters from ../config and package
    the parameters into a Config container object for easy,
    orderly access. Then, use the initialized Config object
    to generate the appropriate pulses, package the
    generated pulses along with their configurations and
    profiled components into a PulseProfile container object.
    This module also uses file_io.py to cache every generated
    pulseProfile into ../output/pulses. If a pulse specififed by
    the initialized Config already exists in ../output/pulses, then
    that pulse will be loaded from disk instead of re-generating the
    pulse.
"""
from __future__ import annotations
import qutip as qt
import numpy as np
import file_io
import inspect
import hashing
from composite_systems import CompositeSystem
from static_system import build_static_system
from dataclasses import dataclass, field
from pulse import Pulse, StaticPulseAttr
from state_labeling import CompCoord, CompTensor
from typing import TypeAlias, TypeVar, Any, Generic, Callable, Iterable
from functools import partial

dummy_res = qt.mesolve(qt.qeye(2), qt.basis(2, 0), [0, 1])
Qobj: TypeAlias = qt.Qobj
result: TypeAlias = qt.solver.Result
CircuitParams: TypeAlias = dict[str, float | dict[str, float]]
PulseParams = TypeVar('PulseParams')
PulseDict: TypeAlias = dict[str, str | CircuitParams | PulseParams]
# PulseProfile: TypeAlias = tuple[np.ndarray[float], dict[str, complex]]


@dataclass
class PulseConfig(Generic[PulseParams]):
    pulse_params: PulseParams
    geo_phase: float
    name: str
    circuit_config: CircuitParams | None = None
    save_components: dict[str, dict[str, Any]] = field(default_factory={})
    target_unitary: str = 'CZ'

    def as_dict(self) -> PulseDict:
        desc: PulseDict = {}  # init descriptor dictionary
        desc['name'] = self.name
        desc.update(self.pulse_params)
        desc['geo_phase'] = self.geo_phase
        desc['circuit_config'] = self.circuit_config
        desc['save_components'] = self.save_components
        desc['target_unitary'] = self.target_unitary
        return desc

    def as_noNone_dict(self) -> PulseDict:
        """Same behavior as as_dict, but replaces all instances of \
        None with an empty dictionary. Used when None types lead to
        undefined behavior."""
        return file_io.tidy_cache(self.as_dict())

    def __hash__(self) -> int:
        return hashing.hash_dict(self.as_dict())

    def copy(self) -> PulseConfig:
        new_pulse_params = {}
        self_desc = self.as_dict()
        for key in self.pulse_params.keys():
            new_pulse_params[key] = self_desc.pop(key)
        return PulseConfig(new_pulse_params, **self_desc)


@dataclass
class PulseProfile(Generic[PulseParams]):
    """ Simple container class for pulse data
    """
    name: str
    geo_phase: float
    pulse: np.ndarray[float]
    pulse_config: PulseConfig
    circuit: CompositeSystem | None = None
    profiled_components: dict[str, np.ndarray[float | complex]] = \
        field(default_factory=dict)

    def as_dict(self) -> PulseDict:
        return self.pulse_config.as_dict()

    def __hash__(self) -> int:
        return hash((tuple(self.pulse), self.pulse_config.__hash__()))

    def copy(self) -> PulseProfile:
        return PulseProfile(
            name=self.name,
            geo_phase=self.geo_phase,
            pulse=self.pulse,
            pulse_config=self.pulse_config.copy(),
            circuit=self.circuit,
            profiled_components=self.profiled_components.copy()
        )


@dataclass
class Config(Generic[PulseParams]):
    """Simple container class for simulation configuration details.

    Public Instance Variables
    __________________
    ct_params: Circuit configuration parameters, must contain
    variable of type config with string keys A, B, C, interaction, with
    values
        A: dict{'E_C': float, 'E_J': float, 'E_L': float,
                'phi_ext': float, 'w': float}
        B: Same as A
        C: dict{'U': float, 'w': float}
        interaction: {'g_AC': float, 'g_BC': float, 'g_AB': float}
    E_C, E_J, E_L are the standard fluxonium energy parameters, phi_ext
    is the external flux threading the linear inductor, and w is the
    qubit frequency. 'U' is the transmon coupler 'C' anharmonicity, and
    g_ij is the interaction strength between circuit components i,j.

    pulses: Dictionary with key '<gate name>' and value corresponding
    to custom pulse parameters for the desired gate. Description of
    needed pulse parameters may be found in ../config/pulses.yaml

    default_pulse_conf (Optional): Dictionary with default pulse parameters,
    used to fill in parameters not provided by the pulse configs in
    `pulses`.
    """

    ct_params: CircuitParams
    pulse_config_dict: dict[str, PulseConfig | PulseParams]
    default_pulse_params: PulseParams | None = None


def collect_sim_params_from_configs() -> Config:
    """Collect simulation parameters from ../config directory.

    Preconditon
    -----------
    The following files must be written in ../config:

    pulses.yaml
    pulse_parameters.yaml
    circuit_parameters.yaml

    Returns
    -------
    Config
        Config object with attributes ct_params, pulses,
    default_pulse_conf, containing the data from the files
    circuit_parameters.yaml, pulses.yaml, and pulse_parameters.yaml,
    respectively.
    """
    circuits = file_io.get_ct_params()
    pulse_defaults = file_io.get_pulse_defaults()
    pulse_collection_params = file_io.get_pulse_specs()
    return Config(ct_params=circuits,
                  default_pulse_params=pulse_defaults,
                  pulse_config_dict=pulse_collection_params)


def setup_sim(sim_config: Config) -> dict[str, PulseProfile]:
    pulse_configs = setup_pulse_param_dict(sim_config.pulse_config_dict,
                                           sim_config.default_pulse_params,
                                           sim_config.ct_params['pulse_gen_ct'])
    circuit = setup_circuit(sim_config.ct_params['pulse_gen_ct'])
    pulse_profiles = get_pulse_profile_dict(pulse_configs, circuit)
    return pulse_profiles


def setup_sim_from_configs(pulse_lbls: list[str] = []) -> tuple[Config, dict[str, PulseProfile]]:
    config: Config = collect_sim_params_from_configs()
    # if pulse_lbls is nonempty, only setup sim for pulses with labels
    # in pulse_lbls
    if pulse_lbls:
        for lbl in config.pulse_config_dict:
            if lbl not in pulse_lbls:
                config.pulse_config_dict.pop(lbl)
    return config, setup_sim(config)


def setup_circuit(ct_params: CircuitParams) -> CompositeSystem:
    # in static_systems.build_bare_systems, some parameters corresponding
    # to subsystem construction are provided as a dictionary, while parameters
    # setting attributes of the whole hilbert space are provided as regular
    # arguments, so we must first separate ct_params into two dictionaries
    # corresponding to the two parameter types
    copy_params = ct_params.copy()
    global_args: dict[str, float] = copy_params.pop('global')
    return build_static_system(copy_params, **global_args)


def setup_pulse_param_dict(pulse_param_dict: dict[str, PulseParams],
                           default_params: PulseParams,
                           ct_params: CircuitParams) ->\
        dict[str, PulseConfig]:
    return {gate_name: setup_pulse_params(gate_name,
                                          non_default_params,
                                          default_params,
                                          ct_params)
            for gate_name, non_default_params in pulse_param_dict.items()}


def get_pulse_profile_dict(pulse_config_dict: dict[str, PulseConfig],
                           circuit: CompositeSystem)\
        -> dict[str, PulseProfile]:
    return {gate_name: get_pulse_profile(pulse_config, circuit)
            for gate_name, pulse_config in pulse_config_dict.items()}


def setup_pulse_params(gate_name: str,
                       gate_pulse_params: PulseParams,
                       default_params: PulseParams,
                       ct_params: CircuitParams) -> PulseConfig:
    non_default_params = gate_pulse_params.copy()  # prevent side effects on input
    if 'save_components' in non_default_params:
        save_components = non_default_params.pop('save_components')
    else:
        save_components = None
    if 'target_unitary' in non_default_params:
        target_unitary = non_default_params.pop('target_unitary')
    else:
        target_unitary = 'CZ'
    params = default_params.copy()
    params.update(non_default_params)
    # params = non_default_params.copy()
    # params.update({lbl:value for lbl, value in default_params.items()
    #               if lbl not in params})
    # params += {lbl: value for lbl, value in default_params.items()
    # if lbl not in params}
    phase = params.pop('geo_phase')
    if 'circuit_config' in params:
        circuit_config: CircuitParams = params['circuit_config']
    else:
        circuit_config: CircuitParams = ct_params
    pulse_config = PulseConfig(pulse_params=params,
                               save_components=save_components,
                               name=gate_name,
                               geo_phase=phase,
                               circuit_config=circuit_config,
                               target_unitary=target_unitary)
    return pulse_config


def get_pulse_profile(pulse_config: PulseConfig,
                      circuit: CompositeSystem,
                      cache_results: bool = True)\
        -> PulseProfile:
    if pulse_config.circuit_config is not None:
        pulse_ct: CompositeSystem =\
            setup_circuit(pulse_config.circuit_config)
    else:
        pulse_ct: CompositeSystem = circuit
    cached_pulse_configs: dict[str, PulseParams] = \
        file_io.get_cache_description()
    pulse_array: np.ndarray | None = None
    components: dict[str, np.ndarray[complex]] = {}
    # determine whether pulse or components are cached
    pulse_builder: Pulse | None = None
    if pulse_cached(cached_pulse_configs, pulse_config):
        pulse_array = file_io.load_pulse(pulse_config)
    else:
        pulse_builder: Pulse = Pulse(pulse_config.pulse_params, pulse_ct)
        pulse_array: np.ndarray = get_pulse(pulse_config, pulse_builder)
        if cache_results:
            file_io.cache_pulse(pulse_config, pulse_array)
    for comp_name in pulse_config.save_components:
        if component_cached(cached_pulse_configs,
                            pulse_config,
                            comp_name):
            components[comp_name] = file_io.load_pulse_component(pulse_config,
                                                                 comp_name)
        else:
            component_array = get_component(pulse_config,
                                            comp_name,
                                            pulse_builder,
                                            pulse_ct)
            components[comp_name] = component_array
            if cache_results:
                file_io.cache_pulse_component(pulse_config,
                                              comp_name,
                                              component_array)
    profile = PulseProfile(
        name=pulse_config.name,
        geo_phase=pulse_config.geo_phase,
        pulse=pulse_array,
        pulse_config=pulse_config,
        circuit=pulse_ct,
        profiled_components=components)
    return profile


def get_pulse(pulse_config: PulseConfig, pulse_builder: Pulse | None = None,
              circuit: CompositeSystem | None = None) -> np.ndarray[float]:
    assert not (pulse_builder is None and circuit is None),\
        '`pulse_builder` and `circuit` params cannot both be None'
    if pulse_builder is None:
        pulse_builder = Pulse(pulse_config.pulse_params, circuit)
    dt = pulse_config.pulse_params['dt']
    tg = pulse_config.pulse_params['tg']
    t_ramp = pulse_config.pulse_params['t_ramp']
    tlist = np.arange(0, tg, dt)
    tlist_ramp = np.arange(0, t_ramp, dt)
    geo_phase = pulse_config.geo_phase
    return pulse_builder.build_pulse(tlist, geo_phase, tlist_ramp)


def get_component(pulse_config: PulseConfig,
                  component_name: str,
                  pulse_builder: Pulse | None = None,
                  circuit: CompositeSystem | None = None)\
        -> dict[str, np.ndarray[complex]]:
    assert not (pulse_builder is None and circuit is None),\
        '`pulse_builder` and `circuit` params cannot both be None'
    if pulse_builder is None:
        pulse_builder = Pulse(pulse_config.pulse_params, circuit)
    pulse_component_funcs: list[tuple[str, Callable['...', complex]]] =\
        inspect.getmembers(pulse_builder,
                           predicate=inspect.ismethod)
    pulse_component_funcs: dict[str, Callable['...', complex]] =\
        dict(pulse_component_funcs)  # cast to dict type
    func: Callable['...', complex] = pulse_component_funcs[component_name]
    dt: float = pulse_config.pulse_params['dt']
    tg: float = pulse_config.pulse_params['tg']
    component_kwargs = pulse_config.save_components[component_name]
    if component_kwargs is None:
        component_kwargs = {}
    tlist = np.arange(0, tg, dt)
    return np.array([
        func(t, **component_kwargs) for t in tlist
    ])


def pulse_cached(cached_pulse_configs: dict[str, PulseParams],
                 pulse_config: PulseConfig) -> bool:
    cache: dict[str, PulseParams] = cached_pulse_configs.copy()
    pulse_name: str = pulse_config.name
    if pulse_name in cache:
        cached_params: PulseParams = cache[pulse_name].copy()
        pulse_desc: PulseDict = pulse_config.as_dict()
        # replace None values with empty dicts
        pulse_desc = file_io.tidy_cache(pulse_desc)
        for key in ['file', 'save_components']:
            if key in cached_params and key not in pulse_desc:
                cached_params.pop(key)
        pulse_desc.pop('name')
        for data in [pulse_desc, cached_params]:
            if 'save_components' in data:
                data.pop('save_components')
        return pulse_desc == cached_params
    else:
        return False


def component_cached(cached_pulse_configs: dict[str, PulseParams],
                     pulse_config: PulseConfig,
                     component_name) -> bool:
    assert component_name in pulse_config.save_components,\
        f'Wanted to check if {component_name} saved in cache\
    for {pulse_config.name}'\
    + f', but {component_name} is not specified by config\
    for {pulse_config.name}'
    cache: dict[str, PulseParams] = cached_pulse_configs.copy()
    if pulse_config.name in cache:
        if 'save_components' in cache[pulse_config.name]:
            component_args = pulse_config.save_components[component_name]
            # replace any instance of None with empty dictionary
            component_args = file_io.tidy_cache(component_args)
            cached_components: dict[str, dict[str, complex] | None] = \
                cache[pulse_config.name]['save_components']
            if component_name in cached_components:
                if 'file' in cached_components[component_name]:
                    cached_components[component_name].pop('file')
                    return component_args == cached_components[component_name]
    return False
