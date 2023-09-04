import yaml
from pathlib import Path
from qutip import Qobj, qsave, qload
from os.path import isfile
import os
import numpy as np
from typing import TypeVar, TypeAlias
from numbers import Number

CircuitParams: TypeAlias = dict[str, float | dict[str, float]]
T = TypeVar("T")
PulseDict = TypeVar("PulseDict")
GateDict = TypeVar("GateDict")
GateProfile = TypeVar("GateProfile")
PulseProfile = TypeVar("PulseProfile")
PulseConfig = TypeVar("PulseConfig")
PulseDict = TypeVar("PulseDict")
DataObj = TypeVar("DataObj", PulseConfig, PulseProfile, GateProfile)
DataDict = TypeVar("DataDict", PulseDict, GateDict)
# filename generators

# Setup Paths
project_root = Path(__file__).parents[1]
# setup config path
config_dir = os.path.join(project_root, "config")
# setup results path
output_dir = os.path.join(project_root, "output")


# filename setup
def add_path(fname: str, dir_label: str) -> str:
    assert dir_label in [
        "config",
        "output",
        "pulse",
        "Pulse",
        "pulses",
        "gate",
        "Gate",
        "sim_results",
    ], f'dir_label must be "config" or "output", got "{dir_label}"'
    paths = {
        "config": config_dir,
        "output": output_dir,
        "pulse": os.path.join(output_dir, "pulses"),
        "Pulse": os.path.join(output_dir, "pulses"),
        "pulses": os.path.join(output_dir, "pulses"),
        "gate": os.path.join(output_dir, "sim_results"),
        "Gate": os.path.join(output_dir, "sim_results"),
        "sim_results": os.path.join(output_dir, "sim_results"),
    }
    return os.path.join(paths[dir_label], fname)


def build_pulse_fname(pulse_config: PulseConfig | PulseProfile) -> str:
    return f"{pulse_config.name}.npy"


def build_pulse_path(pulse_config: PulseConfig | PulseProfile) -> str:
    fname = build_pulse_fname(pulse_config)
    return add_path(fname, "pulse")


# def build_pulse_component_fname(
    # pulse_desc: PulseConfig | PulseProfile, component_name: str
# ) -> str:
    # return f"{pulse_desc.name}-{component_name}.npy"


# def build_pulse_component_path(
    # pulse_desc: PulseConfig | PulseProfile, component_name: str
# ) -> str:
    # fname = build_pulse_component_fname(pulse_desc, component_name)
    # return add_path(fname, "pulse")
#

def build_gate_fname(gate: GateProfile) -> str:
    return f"{gate.name}.qu"


def build_gate_path(gate: GateProfile) -> str:
    fname = build_gate_fname(gate)
    return add_path(fname, "gate")


# File Writing


def cache_pulse(
    pulse: PulseConfig | PulseProfile,
        pulse_array: np.ndarray[float] | None = None,
        multiprocess: bool = False
) -> None:
    # if pulse_array is None, pulse must be of type PulseProfile
    fpath = build_pulse_path(pulse)
    fname = build_pulse_fname(pulse)
    if multiprocess:
        save_desc_multiprocess(pulse, fname, "pulse")
    else:
        save_desc(pulse, fname, "pulse")
    np.save(fpath, pulse_array)


#   def cache_pulse_component(
    #   pulse: PulseConfig | PulseProfile,
    #   comp_name: str,
    #   component_array: np.ndarray[float] | None = None,
    #   multiprocess: bool = False
#   ) -> None:
    #   if component_array is None:
    #   component_array = pulse.profiled_components[comp_name]
    #   fname = build_pulse_component_fname(pulse, comp_name)
    #   fpath = build_pulse_component_path(pulse, comp_name)
    #   if multiprocess:
    #   save_component_desc_multiprocess(pulse, comp_name, fname)
    #   else:
    #   save_component_desc(pulse, comp_name, fname)
    #   np.save(fpath, component_array)


def cache_gate(gate: GateProfile,
               multiprocess: bool = False) -> None:
    fpath = build_gate_path(gate)
    fname = build_gate_fname(gate)
    if multiprocess:
        save_desc_multiprocess(gate, fname, mode='Gate')
    else:
        save_desc(gate, fname, mode="Gate")
    # unitary = gate.unitary
    # trajectories = gate.trajectories
    # data: dict[str, Qobj | dict[str, np.ndarray[complex]]] = {
    # "unitary": unitary,
    # "trajectories": trajectories,
    # }
    # qutip annoyingly always appends .qu to filenames, even if it
    # already ends in .qu, so we truncate the extension
    if fpath[-3:] == ".qu":
        fpath = fpath[:-3]
    qsave(gate, fpath)


# File logging


# def save_component_desc(pulse: PulseProfile, component_name: str, fname: str)\
    # -> None:
    # pulse_name: str = pulse.name
    # pulse_desc = pulse.as_noNone_dict()
    # component_desc: dict[str, float | complex] =\
    # pulse_desc['save_components'][component_name]
    # if fname[-4:] != ".npy":
    # fname = fname + ".npy"
    # component_desc['file'] = fname
    # cache = get_cache_description("pulse")
    # if "save_components" not in cache[pulse_name]:
    # cache[pulse_name]["save_components"] = {}
    # cache[pulse_name]['save_components'][component_name] = component_desc
    # cache_path = add_path("cache_desc.yaml", "pulses")
    # with open(cache_path, "w") as yaml_file:
    # yaml.dump(cache, yaml_file)
#
#
# def save_component_desc_multiprocess(
    # pulse: PulseProfile,
    # component_name: str,
    # fname: str) -> None:
    # pulse_name: str = pulse.name
    # pulse_desc = pulse.as_noNone_dict()
    # component_desc: dict[str, float | complex] =\
    # pulse_desc['save_components'][component_name]
    # if fname[-4:] != ".npy":
    # fname = fname + ".npy"
    # component_desc['file'] = fname
    # yaml_name = add_path(f'{pulse_name}-{component_name}-mpc.yaml',
    # 'pulses')
    # with open(yaml_name, "w") as yaml_file:
    # yaml.dump({pulse_name: {
    # component_name: component_desc
    # }}, yaml_file)


#   def pool_component_desc():
    #   dir = os.path.join(project_root, 'output', 'pulses')
    #   cached_files = os.listdir(dir)
    #   cache = get_cache_description('pulses')
    #   for file in cached_files:
    #   if file[-8:] == 'mpc.yaml':
    #   file_path = os.path.join(project_root,
    #  'output',
    #  'pulses',
    #  file)
    #   with open(file_path, 'r') as yaml_file:
    #   desc = yaml.safe_load(yaml_file)
    #   pulse_name = list(desc.keys())[0]
    #   comp_name = list(desc[pulse_name].keys())[0]
    #   comp_desc = desc[pulse_name][comp_name]
    #   if 'save_components' not in cache[pulse_name]:
    #   cache[pulse_name]['save_components'] = {}
    #   cache[pulse_name]['save_components'][comp_name] = \
    #   comp_desc
    #   os.remove(file_path)
    #   cache_path = add_path('cache_desc.yaml', 'pulses')
    #   with open(cache_path, 'w') as yaml_file:
    #   yaml.dump(cache, yaml_file)


def pool_desc(mode: str):
    mode_dict = {'gate': 'sim_results',
                 'pulse': 'pulses'}
    dir = os.path.join(project_root, 'output', mode_dict[mode])
    cached_files = os.listdir(dir)
    pool = {}
    for file in cached_files:
        if file[-7:] == 'mp.yaml':
            file_path = os.path.join(project_root, 'output',
                                     mode_dict[mode], file)
            with open(file_path, 'r') as yamlfile:
                pool.update(yaml.safe_load(yamlfile))
            os.remove(file_path)

    cache = get_cache_description(mode)
    for name, desc in pool.items():
        cache[name] = desc
    cache_path = add_path("cache_desc.yaml", mode)
    with open(cache_path, 'w') as yamlfile:
        yaml.dump(cache, yamlfile)


def save_desc(target: DataObj, fname: str, mode: str = "pulse") -> None:
    assert mode in [
        "pulse",
        "Pulse",
        "gate",
        "Gate",
        "sim_results",
    ], f"{mode} not a valid mode parameter"
    cache_path = add_path("cache_desc.yaml", mode)
    desc = target.as_noNone_dict()
    desc["file"] = fname
    cache = get_cache_description(mode)
    name = desc.pop("name")
    cache[name] = desc
    with open(cache_path, "w") as yaml_file:
        yaml.dump(cache, yaml_file)


def save_desc_multiprocess(target: DataObj,
                           fname: str,
                           mode: str = 'pulse') -> None:
    assert mode in [
        "pulse",
        "Pulse",
        "gate",
        "Gate",
        "sim_results",
    ], f"{mode} not a valid mode parameter"
#    cache_path = add_path("cache_desc.yaml", mode)
    desc = target.as_noNone_dict()
    desc["file"] = fname
    name = desc.pop("name")
    yaml_name = add_path(f'{name}-mp.yaml', mode)
    with open(yaml_name, "w") as yaml_file:
        yaml.dump({name: desc}, yaml_file)

# File Reading


def load_pulse(pulse_config: PulseConfig) -> np.ndarray[float]:
    cache = get_cache_description("pulse")
    pulse_name: str = pulse_config.name
    assert (
        pulse_name in cache
    ), f"pulse {pulse_name} has not been saved in ../output/pulses"
    fname: str = cache[pulse_name]["file"]
    fpath: str = add_path(fname, "pulses")
    if fpath[-4:] != ".npy":
        fpath = fpath + ".npy"
    return np.load(fpath, allow_pickle=True)


def load_unitary(Uname: str) -> Qobj:
    fpath = os.path.join(project_root,
                         'config',
                         'target_unitaries',
                         f'{Uname}')
    return qload(fpath)


def convert_units(params: dict[str, float | dict[str, float]],
                  unit_conversions:
                  dict[str, float | dict[str, float]] | None | Number
                  ) -> dict:
    if unit_conversions is None:
        return params

    elif isinstance(unit_conversions, Number):
        for lbl, param_val in params:
            if isinstance(param_val, Number):
                params[lbl] = unit_conversions * param_val
            elif isinstance(param_val, dict):
                params[lbl] = convert_units(param_val,
                                            unit_conversions)
            else:
                exc = TypeError(
                    f'Parameters must be of numeric type, or a \
                    dictionary of string labels and numeric values.\
                     Got {type(param_val)}'
                )
                raise exc

    elif isinstance(unit_conversions, dict):
        for lbl, conversion in unit_conversions:
            if lbl in params:
                param_val = params[lbl]
                if isinstance(conversion, dict):
                    params[lbl] = convert_units(param_val, conversion)
                elif isinstance(conversion, Number):
                    params[lbl] = param_val * conversion
                else:
                    exc = TypeError(f'Unit conversions must be of Numeric\
                    Type, recieved {type(conversion)}')
                    raise exc
    else:
        exc = TypeError(f'Unit conversions must be of Numeric type, or \
        a (possibly nested) dictionary of string keys and Numeric values.\
         Got {type(unit_conversions)}')
    return params

    # def load_pulse_component(
    # pulse_config: PulseConfig, component_name: str
    # ) -> np.ndarray[complex | float]:
    # cache: dict[str, PulseDict] = get_cache_description("pulse")
    # pulse_name: str = pulse_config.name
    # assert (
    # pulse_name in cache
    # ), f"pulse {pulse_name} has not been saved in ../output/pulses"
    # assert (
    # "save_components" in cache[pulse_name]
    # ), f"pulse {pulse_name} does not have any saved components"
    # pulse_components: dict[str, str] = cache[pulse_name]["save_components"]
    # assert (
    # component_name in pulse_components
    # ), f"component {component_name} has not been profiled for {pulse_name}"
    # fname: str = pulse_components[component_name]["file"]
    # fpath: str = add_path(fname, "pulses")
    # if fpath[-4:] != ".npy":
    # fpath = fpath + ".npy"
    # return np.load(fpath, allow_pickle=True)


def load_gate(gate_name) -> GateProfile:
    cache = get_cache_description("gate")
    assert gate_name in cache, f"{gate_name} has not been cached"
    fname: str = cache[gate_name]["file"]
    fpath: str = add_path(fname, "gate")
    if fname[-3:] == ".qu":
        fpath = fpath[:-3]
    return qload(fpath)


def get_params(path: str,
               conversions: dict[str, Number |
                                 dict[str, Number]] = None
               ) -> dict[str, float | dict[str, float]]:
    with open(path, "r") as stream:
        try:
            params: dict | None = yaml.safe_load(stream)
            params = tidy_cache(params)
        except yaml.YAMLError as exc:
            print(exc)
    params = convert_units(params, conversions)
    return params


def get_ct_params(conversions: dict[str, Number |
                                    dict[str, Number]] = None
                  ) -> dict[str, float | dict[str, float]]:
    fpath = add_path("circuit_parameters.yaml", "config")
    return get_params(fpath, conversions)


def get_pulse_defaults(conversions: dict[str, Number |
                                         dict[str, Number]] = None
                       ) -> dict[str, float | dict[str, float]]:
    fpath = add_path("pulse_parameters.yaml", "config")
    return get_params(fpath, conversions)


def get_pulse_specs(conversions: dict[str, Number |
                                      dict[str, Number]] = None
                    ) -> dict[str, float | dict[str, float]]:
    fpath = add_path("pulses.yaml", "config")
    return get_params(fpath, conversions)


def get_cache_description(cache_lbl: str = "pulse") -> dict[str, DataDict]:
    assert cache_lbl in [
        "pulse",
        "pulses",
        "gate",
        "Gate",
        "Pulse",
        "sim_results",
    ], f'`cache_lbl` must be "pulse", "gate", or "sim_results", got {cache_lbl}'
    fpath = add_path("cache_desc.yaml", cache_lbl)
    if is_file(fpath):
        return tidy_cache(get_params(fpath))
    else:
        return {}


# utility func to replace None with empty dictionaries


def tidy_cache(cache_dirty: dict) -> dict:
    if cache_dirty is None:
        return {}
    cache = cache_dirty.copy()
    for key, value in cache.items():
        if isinstance(value, dict):
            cache[key] = tidy_cache(value)
        elif value is None:
            cache[key] = dict()
    return cache


# Existence Checking


def pulse_cached(pulse_desc: PulseConfig | PulseProfile) -> bool:
    desc: PulseDict = pulse_desc.as_dict()
    name: str = desc.pop("name")
    # remove profiled components from pulse description
    # if "save_components" in desc:
    # desc.pop("save_components")
    cache: dict[str, PulseDict] = get_cache_description("pulse")
    if name in cache:
        cached_desc: PulseDict = cache[name]
        matched_params: list[bool] = [
            desc[lbl] == cached_desc[lbl] for lbl in desc]
        return all(matched_params)
    else:
        return False


def gate_cached(gate_description: GateDict | GateProfile) -> bool:
    if isinstance(gate_description, dict):
        gate_desc: dict = gate_description.copy()
    else:
        gate_desc = gate_description.as_dict()
    gate_name = gate_desc['name']
    cache = get_cache_description('gate')
    if gate_name in cache:
        cached_desc = cache[gate_name]
        if 'file' in cached_desc:
            identity_labels = ['circuit_config',
                               'pulse_config']
            return all([
                cached_desc[lbl] == gate_desc[lbl]
                for lbl in identity_labels
            ])


def is_file(path: str) -> bool:
    return isfile(path)
