import yaml
from pathlib import Path
from qutip import Qobj, qsave, qload
from os.path import isfile
import os
import numpy as np
from typing import TypeVar, TypeAlias

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

################# Setup Paths
project_root = Path(__file__).parents[1]
# setup config path
config_dir = os.path.join(project_root, "config")
# setup results path
output_dir = os.path.join(project_root, "output")


############ filename setup
def add_path(fname: str, dir_label: str) -> str:
    assert dir_label in [
        "config",
        "output",
        "pulse",
        "pulses",
        "gate",
        "sim_results",
    ], f'dir_label must be "config" or "output", got "{dir_label}"'
    paths = {
        "config": config_dir,
        "output": output_dir,
        "pulse": os.path.join(output_dir, "pulses"),
        "pulses": os.path.join(output_dir, "pulses"),
        "gate": os.path.join(output_dir, "sim_results"),
        "sim_results": os.path.join(output_dir, "sim_results"),
    }
    return os.path.join(paths[dir_label], fname)


def build_pulse_fname(pulse_config: PulseConfig | PulseProfile) -> str:
    return f"{pulse_config.name}-{hash(pulse_config)}.npy"


def build_pulse_path(pulse_config: PulseConfig | PulseProfile) -> str:
    fname = build_pulse_fname(pulse_config)
    return add_path(fname, "pulse")


def build_pulse_component_fname(
    pulse_desc: PulseConfig | PulseProfile, component_name: str
) -> str:
    return f"{pulse_desc.name}-{component_name}-{hash(pulse_desc)}.npy"


def build_pulse_component_path(
    pulse_desc: PulseConfig | PulseProfile, component_name: str
) -> str:
    fname = build_pulse_component_fname(pulse_desc, component_name)
    return add_path(fname, "pulse")


def build_gate_fname(gate: GateProfile) -> str:
    return f"{gate.name}-{hash(gate)}.qu"


def build_gate_path(gate: GateProfile) -> str:
    fname = build_gate_fname(gate)
    return add_path(fname, "gate")


######## File Writing


def cache_pulse(
    pulse: PulseConfig | PulseProfile, pulse_array: np.ndarray[float] | None = None
) -> None:
    # if pulse_array is None, pulse must be of type PulseProfile
    fpath = build_pulse_path(pulse)
    fname = build_pulse_fname(pulse)
    save_desc(pulse, fname, "pulse")
    np.save(fpath, pulse_array)


def cache_pulse_component(
    pulse: PulseConfig | PulseProfile,
    comp_name: str,
    component_array: np.ndarray[float] | None = None,
) -> None:
    if component_array is None:
        component_array = pulse.profiled_components[comp_name]
    fname = build_pulse_component_fname(pulse, comp_name)
    fpath = build_pulse_component_path(pulse, comp_name)
    save_component_desc(pulse, comp_name, fname)
    np.save(fpath, component_array)


def cache_gate(gate: GateProfile) -> None:
    fpath = build_gate_path(gate)
    fname = build_gate_fname(gate)
    save_desc(gate, fname, mode="Gate")
    # unitary = gate.unitary
    # trajectories = gate.trajectories
    # data: dict[str, Qobj | dict[str, np.ndarray[complex]]] = {
    # "unitary": unitary,
    # "trajectories": trajectories,
    # }
    ## qutip annoyingly always appends .qu to filenames, even if it
    ## already ends in .qu, so we truncate the extension
    if fpath[-3:] == ".qu":
        fpath = fpath[:-3]
    qsave(gate, fpath)


############# File logging


def save_component_desc(pulse: PulseProfile, component_name: str, fname: str)\
    -> None:
    pulse_name: str = pulse.name
    pulse_desc = pulse.as_noNone_dict()
    component_desc: dict[str, float | complex] =\
        pulse_desc['save_components'][component_name]
    if fname[-4:] != ".npy":
        fname = fname + ".npy"
    component_desc['file'] = fname
    cache = get_cache_description("pulse")
    if "save_components" not in cache[pulse_name]:
        cache[pulse_name]["save_components"] = {}
    cache[pulse_name]['save_components'][component_name] = component_desc
    cache_path = add_path("cache_desc.yaml", "pulses")
    with open(cache_path, "w") as yaml_file:
        yaml.dump(cache, yaml_file)


def save_desc(target: DataObj, fname: str, mode: str = "pulse") -> None:
    assert mode in [
        "pulse",
        "gate",
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


############ File Reading


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
    return np.load(fpath)


def load_pulse_component(
    pulse_config: PulseConfig, component_name: str
) -> np.ndarray[complex | float]:
    cache: dict[str, PulseDict] = get_cache_description("pulse")
    pulse_name: str = pulse_config.name
    assert (
        pulse_name in cache
    ), f"pulse {pulse_name} has not been saved in ../output/pulses"
    assert (
        "save_components" in cache[pulse_name]
    ), f"pulse {pulse_name} does not have any saved components"
    pulse_components: dict[str, str] = cache[pulse_name]["save_components"]
    assert (
        component_name in pulse_components
    ), f"component {component_name} has not been profiled for {pulse_name}"
    fname: str = pulse_components[component_name]["file"]
    fpath: str = add_path(fname, "pulses")
    if fpath[-4:] != ".npy":
        fpath = fpath + ".npy"
    return np.load(fpath, allow_pickle=True)


def load_gate(gate_name) -> dict[str, Qobj | dict[str, np.ndarray[complex]]]:
    cache = get_cache_description("gate")
    assert gate_name in cache, f"{gate_name} has not been cached"
    fname: str = cache[gate_name]["file"]
    fpath: str = add_path(fname, "gate")
    if fname[-3:] != ".qu":
        fpath = fpath + ".qu"
    return qload(fpath)


def get_params(path: str) -> dict[str, float | dict[str, float]]:
    with open(path, "r") as stream:
        try:
            params: dict | None = yaml.safe_load(stream)
            if params is None:
                params = {}
        except yaml.YAMLError as exc:
            print(exc)
    return params


def get_ct_params() -> dict[str, float | dict[str, float]]:
    fpath = add_path("circuit_parameters.yaml", "config")
    return get_params(fpath)


def get_pulse_defaults() -> dict[str, float | dict[str, float]]:
    fpath = add_path("pulse_parameters.yaml", "config")
    return get_params(fpath)


def get_pulse_specs() -> dict[str, float | dict[str, float]]:
    fpath = add_path("pulses.yaml", "config")
    return get_params(fpath)


def get_cache_description(cache_lbl: str = "pulse") -> dict[str, DataDict]:
    assert cache_lbl in [
        "pulse",
        "gate",
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


######################### Existence Checking


def pulse_cached(pulse_desc: PulseConfig | PulseProfile) -> bool:
    desc: PulseDict = pulse_desc.as_dict()
    name: str = desc.pop("name")
    # remove profiled components from pulse description
    if "save_components" in desc:
        desc.pop("save_components")
    cache: dict[str, PulseDict] = get_cache_description("pulse")
    if name in cache:
        cached_desc: PulseDict = cache[name]
        matched_params: list[bool] = [desc[lbl] == cached_desc[lbl] for lbl in desc]
        return all(matched_params)
    else:
        return False


def is_file(path: str) -> bool:
    return isfile(path)
