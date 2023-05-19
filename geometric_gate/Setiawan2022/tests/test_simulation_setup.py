#!/usr/bin/env python3
"""Unit tests for setup_sim.py"""


import yaml
import numpy as np
import qutip as qt
DEBUG = True
# package imports

# module imports
# set up path
if DEBUG:
    import sys
    import os
    from pathlib import Path

    project_root = Path(__file__).parents[1]

    # add src and test modules to path
    sys.path.append(os.path.join(project_root, "src"))
    sys.path.append(os.path.join(project_root, "./tests"))
    sys.path.append("./src")
    ct_path = os.path.join(project_root, "config/circuit_parameters.yaml")
    pulse_defaults_path = os.path.join(project_root,
                                       "config/pulse_parameters.yaml")
    gate_param_path = os.path.join(project_root, "config/pulses.yaml")
elif __name__ == "__main__":
    import sys

    sys.path.append("../src")
    ct_path = "../config/circuit_parameters.yaml"
    pulse_defaults_path = "../config/pulse_parameters.yaml"
    gate_param_path = "../config/pulses.yaml"
else:
    ct_path = "./config/circuit_parameters.yaml"
    pulse_defaults_path = "./config/pulse_parameters.yaml"
    gate_param_path = "./config/pulses.yaml"
if 1:  # hacky, but prevent blacken from moving this to top of file
    import simulation_setup as sim_setup
    from composite_systems import CompositeSystem
    from static_system import build_static_system
    from file_io import get_params

# import circuit and pulse parameters for handrolling
ct_params = get_params(ct_path)
breakpoint()
pulse_defaults = get_params(pulse_defaults_path)
gate_params = get_params(gate_param_path)


def test_collect_sim_params_from_configs():
    sim_setup.collect_sim_params_from_configs()


def test_setup_circuit():
    config = sim_setup.collect_sim_params_from_configs()
    ct: CompositeSystem = sim_setup.setup_circuit(config.ct_params)
    ct_handrolled = build_static_system(ct_params)
    assert (
        ct == ct_handrolled
    ), "setup_circuit returned circuit with different properties than expected"


def test_setup_pulse_params():
    config = sim_setup.collect_sim_params_from_configs()
    cz_pulse_config = config.pulse_config_dict["CZ"]
    default_pulse_config = config.default_pulse_params
    name = "CZ"
    ct_params = config.ct_params
    sim_setup.setup_pulse_params(
        name, cz_pulse_config, default_pulse_config, ct_params)


def test_setup_pulse_param_dict():
    config = sim_setup.collect_sim_params_from_configs()
    sim_setup.setup_pulse_param_dict(
        config.pulse_config_dict, config.default_pulse_params, config.ct_params
    )


def test_get_pulse_profile():
    config = sim_setup.collect_sim_params_from_configs()
    pulse_params = sim_setup.setup_pulse_param_dict(
        config.pulse_config_dict, config.default_pulse_params, config.ct_params
    )
    ct: CompositeSystem = sim_setup.setup_circuit(config.ct_params)
    sim_setup.get_pulse_profile(pulse_params["CZ"], ct)


def test_setup_sim_from_configs():
    sim_setup.setup_sim_from_configs()


if DEBUG:
    test_get_pulse_profile()
