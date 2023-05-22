import file_io
import numpy as np
import qutip as qt
import pytest
from composite_systems import CompositeSystem
from state_labeling import get_dressed_comp_states
from static_system import build_static_system, build_bare_system
from simulation_setup import setup_sim_from_configs, PulseProfile
import gate
from gate import GateProfile
from math import sqrt

DEBUG = False
if DEBUG:
    from pathlib import Path
    import os
    import sys

    project_root = Path(__file__).parents[1]
    os.chdir(project_root)
    sys.path.append(os.path.join(project_root, "./src"))
ct_params: dict[str, float | dict[str, float]] = file_io.get_params(
    "./config/circuit_parameters.yaml"
)
circuit: CompositeSystem = build_static_system(ct_params["pulse_gen_ct"])
n_comp_states = 3
comp_states, comp_idxs = get_dressed_comp_states(circuit, n_comp_states)


def test_spec_to_state():
    str_spec = "geg"
    tup_spec = ("+xg", {"egg": 1 / sqrt(2), "ggg": 1 / sqrt(2)})
    gate.spec_to_state(str_spec, comp_states)
    gate.spec_to_state(tup_spec, comp_states)


def test_assemble_init_states():
    config, pulse_dict = setup_sim_from_configs()
    pulse_config = pulse_dict["CZ"]
    gate.assemble_init_states(pulse_config, circuit, comp_states)


def test_assemble_empty_profile():
    config, pulse_dict = setup_sim_from_configs()
    pulse_profile: PulseProfile = pulse_dict["CZ"]
    # test with no params given other than name and pulse
    gate_profile: GateProfile = gate.assemble_empty_profile(
        "CZ", pulse_profile)
    err_msg = ill_defined_gate_profile(gate_profile)
    assert not err_msg, err_msg

    # test with circuit given
    bare_circuit = build_bare_system(ct_params["pulse_gen_ct"])
    gate_profile: GateProfile = gate.assemble_empty_profile(
        "CZ", pulse_profile, bare_circuit, "foobar"
    )
    err_msg = ill_defined_gate_profile(gate_profile)
    assert not err_msg, err_msg
    assert not (
        pulse_profile.circuit == gate_profile.circuit
    ), "gate profile not assembled with provided circuit"
    assert (
        gate_profile.circuit_config == "foobar"
    ), "gate profile not assembled with provided circuit config"
    name = "CZ"
    with pytest.warns(
        UserWarning,
        match=f"Circuit initialized for gate {name} but no circuit\
                 configuration dict given, caching may fail",
    ):
        gate.assemble_empty_profile(name, pulse_profile, bare_circuit)
    with pytest.raises(AssertionError):
        pulse_profile.circuit = None
        pulse_profile.pulse_config.circuit_config = None
        gate.assemble_empty_profile(name, pulse_profile)


def test_get_trajectories():
    config, pulse_dict = setup_sim_from_configs()
    pulse_profile: PulseProfile = pulse_dict["CZ"]
    gate_profile: GateProfile = gate.assemble_empty_profile(
        "CZ", pulse_profile)
    ct: CompositeSystem = gate_profile.circuit
    comp_states, _ = get_dressed_comp_states(ct)
    gate.get_trajectories(pulse_profile, ct, comp_states)


def test_profile_gate():
    config, pulse_dict = setup_sim_from_configs()
    gate.profile_gate("CZ", pulse_dict["CZ"])


def ill_defined_gate_profile(gate_profile: GateProfile) -> str:
    """Returns empty string if gate_profile has attributes
    of correct type. Otherwise returns a message describing
    the wrongly-typed attributes"""
    msg = ""
    if gate_profile.circuit is None:
        msg += "Gate has no circuit defined \n"
    if gate_profile.circuit_config is None:
        msg += "Gate has no circuit config\n"
    if not isinstance(gate_profile.pulse, np.ndarray):
        msg += f"Expected gate pulse to be np.ndarray[float],\
         got {type(gate_profile.pulse)} instead.\n"
    if not isinstance(gate_profile.pulse_profile, PulseProfile):
        msg += f"Expected gate pulse profile to be of type PulseProfile, \
        got {type(gate_profile.pulse_profile)} instead"
    return msg
