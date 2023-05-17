#!/usr/bin/env python

from __future__ import annotations
import qutip as qt
import numpy as np
import file_io
import hashing
import simulation_setup
from simulation_setup import PulseDict, CircuitParams
from simulation_setup import PulseParams, PulseProfile
from simulation_setup import PulseConfig
from dataclasses import dataclass, field
from pulse import Pulse, StaticPulseAttr
from static_system import build_bare_system
from composite_systems import CompositeSystem, Qobj
from state_labeling import get_dressed_comp_states
from typing import TypeAlias, TypeVar, Any, Generic

GateDict: TypeAlias = dict[str, str | PulseDict | CircuitParams | bool | list[str]]

@dataclass
class GateProfile(Generic[CircuitParams, PulseConfig, PulseParams]):
    """Simple container class for gate data."""

    def __init__(self,
                 name: str,
                 pulse_profile: PulseProfile,
                 gate_unitary: Qobj | None = None,
                 gate_circuit: CompositeSystem | None = None,
                 gate_circuit_config: CircuitParams | None = None,
                 trajectories: dict[str, qt.solver.Result] = {},
                 fidelity: float | None = None) -> GateProfile:
        self.name: str = name
        self.pulse: np.ndarray[float] = pulse_profile.pulse
        self.pulse_profile: PulseProfile = pulse_profile
        self.unitary: Qobj | None = gate_unitary
        if gate_circuit is None:
            self.gate_circuit: Qobj = pulse_profile.circuit
        else:
            self.gate_circuit: Qobj = gate_circuit
        if gate_circuit_config is None:
            self.gate_circuit_config: CircuitParams =\
                pulse_profile.pulse_config.circuit_config
        else:
            self.gate_circuit_config: CircuitParams = gate_circuit_config
        self.trajectories: dict[str, qt.solver.Result] = trajectories
        self.fidelity: float = fidelity

    def as_dict(self) -> GateDict:
        desc: GateDict = {}  # init descriptor dictionary
        desc['name'] = self.name
        desc['pulse_config'] = self.pulse_profile.as_dict()
        desc['computed_unitary'] = self.gate_unitary is not None
        desc['circuit_config'] = self.gate_circuit_config
        desc['trajectories'] = [init_state for init_state in self.trajectories]
        desc['fidelity'] = self.fidelity
        return desc

    def __hash__(self) -> int:
        return hashing.hash_dict(self.as_dict())
