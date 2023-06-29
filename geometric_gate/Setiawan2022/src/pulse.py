""" File contains functions and objects used to generate geometric pulse
 as described in Method 2 of Setiawan et. al. 2022 in the literature folder
"""
from __future__ import annotations
from functools import cache, cached_property, singledispatchmethod, singledispatch
import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import TypeAlias, Callable, Any
from collections import abc
import qutip as qt
import yaml
import os
import inspect
import static_system
from composite_systems import CompositeSystem
from state_labeling import CompTensor, CompCoord, get_dressed_comp_states
from dataclasses import dataclass
from scipy.integrate import quad

DeltaDict: TypeAlias = dict[
    str, dict[tuple[int, 3], dict[tuple[int, 3], dict[int, float]]]
]
Qobj: TypeAlias = qt.Qobj


# the following object populates with static coefficients in pulse
@dataclass
class StaticPulseAttr:
    class_idx = 0

    @staticmethod
    def __get_unique_id() -> int:
        StaticPulseAttr.class_idx += 1
        return StaticPulseAttr.class_idx

    def __init__(
        self,
        pulse_params: dict[dict[float]],
        circuit: CompositeSystem,
        n_comp_state: int = 5,
    ) -> StaticPulseAttr:
        """Class contains pulse variables that do not change over timesteps, but
        are computationally intensive to compute
        Public attributes:
           w_mod: Dictionary with keys 'A', 'B' and values corresponding to
        adjusted transition frequencies for the exited and ground states in the
        lambda configuration for fluxionia A and B. Frequencies are adjusted to
        correct for stark-shift and diabatic dynamics."""

        # give current object it's id
        self._id = StaticPulseAttr.__get_unique_id()
        # Set up circuit and state indexing
        self._ct: CompositeSystem = circuit
        self._ct.freeze()  # Markes CompositeSystem as immutable object for fast-hashing
        self._params: dict[dict[float]] = pulse_params
        self._n_comp_state: int = n_comp_state
        self._eigenstates: np.ndarray[Qobj]
        self._eigenenergies: np.ndarray[float]
        self._eigenenergies, self._eigenstates = self._ct.H.eigenstates()
        # set ground state to 0 energy for consistency
        self._eigenenergies -= self._eigenenergies[0]
        self._nlev = len(self._eigenenergies)
        self._comp_coord: CompCoord
        self._comp_states: CompTensor
        self._comp_states, self._comp_coord = get_dressed_comp_states(
            self._ct, n_comp_state
        )
        self.w_mod: dict[str, float] = {
            "A": self.trans_freq("ee0", "ge1"),
            "B": self.trans_freq("gf0", "ge1"),
        }

    def __hash__(self):
        return self._id

    @property
    def nlev(self) -> int:
        return self._nlev

    @property
    def comp_coord(self) -> CompCoord:
        return self._comp_coord

    @property
    def comp_states(self) -> CompTensor:
        return self._comp_states

    @property
    def id(self) -> int:
        return self._id

    def trans_freq(
        self, k: int | str | tuple[int, 3], l: int | str | tuple[int, 3]
    ) -> float:
        e_k = self.eigenen(k)
        e_l = self.eigenen(l)
        return e_k - e_l

    @cache
    def __str_to_tuple(self, state_lbl: str) -> tuple[int, 3]:
        assert (
            len(state_lbl) == 3
        ), f"Expected state label length 3, got {len(state_lbl)}"
        idx = [0, 0, 0]
        char_to_int: dict[str, int] = {
            "g": 0, "e": 1, "f": 2, "0": 0, "1": 1, "2": 2}
        for i, char in enumerate(state_lbl):
            idx[i] = char_to_int[char]
        return tuple(idx)

    def eigenen(self, state: str | int | tuple[int, 3]) -> float:
        """Returns the eigenenergy corresponding to a state, given \
        a unique identifier of the state.

        Parameters
        ----------
        state : str | int | tuple[int, 3]
            Unique identifier for the state.

            If str, must be a three-character string\
            with characters 'g', 'e', or 'f'. A string 'xyz' will correspond to the product\
            state |xyz>, where x,y,z are the states of fluxonium A, fluxonium B, and the \
            transmon coupler, respectively.

            If int, the input `i` will correspond to the i'th eigenstate of the \
            dressed hamiltonian.

            If 3-element integer tuple, the input `(x,y,z)` will correspond to \
            the state |xyz>.

        Returns
        -------
        float
            Eigenvalue corresponding to the given eigenstate input
        """

        idx = self.state_idx(state)
        return self._eigenenergies[idx]

    def state(self, state_idx: str | tuple[int, 3] | int) -> Qobj:
        """Returns the eigenstate vector given a unique identifier of \
        the eigenstate.

        Parameters
        ----------
        state_idx : str | tuple[int, 3] | int
            Unique identifier for the state.

            If str, must be a three-character string\
            with characters 'g', 'e', or 'f'. A string 'xyz' will correspond to the product\
            state |xyz>, where x,y,z are the states of fluxonium A, fluxonium B, and the \
            transmon coupler, respectively.

            If int, the input `i` will correspond to the i'th eigenstate of the \
            dressed hamiltonian.

            If 3-element integer tuple, the input `(x,y,z)` will correspond to \
            the state |xyz>.

        Returns
        -------
        Qobj
           Eigenstate vector in the bare product basis.
        """

        if isinstance(state_idx, str):
            state_idx = self.__str_to_tuple(state_idx)
        if isinstance(state_idx, int):
            return self._eigenstates[state_idx]
        else:
            k, l, m = state_idx
            return self._comp_states[k, l, m]

    def state_idx(self, state: str | tuple[int, 3] | Qobj | int) -> int:
        """Index of dressed eigenstate corresponding to computational\
        state in bare product basis.

       Given a unique identifier of a bare computational state, returns the \
       index of the corresponding dressed eigenstate.

        Parameters
        ----------
        state : str | tuple[int, 3] | Qobj | int
            If str, must be a three-character string\
            with characters 'g', 'e', or 'f'. A string 'xyz' will correspond to the product\
            state |xyz>, where x,y,z are the states of fluxonium A, fluxonium B, and the \
            transmon coupler, respectively.

            If 3-element integer tuple, the input `(x,y,z)` will correspond to \
            the state |xyz>.

        Returns
        -------
        int
           Index of the dressed eigenstate corresponding to the input, where the \
        eigenstates are ordered by increasing eigenenergy.
        """

        idx = state  # idx dummy variable to ensure no side effects on state
        if isinstance(idx, str):
            n, m, k = self.__str_to_tuple(state)
            idx: int = self._comp_coord[n, m, k]
        elif isinstance(state, abc.Iterable):
            n, m, k = state
            idx: int = self._comp_coord[n, m, k]
        elif isinstance(idx, Qobj):
            idx: int = np.array(
                [qt.isequal(idx, eig) for eig in self._eigenstates]
            ).nonzero()[0][0]
        return idx

    @cache
    def mod_trans_detunings(
        self,
        sys: str,
        s1: tuple[int, 3] | str | int,
        s2: tuple[int, 3] | str | int,
        pm: int,
    ) -> float:
        """Class method generates the detuning of the transition |s1> <--> |s2> from\
        the modulation tone applied to fluxonium sys, as seen in the capital delta term
        in (C2). So, for parameters `(sys, s1, s2, pm)`, function will return\
              Delta^sys_{s1,s2,pm} in equation (C2).

        Args:
            sys (str): A or B. Corresponds to fluxonium A or B.
            s1 (tuple[int,3]): Computational state 1 involved in transition |s2> -> |s1>, indexed\
             in the bare product basis
            s2 (tuple[int,3]): Computational state 2 involved in transition |s2> -> |s1>, indexed\
             in the bare product basis
            pm (int): +1 or -1. Corresponds to sigma index in equation C2

        Returns:
            float: Detuning of transition |s1> <--> |s2> from modulation tone applied to fluxonium sys.
        """
        assert pm in [-1, 1], f"parameter `pm` must be 1 or -1, got {pm}"
        assert sys in ["A", "B"], f"parameter sys must be A or B, got {sys}"
        if isinstance(s1, str):
            s1 = self.__str_to_tuple(s1)
        if isinstance(s2, str):
            s2 = self.__str_to_tuple(s2)
        w_mod: float = self.w_mod[sys]
        return self.trans_freq(s1, s2) + pm * w_mod


# begin pulse def
class Pulse:
    """
    Generator for pulses implementing a geometric C-phase gate \
    with phase gamma.

    Public Attributes
    -----------------
    n_comp_states: int
        Number of eigenstates considered when generating the pulse.

    Public Methods
    --------------

    """

    def __init__(
        self,
        pulse_params: dict[dict[float]],
        circuit: CompositeSystem,
        dt: float | None = None,
        t_ramp: float | None = None,
        tg: float | None = None,
        n_comp_states: int = 5,
    ):
        """Class constructor.

        Parameters
        ----------
        pulse_params : dict[dict[float]]
            Dictionary of pulse parameters as described in docstrings in
            gen_configs.py.
        circuit : CompositeSystem
            Circuit used to generate pulse. Pulse design assumes the gate \
            will be implemented on a circuit identical to this parameter.
        dt : float, optional
            Timestep size to use in integration. If not provided, must be \
            provided in `pulse_params`. NOTE: This is not the same as the \
            timestep size of the discrete pulse itself.
        t_ramp : float, optional
            Duration of cosine ramp added to beggining and end of pulse, \
        in ns. If not provided, must be provided in `pulse_params`.
        tg : float, optional
            Gate duration in ns. Does not include ramp time.
        n_comp_states : int
            Number of energy eigenstates to consider in pulse generation. \
            defaults to 5.
        """

        self.static_attr: StaticPulseAttr = StaticPulseAttr(
            pulse_params, circuit, n_comp_states
        )
        self._ct: CompositeSystem = circuit
        if dt is None:
            self._dt: float = pulse_params["dt"]
        else:
            self._dt: float = dt
        if t_ramp is not None:
            self._t_ramp = t_ramp
        else:
            self._t_ramp = pulse_params["t_ramp"]
        if tg is None:
            self._tg = pulse_params["tg"]
        else:
            self._tg = tg
        self._params: dict[str, dict[str, float] | float] = pulse_params
        if "n_comp_states" in pulse_params:
            self.n_comp_states: int = pulse_params["n_comp_states"]
        else:
            self.n_comp_states: int = n_comp_states
        self._omega_0: float = 2 * np.pi * \
            self._params["omega_0"] / pulse_params["tg"]

    def __omega_A(self, t: np.ndarray[float]) -> np.ndarray[float]:
        ramp_up_t = t[t < self._t_ramp]
        ramp_down_t = t[self._tg + self._t_ramp < t]
        pulse_t = t[(self._t_ramp <= t) & (t <= self._tg + self._t_ramp)]
        t_ = pulse_t - self._t_ramp

        ramp_up = np.zeros(shape=ramp_up_t.shape, dtype=float)
        ramp_down = np.zeros(shape=ramp_down_t.shape, dtype=float)

        theta = self.__theta(t_, 0)
        dtheta = self.__theta(t_, 1)
        d2theta = self.__theta(t_, 2)
        pulse_env = np.sin(theta) + 4 * np.cos(theta) * d2theta / (
            self._omega_0**2 + 4 * dtheta**2
        )
        omega_A: np.ndarray[float] = np.concatenate(
            (ramp_up, self._omega_0 * pulse_env, ramp_down)
        )
        return omega_A

    def __omega_B(self, t: np.ndarray[float], geo_phase: float) -> np.ndarray[complex]:
        ramp_up_t = t[t < self._t_ramp]
        ramp_down_t = t[self._tg + self._t_ramp < t] - \
            (self._tg + self._t_ramp)
        pulse_t = t[(self._t_ramp <= t) & (t <= self._tg + self._t_ramp)]
        t_ = pulse_t - self._t_ramp
        theta = self.__theta(t_, 0)
        dtheta = self.__theta(t_, 1)
        d2theta = self.__theta(t_, 2)
        phase = np.exp(1.0j * self.__phase_arg(t_, geo_phase))
        pulse_env = np.cos(theta) - 4 * np.sin(theta) * d2theta / (
            self._omega_0**2 + 4 * dtheta**2
        )
        omega_B: np.ndarray[float] = self._omega_0 * phase * pulse_env
        ramp_up: np.ndarray[float] = omega_B[0] * (
            1 - np.cos(ramp_up_t * np.pi / (self._t_ramp * 2))
        )
        ramp_down: np.ndarray[float] = omega_B[-1] * (
            np.cos(ramp_down_t * np.pi / (self._t_ramp * 2))
        )
        return np.concatenate((ramp_up, omega_B, ramp_down))

    def __phase_arg(self, t: np.ndarray[float], geo_phase: float) -> float:
        tg = self._params["tg"]
        return geo_phase * np.heaviside(t - tg / 2, 1)

    def __theta(self, t: np.ndarray[float], deriv: int = 0) -> float:
        # implements eq A2
        tg = self._params["tg"]
        interval_1 = t[t <= tg / 2]
        interval_2 = t[(tg / 2 < t) & (t <= tg)]
        theta_interval_1 = self.__theta_interval_1(interval_1, deriv)
        theta_interval_2 = self.__theta_interval_2(interval_2, deriv)
        thetas = np.concatenate((theta_interval_1, theta_interval_2))
        return thetas

    def __theta_interval_1(self, t: np.ndarray[float], d: int) -> float:
        tg = self._params["tg"]
        derivs: list[float] = [np.pi / 2,
                               np.pi / (2 * tg), np.pi / (2 * tg**2)]
        return derivs[d] * self.__polynom(t / tg, d=d)

    def __theta_interval_2(self, t: np.ndarray[float], d: int) -> float:
        tg = self._params["tg"]
        derivs: list[float] = [
            (np.pi / 2), (np.pi / (2 * tg)), (np.pi / (2 * tg**2))]
        if d == 0:
            return derivs[d] * (1 - self.__polynom(t / tg - 1 / 2, d=d))
        else:
            return -1 * derivs[d] * self.__polynom(t / tg - 1 / 2, d=d)

    def __polynom(self, t: float, d: int = 0) -> float:
        # implements eq A3
        derivs: list[tuple[float, 6]] = [
            (6 * 2**5, -15 * 2**4, 10 * 2**3, 5, 4, 3),
            (960, -960, 240, 4, 3, 2),
            (3840, -2880, 480, 3, 2, 1),
        ]
        return self.__p(t, *derivs[d])

    def __p(
        self, x: float, c1: float, c2: float, c3: float, e1: float, e2: float, e3: float
    ) -> float:
        return c1 * x**e1 + c2 * x**e2 + c3 * x**e3

    def __g_ac(self, t: np.ndarray[float], geo_phase: float) -> dict[str, float]:
        ge1_idx: int = self.static_attr.state_idx("ge1")
        ee0_idx: int = self.static_attr.state_idx("ee0")
        gf0_idx: int = self.static_attr.state_idx("gf0")
        return {
            "A": self.__omega_A(t) / (self.__adag_a[ge1_idx, ee0_idx]),
            "B": self.__omega_B(t, geo_phase) / (self.__adag_a[ge1_idx, gf0_idx]),
        }

    @cached_property
    def highest_leakage_st(self) -> int:
        leakage_states_idxes = [
            self.static_attr.comp_coord[i, j, k]
            for i in range(self.n_comp_states)
            for j in range(self.n_comp_states)
            for k in range(self.n_comp_states)
        ]
        return max(leakage_states_idxes)

    # @cache
    def __delta_ek(
        self,
        g_ac: dict[str, np.ndarray[float | complex]],
        state: tuple[int, 3] | str,
    ) -> np.ndarray[float]:
        # implements eq C2b
        leakage_states: list[int] = list(range(self.highest_leakage_st + 1))
        sum_c2b = 0
        for l in leakage_states:
            for flux_lbl in ["A", "B"]:
                for sgn in [-1, 1]:
                    sum_c2b += self.__C2b__summand(
                        g_ac[flux_lbl], state, l, flux_lbl, sgn
                    )
        return sum_c2b

    def __C2b__summand(
        self,
        g_ac: np.ndarray[float | complex],
        k: tuple[int, 3] | str | int,
        l: int,
        flux_lbl: str,
        sgn: int,
    ) -> float:
        """Function implements summand in equation C2b

        Args:
            t (float): current timestep
            k (tuple[int,3] | str | int): Label for state k in eq C2b
            l (int): Label for leakeage state l of total hamiltonian iterated over in eq C2b
            flux_lbl (str): A or B, label for state j in eq C2b
            sgn (int): +1 or -1, sigma in eq C2b

        Returns:
            float: Summand term index (j,sigma,l) in C2b
        """
        denominator: float = 4 * self.static_attr.mod_trans_detunings(
            flux_lbl, k, l, sgn
        )
        if abs(denominator) <= 1e-3:
            return 0
        k_idx: int = self.static_attr.state_idx(k)
        adag_a: np.ndarray = (
            self._adag_a_as_matrix
        )  # see next function definition for explanation of adag_a
        k_adaga_l: complex = adag_a[k_idx][l]
        numerator: float = (g_ac * k_adaga_l).conjugate() * (g_ac * k_adaga_l)
        return numerator / denominator

    @cached_property
    def __adag_a(self) -> Qobj:
        """this function is used only for optimization of bottleneck observed in
        profiling. Returns operator a.dag()*a in eigenbasis where a is the QHO destruction operator
        acting on the transmon coupler
        """
        adag_a: Qobj = self._ct.get_raised_op(
            "C", ["a"], lambda a: a.dag() * a)
        eigenbasis: np.array[Qobj] = self._ct.H.eigenstates()[1]
        adag_a = adag_a.transform(eigenbasis)
        return adag_a

    @cached_property
    def _adag_a_as_matrix(self) -> np.ndarray:
        """this function is used only for optimization of bottleneck observed in
        profiling. Returns full numpy matrix for operator a.dag()*a in eigenbasis where a is the QHO destruction operator
        acting on the transmon coupler
        """
        return self.__adag_a.full()

    def __delta_wmod(
        self, g_ac: dict[str, np.ndarray[complex | float]]
    ) -> dict[str, np.ndarray[float]]:
        delta_ge1: np.ndarray[float] = self.__delta_ek(g_ac, "ge1")
        delta_ee0: np.ndarray[float] = self.__delta_ek(g_ac, "ee0")
        delta_gf0: np.ndarray[float] = self.__delta_ek(g_ac, "gf0")
        return {"A": delta_ge1 - delta_ee0, "B": delta_ge1 - delta_gf0}

    def __w_mod(
        self, g_ac: dict[str, np.ndarray[complex | float]]
    ) -> dict[str, np.ndarray[float]]:
        deltas: dict[str, np.ndarray[float]] = self.__delta_wmod(g_ac)
        return {
            "A": self.static_attr.w_mod["A"] + deltas["A"],
            "B": self.static_attr.w_mod["B"] + deltas["B"],
        }

    def build_pulse(self, tlist: np.ndarray[float], geo_phase: float) -> np.ndarray[float]:
        """Builds the pulse applied to the transmon coupler described by eq (31)

        Given an array of timesteps and a phase angle, returns a pulse that when \
        applied to two fluxonia (A,B) coupled by a transmon, will implement a C-phase \
        gate, where the computational states are as given in Setiawan et. al.

        Parameters
        ----------
        tlist : np.ndarray[float]
            Timestep list. For each t in tlist, the pulse at time t will \
            be returned.
        geo_phase : float
            Geometric phase to print on fluxonium B, if fluxonium A is in the first \
        excited state.

        Returns
        -------
        np.ndarray[float]
            Value of the pulse at each time `t` in `tlist`

        """

        d_wC: np.ndarray[float] | int = 0
        gs: dict[str, np.ndarray[float]] = self.__g_ac(tlist, geo_phase)
        w_mods: dict[str, np.ndarray[float]] = self.__w_mod(gs)
        for flux in ["A", "B"]:
            exp_arg = 1.0j * cumulative_trapezoid(w_mods[flux], tlist)
            exp_val = np.exp(exp_arg)
            exp_term = gs[flux][1:] * exp_val
            d_wC += np.real(exp_term)
        return d_wC

    # Diagnostic functions
    def get_integrand_func(
        self, tlist: np.ndarray[int], state: str, geo_phase: float
    ) -> np.ndarray[float]:
        res_lst = self.__delta_ek(tlist, state, geo_phase)
        return res_lst


# The following section is used to add pulses to the
# `../pulses/` directory. To add a new pulse, write the
# parameters of the new pulse to `../pulses/pulses.yaml`
# as specified in that file's documentation. Then, run
# this file as a script from this directory `$python pulse.py`


# extract configuration parameters
def get_params(path: str) -> dict[str, float | dict[str, float]]:
    with open(path, "r") as stream:
        try:
            params: dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


# the following raises an error during profiling, but is not used
# by the profile, so we wrap in a try-catch
try:
    # define default constants
    default_p_params: dict[str, float | dict[str, float]] = get_params(
        "../config/pulse_parameters.yaml"
    )
    default_tg: float = default_p_params["tg"]
    default_amp: float = default_p_params["omega_0"]
except FileNotFoundError:
    default_p_params = {}
    default_tg = 0
    default_amp = 0


def build_npy_fname(
    gate_lbl: str,
    tg: float = default_tg,
    amp: float = default_amp,
    dt: float = 1e-2,
    include_dir: bool = True,
) -> str:
    fname = f"{gate_lbl}_{tg}ns_{amp}GHz_{dt}dt.npy"
    if include_dir:
        fname = "../output/pulses/" + fname
    return fname


# define funcs
def is_prebuilt(
    gate_label: str | int,
    tg: str | int = default_tg,
    amp: float | int = default_amp,
    dt=1e-2,
) -> bool:
    prebuilt_pulses = os.listdir("../output/pulses")
    fname: str = build_npy_fname(gate_label, tg, amp, dt, include_dir=False)
    # fname:str = f'{geo_phase}_{tg}ns_{dt}dt_{amp}GHz_pulse.npy'
    return fname in prebuilt_pulses


def build_pulse(
    gate_label: str,
    geo_phase: float,
    circuit: CompositeSystem,
    tg: float = default_tg,
    omega_0: float = default_amp,
    dt: float = 0.01,
    save_spec=False,
    **kwargs,
) -> np.ndarray[float]:
    fname = build_npy_fname(gate_label, tg, omega_0, dt)
    if is_prebuilt(gate_label, tg, omega_0, dt) and "save_components" not in kwargs:
        return np.load(fname)
    else:
        pulse_params: dict[str, float | dict[str, float]] = {
            **default_p_params,
            "tg": tg,
            "omega_0": omega_0,
        }
        tlist = np.arange(0, tg, dt)
        pulse_generator: Pulse = Pulse(
            pulse_params=pulse_params, circuit=circuit, dt=dt
        )
        pulse_spec = {
            gate_label: {"geo_phase": geo_phase,
                         "tg": tg, "omega_0": omega_0, "dt": dt}
        }
        if save_spec:
            write_pulse_spec(pulse_spec)
        if "save_components" in kwargs:
            save_pulse_components(
                pulse_generator, gate_label, pulse_spec[gate_label], **kwargs
            )
        if is_prebuilt(gate_label, tg, omega_0, dt):
            return np.load(fname)
        else:
            return pulse_generator.build_pulse(tlist, geo_phase, fname)


def write_pulse_spec(pulse_spec: dict[str, dict[str, float]]) -> None:
    current_specs = get_params("../config/pulses.yaml")
    updated_specs = {**current_specs, **pulse_spec}
    with open("../config/pulses.yaml", "w") as yfile:
        try:
            yaml.safe_dump(updated_specs, yfile)
        except yaml.YAMLError as exc:
            print(exc)


def build_pulses(pulses_params: list[dict[str, float | str]] | None = None) -> None:
    """function checks that all pulses specified in \
    `../config/pulses.yaml` are written to the `../output/pulses`
    directory as a .npy file. If an entry in pulses.yaml is
    found that does not have a corresponding .npy file, that
    pulse will be generated and it's .npy file written.
    """
    if pulses_params is None:
        with open("../config/pulses.yaml", "r") as stream:
            try:
                pulse_specs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        for gate_lbl, spec in pulse_specs.items():
            build_pulse(gate_label=gate_lbl, **spec)
    else:
        return list([build_pulse(**pulse_params) for pulse_params in pulses_params])


def profile_pulse_component(
    func: Callable["...", float | complex],
    tlist: np.ndarray[float],
    **kwargs: dict[str, float],
) -> np.ndarray[Any]:
    result_array = np.array([func(t=t, **kwargs) for t in tlist])
    return result_array


def save_pulse_component(
    fname: str,
    gate_lbl: str,
    tg: float,
    omega_0: float,
    dt: float,
    func: Callable["...", float | complex],
    **kwargs: dict[str, float],
) -> None:
    if is_prebuilt(f"{gate_lbl}_{fname}", tg, omega_0, dt):
        return None
    else:
        tlist = np.arange(0, tg, dt)
        result_array = profile_pulse_component(func, tlist, **kwargs)
        fname = build_npy_fname(f"{gate_lbl}_{fname}", tg, omega_0, dt)
        np.save(fname, result_array)


def save_pulse_components(
    pulse_generator: Pulse, gate_label: str, pulse_spec: dict[str, float], **kwargs
):
    pulse_components = kwargs["save_components"]
    pulse_generator_methods = inspect.getmembers(
        pulse_generator, predicate=inspect.ismethod
    )
    pulse_generator_methods = dict(pulse_generator_methods)
    for component_method, component_method_kwargs in pulse_components.items():
        if component_method_kwargs is None:
            component_method_kwargs = {}
        func: Callable["...", Any] = pulse_generator_methods[component_method]
        tg = pulse_spec["tg"]
        omega_0 = pulse_spec["omega_0"]
        dt = pulse_spec["dt"]
        save_pulse_component(
            fname=component_method,
            gate_lbl=gate_label,
            tg=tg,
            omega_0=omega_0,
            dt=dt,
            func=func,
            **component_method_kwargs,
        )


if __name__ == "__main__":
    pass
# build_pulses()
