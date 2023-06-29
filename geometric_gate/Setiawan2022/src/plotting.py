import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import inspect
import gate
import file_io
import simulation_setup as setup
import composite_systems as systems
import state_labeling as lbl
from simulation_setup import PulseProfile
from pulse import Pulse
from typing import Callable
from composite_systems import CompositeSystem

plt.rcParams['text.usetex'] = True
# Determine which pulses to plot
pulse_lbls = [f'CZ-{tg}ns' for tg in range(20, 30, 10)]
configs = setup.setup_sim_from_configs(pulse_lbls=pulse_lbls)
pulses = configs[1]


def get_pulse_component(pulse_builder,
                        component_name,
                        **component_kwargs):

    pulse_funcs: list[tuple[str, Callable["...", complex]]] =\
        inspect.getmembers(pulse_builder,
                           predicate=inspect.ismethod)
    pulse_funcs: dict[str, Callable['...', complex]] = dict(pulse_funcs)
    func = pulse_funcs[component_name]
    return func(**component_kwargs)


def get_populations(traj, comp_state_tensor):
    comp_states = {'ggg': comp_state_tensor[0, 0, 0],
                   'geg': comp_state_tensor[0, 1, 0],
                   'egg': comp_state_tensor[1, 0, 0],
                   'eeg': comp_state_tensor[1, 1, 0],
                   'ge1': comp_state_tensor[0, 1, 1],
                   'gf0': comp_state_tensor[0, 2, 0]}
    populations: dict[str, np.ndarray[float]] = {
        lbl: np.empty(len(traj), dtype=float)
        for lbl in comp_states
    }
    for i, psi in enumerate(traj):
        for state_lbl, array in populations.items():
            basis_state = comp_states[state_lbl]
            p = (basis_state.dag()*psi).tr()
            p *= p.conjugate()
            array[i] = abs(p)
    return populations


def get_fidelity_list(trajectories,
                      circuit,
                      comp_state_tensor,
                      comp_state_coords):
    init_states = ['ggg', 'geg', 'egg', 'eeg']
    U_ideal = file_io.load_unitary('CZ')
    trajectory_len = len(trajectories['ggg'])
    fidelity_list = np.empty(trajectory_len, dtype=float)
    for i in range(trajectory_len):
        evolved_state = {lbl: [trajectories[lbl][i]]
                         for lbl in init_states}
        _, U = gate.compute_unitary(evolved_state,
                                    comp_state_tensor,
                                    comp_state_coords)
        fidelity_list[i] = gate.compute_fidelity(U_ideal, U, circuit,
                                                 comp_state_coords)
    return fidelity_list


def plot_pulse(pulse_profile):
    pulse_arr = pulse_profile.pulse
    ct = pulse_profile.circuit
    pulse_generator = Pulse(pulse_profile.pulse_config.pulse_params,
                            ct)
    # build tlist
    tg = pulse_profile.pulse_config.pulse_params['tg']
    dt = pulse_profile.pulse_config.pulse_params['dt']
    t_ramp = pulse_profile.pulse_config.pulse_params['t_ramp']
    tlist = np.arange(0, tg + 2 * t_ramp, dt)
    # get omegas
    omega_A: np.ndarray[float] = get_pulse_component(pulse_generator,
                                                     '_Pulse__omega_A',
                                                     t=tlist)
    omega_B: np.ndarray[complex] = get_pulse_component(pulse_generator,
                                                       '_Pulse__omega_B',
                                                       t=tlist,
                                                       geo_phase=np.pi)
    # get gs
    g_ac: dict[str, np.ndarray[complex]] = \
        get_pulse_component(pulse_generator,
                            '_Pulse__g_ac',
                            t=tlist,
                            geo_phase=np.pi)
    g_ac_A = g_ac['A']
    g_ac_B = g_ac['B']

    # get delta_wmod
    delta_wmod: dict[str, np.ndarray[complex]] = \
        get_pulse_component(pulse_generator,
                            '_Pulse__delta_wmod',
                            g_ac=g_ac)
    delta_wmod_A = delta_wmod['A']
    delta_wmod_B = delta_wmod['B']

    # make plots

    # omegas
    fig_omegas, ax_omegas = plt.subplots()
    ax_omegas.plot(tlist, omega_A/(2*np.pi), label=r'$\Omega_A$')
    ax_omegas.plot(tlist, abs(omega_B)/(2*np.pi), label=r'$\Omega_B$')
    ax_omegas.legend()
    ax_omegas.set_title(r'Pulse Envelopes of the Two Drives')
    ax_omegas.set_xlabel(r'time (ns)')
    ax_omegas.set_ylabel(r'$\Omega_j/2\pi$ (MHz)')
    fname = f'{pulse_profile.name}_omegas.pdf'
    fig_omegas.savefig(f'../plots/tg_instability/{fname}')

    # delta_wmods
    fig_deltas, ax_deltas = plt.subplots()
    ax_deltas.plot(tlist, delta_wmod_A/(2*np.pi), label=r'$\delta \omega_{A}$')
    ax_deltas.plot(tlist, delta_wmod_B/(2*np.pi),
                   label=r'$|\delta \omega_{B}|$')
    ax_deltas.legend()
    ax_deltas.set_title(r'Modulation Frequency Shifts')
    ax_deltas.set_ylabel(r'$\delta \omega_{j}$ (MHz)')
    ax_deltas.set_xlabel(r'time (ns)')
    fname = f'{pulse_profile.name}_delta_wmods.pdf'
    fig_deltas.savefig(f'../plots/tg_instability/{fname}')

    # pulse
    fig_pulse, ax_pulse = plt.subplots()
    ax_pulse.plot(tlist[:-1], pulse_arr)
    ax_pulse.set_title(r'Control Pulse')
    ax_pulse.set_ylabel(r'Amplitude (MHz)')
    ax_pulse.set_xlabel(r'time (ns)')
    fname = f'{pulse_profile.name}.pdf'
    fig_pulse.savefig(f'../plots/tg_instability/{fname}')


def plot_trajectories(pulse_profile: PulseProfile):
    ct: CompositeSystem = pulse_profile.circuit
    tg = pulse_profile.pulse_config.pulse_params['tg']
    dt = pulse_profile.pulse_config.pulse_params['dt']
    t_ramp = pulse_profile.pulse_config.pulse_params['t_ramp']
    tlist = np.arange(0, tg + 2 * t_ramp, dt)
    n: qt.Qobj = ct.get_raised_op('C', ['a'],
                                  lambda a: a.dag()*a)
    comp_state_tensor, comp_state_coords = lbl.get_dressed_comp_states(ct, 3)
    init_states = ['ggg', 'egg', 'geg']
    trajectories = gate.get_trajectories(pulse_profile, ct, comp_state_tensor,
                                         n)
    populations = {init_state: get_populations(trajectories[init_state],
                                               comp_state_tensor)
                   for init_state in trajectories}
    fidelities = get_fidelity_list(trajectories,
                                   ct,
                                   comp_state_tensor,
                                   comp_state_coords)
    # make plots
    # populations
    fig_pops, axs_pops = plt.subplots(4, 1)
    for j, init_state in enumerate(populations):
        axs_pops[j].set_title(f'Initial state: {init_state}')
        for basis_state, traj in populations[init_state].items():
            axs_pops[j].plot(tlist[1:], traj, label=basis_state)
            axs_pops[j].set_xlabel('time (ns)')
            axs_pops[j].set_ylabel('population')
    axs_pops[0].legend(markerscale=0.5, fontsize='small')
    fig_pops.set_figheight(12)
    fig_pops.tight_layout()
    fname = f'{pulse_profile.name}_populations.pdf'
    fig_pops.savefig(f'../plots/tg_instability/{fname}')

    # fidelities
    fig_f, ax_f = plt.subplots()
    ax_f.plot(tlist[1:], fidelities)
    ax_f.set_xlabel('time (ns)')
    ax_f.set_ylabel('Fidelity')
    fname = f'{pulse_profile.name}_gate_fidelities.pdf'
    fig_f.savefig(f'../plots/tg_instability/{fname}')
    return None


if __name__ == '__main__':
    for pulse_lbl, pulse_profile in pulses.items():
        plot_pulse(pulse_profile)
        plot_trajectories(pulse_profile)
