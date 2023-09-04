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
import sys

plt.rcParams['text.usetex'] = True
# Determine which pulses to plot
pulse_lbls = [f'CZ-{tg}ns' for tg in [45, 90, 100, 200, 300, 400]]
pulse_lbls.insert(0, 'ZERO-400ns')
print('Setting up sim')
configs = setup.setup_sim_from_configs(pulse_lbls=pulse_lbls,
                                       cache_results=False,
                                       use_pulse_cache=False)
print('Done. Running Simulations', '#'*10)
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


def get_populations(traj, comp_state_tensor, step=10):
    comp_states = {'ggg': comp_state_tensor[0, 0, 0],
                   'geg': comp_state_tensor[0, 1, 0],
                   'egg': comp_state_tensor[1, 0, 0],
                   'eeg': comp_state_tensor[1, 1, 0],
                   'ge1': comp_state_tensor[0, 1, 1],
                   'gf0': comp_state_tensor[0, 2, 0]}
    traj_len = len(traj)
    populations: dict[str, np.ndarray[float]] = {
        lbl: np.empty(len(traj)//step, dtype=float)
        for lbl in comp_states
    }
    phases: dict[str, np.ndarray[float]] = {
        lbl: np.empty(len(traj)//step, dtype=float)
        for lbl in comp_states
    }
    for i in range(0, traj_len//step):
        psi = traj[i*step]
        for state_lbl, array in populations.items():
            phi_array = phases[state_lbl]
            basis_state = comp_states[state_lbl]
            p = (basis_state.dag()*psi).tr()
            phi = np.angle(p)
            p *= p.conjugate()
            array[i] = abs(p)
            phi_array[i] = phi
    return populations, phases


def get_phase(U):
    diag = U.diag()
    diag_phase = np.angle(diag)
    for i, phase in enumerate(diag_phase):
        if phase < 0:
            diag_phase[i] = phase + 2*np.pi
    return diag_phase


def get_mag_error(U):
    diag = U.diag()
    diag_mag = np.abs(diag)
    mag_error = diag_mag - np.array([1, 1, 1, 1])
    return mag_error


def get_gate_stats(trajectories,
                   circuit,
                   comp_state_tensor,
                   comp_state_coords,
                   step=100):
    init_states = ['ggg', 'geg', 'egg', 'eeg']
    U_ideal = file_io.load_unitary('CZ')
    trajectory_len = len(trajectories['ggg'])//step
    fidelity_list = np.empty(trajectory_len, dtype=float)
    phase_list = np.empty((trajectory_len, 4), dtype=float)
    mag_error_list = np.empty((trajectory_len, 4), dtype=float)
    U_list = np.empty((trajectory_len, 4, 4), dtype=complex)
    for i in range(trajectory_len):
        evolved_state = {lbl: [trajectories[lbl][i*step]]
                         for lbl in init_states}
        _, U = gate.compute_unitary(evolved_state,
                                    comp_state_tensor,
                                    comp_state_coords)
        phase_list[i] = get_phase(U)
        mag_error_list[i] = get_mag_error(U)
        fidelity_list[i] = gate.compute_fidelity(U_ideal, U, circuit,
                                                 comp_state_coords)
        U_list[i] = U.full()
    return fidelity_list, mag_error_list, phase_list, U_list


def plot_pulse(pulse_profile,
               exp_lbl=''):
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

    # get delta_wmod
    delta_wmod: dict[str, np.ndarray[complex]] = \
        get_pulse_component(pulse_generator,
                            '_Pulse__delta_wmod',
                            g_ac=g_ac)
    delta_wmod_A = delta_wmod['A']
    delta_wmod_B = delta_wmod['B']
    n: qt.Qobj = ct.get_raised_op('C', ['a'],
                                  lambda a: a.dag()*a)
    comp_state_tensor, comp_state_coords = lbl.get_dressed_comp_states(ct, 3)

    # make plots

    # omegas
    fig_omegas, ax_omegas = plt.subplots()
    ax_omegas.plot(tlist, omega_A/(2*np.pi), label=r'$\Omega_A$')
    ax_omegas.plot(tlist, omega_B/(2*np.pi), label=r'$\Omega_B$')
    ax_omegas.legend()
    ax_omegas.set_title(r'Pulse Envelopes of the Two Drives')
    ax_omegas.set_xlabel(r'time (ns)')
    ax_omegas.set_ylabel(r'$\Omega_j/2\pi$ (MHz)')
    fname = f'{exp_lbl}_{pulse_profile.name}_omegas.pdf'
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
    fname = f'{exp_lbl}_{pulse_profile.name}_delta_wmods.pdf'
    fig_deltas.savefig(f'../plots/tg_instability/{fname}')

    # pulse
    fig_pulse, ax_pulse = plt.subplots()
    ax_pulse.plot(tlist[:-1], pulse_arr)
    ax_pulse.set_title(r'Control Pulse')
    ax_pulse.set_ylabel(r'Amplitude (MHz)')
    ax_pulse.set_xlabel(r'time (ns)')
    fname = f'{exp_lbl}_{pulse_profile.name}.pdf'
    fig_pulse.savefig(f'../plots/tg_instability/{fname}')
    plt.close('all')


def get_eigenens(H, H_d, pulse, comp_state_coords):
    eigenens: dict[str, np.ndarray[float]] = {
        'ggg': np.empty(len(pulse)),
        'egg': np.empty(len(pulse)),
        'geg': np.empty(len(pulse)),
        'eeg': np.empty(len(pulse)),
        'ge1': np.empty(len(pulse)),
        'gf0': np.empty(len(pulse))
    }
    for i, p in enumerate(pulse):
        H_eff = H + p*H_d
        en = H_eff.eigenenergies()
        en = en - en[0]  # set zero of energy
        for state, arr in eigenens.items():
            n, m, k = gate.str_to_tuple(state)
            idx = comp_state_coords[n, m, k]
            arr[i] = en[idx]
    return eigenens


def plot_trajectories(pulse_profile: PulseProfile,
                      exp_lbl='',
                      step=100):
    ct: CompositeSystem = pulse_profile.circuit
    tg = pulse_profile.pulse_config.pulse_params['tg']
    dt = pulse_profile.pulse_config.pulse_params['dt']
    t_ramp = pulse_profile.pulse_config.pulse_params['t_ramp']
    tlist = np.arange(0, tg + 2 * t_ramp, dt)
    tlist = tlist[::step]
    n: qt.Qobj = ct.get_raised_op('C', ['a'],
                                  lambda a: a.dag()*a)
    comp_state_tensor, comp_state_coords = lbl.get_dressed_comp_states(ct, 3)
    eigenens = get_eigenens(
        ct.H, n, pulse_profile.pulse[::step], comp_state_coords)
    trajectories = gate.get_trajectories(pulse_profile, ct, comp_state_tensor,
                                         n)
    populations = {}
    phases = {}
    for init_state in trajectories:
        pops, phis = get_populations(trajectories[init_state],
                                     comp_state_tensor,
                                     step)
        populations[init_state] = pops
        phases[init_state] = phis
    fidelities, diag_mag_errors, diag_phase, U_list = \
        get_gate_stats(trajectories,
                       ct,
                       comp_state_tensor,
                       comp_state_coords)
    # make plots
    # eigens
    fig_en, ax_en = plt.subplots()
    for state, ens in eigenens.items():
        ax_en.plot(tlist, ens, label=state)
    ax_en.set_title('Computational and Aux State Energies')
    ax_en.legend()
    ax_en.set_xlabel('time (ns)')
    ax_en.set_ylabel(r'$E_i - E_0$ (GHz)')
    fname = f'{exp_lbl}_{pulse_profile.name}_eigenen.pdf'
    fig_en.savefig(f'../plots/tg_instability/{fname}')
    # populations
    fig_pops, axs_pops = plt.subplots(4, 1)
    fig_ang, axs_ang = plt.subplots(4, 1)
    for j, init_state in enumerate(populations):
        axs_pops[j].set_title(f'Initial state: {init_state}')
        axs_ang[j].set_title(f'Initial state: {init_state}')
        for basis_state, traj in populations[init_state].items():
            axs_pops[j].plot(tlist[1:], traj, label=basis_state)
            axs_pops[j].set_xlabel('time (ns)')
            axs_pops[j].set_ylabel('population')
        for basis_state, traj in phases[init_state].items():
            axs_ang[j].plot(tlist[1:], traj, label=basis_state)
            axs_ang[j].set_xlabel('time (ns)')
            axs_ang[j].set_ylabel('phase (rad)')
    axs_pops[0].legend(markerscale=0.5, fontsize='small')
    axs_ang[0].legend(markerscale=0.5, fontsize='small')
    fig_pops.set_figheight(12)
    fig_pops.tight_layout()
    fig_ang.set_figheight(12)
    fig_ang.tight_layout
    fname = f'{exp_lbl}_{pulse_profile.name}_populations.pdf'
    fig_pops.savefig(f'../plots/tg_instability/{fname}')
    fname = f'{exp_lbl}_{pulse_profile.name}_phases.pdf'
    fig_ang.savefig(f'../plots/tg_instability/{fname}')

    # fidelities
    fig_f, ax_f = plt.subplots()
    fig_mag_err, ax_mag_err = plt.subplots()
    fig_phase, ax_phase = plt.subplots()
    ax_f.plot(tlist[1:], fidelities)
    ax_f.set_xlabel('time (ns)')
    ax_f.set_ylabel('Fidelity')
    ax_f.axhline(y=0.999, color='r', linestyle='--')
    fname = f'{exp_lbl}_{pulse_profile.name}_gate_fidelities.pdf'
    fig_f.savefig(f'../plots/tg_instability/{fname}')

    for i in range(4):
        ax_mag_err.plot(tlist[1:], diag_mag_errors[:, i],
                        label=format(i, 'b'))
        ax_phase.plot(tlist[1:], diag_phase[:, i]/np.pi,
                      label=format(i, 'b'))
    # ax_phase.axhline(y=-1, color='r', linestyle='--')
    ax_phase.axhline(y=1, color='r', linestyle='--')
    ax_mag_err.legend()
    ax_phase.legend()
    ax_mag_err.set_xlabel('time (ns)')
    ax_phase.set_xlabel('time (ns)')
    ax_mag_err.set_ylabel(r'$U_{ii} - U^{ideal}_{ii}$')
    ax_phase.set_ylabel(r'$Arg(U_{ii})/\pi$')
    ax_mag_err.set_title('Magnitude Error of Propagator Diagonal Elements')
    ax_phase.set_title('Phase of Propagator Diagonal Elements')
    mag_fname = f'{exp_lbl}_{pulse_profile.name}_diag_mag_err.pdf'
    phase_fname = f'{exp_lbl}_{pulse_profile.name}_diag_phase.pdf'
    fig_mag_err.savefig(f'../plots/tg_instability/{mag_fname}')
    fig_phase.savefig(f'../plots/tg_instability/{phase_fname}')

    fig_U_phase, ax_U_phase = plt.subplots()
    fig_U_mag, ax_U_mag = plt.subplots()
    # plot unitary elements
    U_mags = np.abs(U_list)
    U_args = np.angle(U_list)
    for i in range(4):
        for j in range(4):
            i_label = format(i, 'b')
            j_label = format(j, 'b')
            label = r'$|' + i_label + r'\rangle$' + \
                ',' + f' $|{j_label}' + r'\rangle$'
            ax_U_phase.plot(tlist[1:], U_args[:, i, j]/np.pi, label=label)
            ax_U_mag.plot(tlist[1:], U_mags[:, i, j], label=label)
    ax_U_phase.axhline(y=-1, color='r', linestyle='--')
    ax_U_phase.axhline(y=1, color='r', linestyle='--')
    ax_U_phase.legend()
    ax_U_mag.legend()
    ax_U_phase.set_xlabel('time (ns)')
    ax_U_phase.set_ylabel(r'Arg(U_{ij}/pi)')
    ax_U_phase.set_title('Phase of Unitary Elements Over Time')
    ax_U_mag.set_xlabel('time (ns)')
    ax_U_mag.set_ylabel(r'|U_{ij}|')
    ax_U_mag.set_title('Magnitude of Unitary Elements Over Time')
    U_phase_fname = f'{exp_lbl}_{pulse_profile.name}_unitary_elem_phase.pdf'
    U_mag_fname = f'{exp_lbl}_{pulse_profile.name}_unitary_elem_magnitude.pdf'
    path = '../plots/tg_instability/'
    fig_U_mag.savefig(path+U_mag_fname)
    fig_U_phase.savefig(path+U_phase_fname)
    plt.close("all")
    return None


if len(sys.argv) >= 2:
    exp_lbl = sys.argv[1]
else:
    exp_lbl = ''
for pulse_lbl, pulse_profile in pulses.items():
    print(10*'#')
    print(f'plotting {pulse_lbl}')
    plot_pulse(pulse_profile, exp_lbl)
    print(f'\tplotting {pulse_lbl} state trajectories')
    plot_trajectories(pulse_profile, exp_lbl)
