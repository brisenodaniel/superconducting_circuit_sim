#!/usr/bin/env python3


def find_comp_states(H_bare=H_A+H_B+H_C, H_int=H_int):
    """
    Function helps identify the states <ge,1|, <ee,0|, <gf,0|.
    To do so, it finds the desired states with H_int turned 'off',
    then smoothly increases the coefficient of H_C to 1. Function
    plots the trajectories of the lowest 3 eigenvalues of the
    static hamiltonian so that we may identify the correct
    states via visual inspection. Function meant to be called
    interactively.
    """
    xlist = np.linspace(0,1,100)
    # get trajectories of desired subsystem states
    trajectories = get_comp_state_traj(H_bare, H_int, xlist)

    #plot each to confirm avoided crossings
    #subsystem A
    A_energy_traj = trajectories['A']
    fig1, ax1 = plt.subplots()
    for i in range(5):
        ax1.plot(A_energy_traj[i,:], label="s{}".format(i))
    ax1.legend()
    plt.show()


def get_comp_state_traj(H_bare, H_int, xlist):
    H_int_list = [x*H_int for x in xlist]
    A_bottom_5 = get_subsystem_state_traj(H_bare, H_int_list, 0)
    B_bottom_5 = get_subsystem_state_traj(H_bare, H_int_list, 1)
    C_bottom_5 = get_subsystem_state_traj(H_bare, H_int_list, 2)
    #build dictionary for easy trajectory access
    traj_dict = {
        'A': A_bottom_5,
        'B': B_bottom_5,
        'C': C_bottom_5
    }
    return traj_dict

def get_subsystem_state_traj(H_bare, H_int_list, subsystem_idx):
    eigenval_list = np.zeros((5,len(H_int_list)))
    for i, H_int in enumerate(H_int_list):
        H = H_bare + H_int
        H_dressed = H.ptrace(subsystem_idx)
        energies = H_dressed.eigenenergies()
        bottom_5 = [energies[0], energies[1], energies[2],
               energies[3], energies[4]]
        eigenval_list[:,i] = bottom_5
    return eigenval_list
