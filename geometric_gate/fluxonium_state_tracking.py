#!/usr/bin/env python3
import geo_gate_fluxonium_sim
import qutip as qt
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
exec(open('geo_gate_fluxonium_sim.py').read())

def find_bare_eigenstates(H_bare=None, nlev=4):
    if H_bare is None:
        H_bare = H_A + H_B + H_C
    #assumes 3 systems
    Ha_bare = H_bare.ptrace(0)
    Hb_bare = H_bare.ptrace(1)
    Hc_bare = H_bare.ptrace(2)

    a_bare_energies, a_bare_states = Ha_bare.eigenstates()
    b_bare_energies, b_bare_states = Hb_bare.eigenstates()
    c_bare_energies, c_bare_states = Hc_bare.eigenstates()

    #extract bare eigenenergies
    a_lowest = a_bare_energies[0:nlev]
    b_lowest = b_bare_energies[0:nlev]
    c_lowest = c_bare_energies[0:nlev]
    #assemble eigenstates
    eigenenergies = np.zeros((nlev,nlev,nlev))
    for n in range(nlev):
        for m in range(nlev):
            for k in range(nlev):
                energy_state = a_lowest[n] +\
                    b_lowest[m] + c_lowest[k]
                eigenenergies[n,m,k] = energy_state
    #check if these coincide with eigenenergies of bare
    #hamiltonian
    bare_energies = H_bare.eigenenergies()
    return bare_energies, eigenenergies
