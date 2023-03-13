#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.constants import hbar, h, pi
from scipy.integrate import quad
from functools import partial
"""
Implementation of geometric gate on fluxonium as outlined in Setiawan et al. (2022)
"""
########################circuit parameters
### Hamiltonian expressed in units of radian frequency
### All quantities in GHz unless otherwise indicated
# Global Parameters
nlev = 5 # Number of energy levels to consider for each subsystem.
j = complex(0,1)
qho_basis = list([qt.basis(nlev,i) for i in range(nlev)])
# Fluxonium A
E_JA = 4.5
E_CA = 1.8
E_LA = 1.5
phi_ex_A = 0.5 #units of GHz*phi_0
w_A = 2*pi*1.79

#Fluxonium B
E_JB = 3.5
E_CB = 1.1
E_LB = 1.0
phi_ex_B = 0.5 #units of GHz*phi_0
w_B = 0.86

# Transmon Coupler
w_C = 1.11
U = 0.005

# Static couplings to transmon
g_AC = 0.63
g_BC = 0.6
g_AB = 0.04

################### Pulse parameters
#global param
omega_0 = 2*np.pi*0.02522 # default maximum pulse amplitude (fig 11 caption)
tg = 45 #ns, defined in fig 11 caption
t_ramp = 0.45 #ns, defined in fig 11 caption

#################### System Hamiltonian

########### Define Operators
def fluxonium_hamiltonian(n,
                          phi,
                          phi_ex,
                          E_C,
                          E_J,
                          E_L,
                          nlev=nlev):
    #helper function, implements eq 20
    Id = qt.qeye(nlev)
    capacitor_energy = 4*E_C*n**2
    josephenson_energy = E_J*phi.cosm()
    inductor_energy = (0.5)*E_L*(phi - 2*np.pi*phi_ex*Id)
    H = capacitor_energy + josephenson_energy + inductor_energy
    #diagonalize hamiltonian, early diagonalization miminizes truncation errors
    eigenval, eigenstate_basis = H.eigenstates()
    H_diag = qt.qdiags(eigenval, offsets=0)
    return H_diag, eigenstate_basis

def fluxonium_hamiltonian_from_ct_params(
        E_C,
        E_J,
        E_L,
        phi_ext,
        nlev=nlev):

    qho_basis = qt.qeye(nlev)
    a = qt.destroy(nlev)
    #build charge and phase operators in qho basis
    n_zpf = (E_L/(32*E_C))**(1/4)
    phi_zpf = (2*E_C/E_L)**(1/4)
    n = n_zpf*(a+a.dag())
    phi = phi_zpf*j*(a - a.dag())
    #obtain hamiltonian in eigenbasis
    H, eigenstate_basis = fluxonium_hamiltonian(n,
                                                phi,
                                                phi_ext,
                                                E_C,
                                                E_J,
                                                E_L,
                                                nlev)
    # save operators defined for hamiltonian in diagonal basis
    ops = {
        'n': n.transform(eigenstate_basis),
        'phi': phi.transform(eigenstate_basis)
    }
    return H, eigenstate_basis, ops

def raise_hamiltonian(H, ops, subsystem_idx:int, nlev=nlev):
    Id = qt.qeye(nlev)
    raised_ops = {}
    #blank operator on full HilBert SPace, up to taking kronecker product
    full_hbspc_op = [Id, Id, Id]
    #raise hamiltonian to joint hilbert space
    full_hbspc_op[subsystem_idx] = H
    H_raised = qt.tensor(full_hbspc_op)
    # raise all operators operating on subsytem hamiltonian
    for op_name in ops:
        subsytem_op = ops[op_name]
        full_hbspc_op[subsystem_idx] = subsytem_op
        raised_ops[op_name] = qt.tensor(full_hbspc_op)
    return H_raised, raised_ops

def find_comp_states(H_dressed,
                     H_bare,
                     a_basis=None,
                     b_basis=None,
                     c_basis=None,
                     nlev=None):
    #Following algorithm assumes avoided crossings due to interaction hamiltonian
    #If avoided crossings occur, the ordering of the eigenstates according to
    #   increasing energy should be the same between bare and dressed eigenstates
    
    #set a_basis, b_basis, c_basis and nlev
    if nlev is None:
        nlev = np.array(H_dressed.dims).ravel()[0]
    bases = [a_basis, b_basis, c_basis]
    qho_basis = list([qt.Qobj(v) for v in qt.qeye(nlev)[:,]])
    for n, basis in enumerate(bases):
        if basis is None:
            bases[n] = qho_basis
    a_basis, b_basis, c_basis = bases


    bare_state_idx, _ = find_bare_states(H_bare, a_basis, b_basis, c_basis, nlev)
    _, dressed_states = H_dressed.eigenstates()
    dressed_comp_states = np.empty(bare_state_idx.shape, dtype=object)

    #following assumes bare_state_idx.shape = (3,3,3), implementation only
    #   valid for 2 qubit gate with auxilary coupler
    #TODO: generalize to arbitrary shapes to implement n-qubit gate
    for n in range(3):
        for m in range(3):
            for k in range(3):
                nmk_state_idx = bare_state_idx[n,m,k]
                nmk_state = dressed_states[nmk_state_idx]
                dressed_comp_states[n,m,k] = nmk_state
    #note that index of computational states does not change from bare states to dressed states
    return bare_state_idx, dressed_comp_states

def find_bare_states(H_bare,
                     a_basis=None,
                     b_basis=None,
                     c_basis=None,
                     nlev=None):
    #default arguments assume all subsystem hamiltonians are written in their eigenbasis
    #TODO quick and dirty implementation. Code can be greatly simplified and optimized
    # by using matrix products and tensor contractions instead of nested for loops
    if nlev is None:
        nlev = np.array(H_bare.dims).ravel()[0]
    bases = []
    qho_basis = list([qt.Qobj(v) for v in qt.qeye(nlev)[:]])
    for basis in [a_basis, b_basis, c_basis]:
        if basis is None:
            bases.append(qho_basis)
        else:
            bases.append(basis)
    a_basis, b_basis, c_basis = bases
    product_states = build_product_basis(a_basis, b_basis, c_basis, nlev)
    bare_energies, bare_states = H_bare.eigenstates()
    gram_tensor = np.empty((3,3,3,nlev**3), dtype=complex)
    for n in range(3):
        for m in range(3):
            for k in range(3):
                for l in range(nlev**3):
                    gram_tensor[n,m,k,l] =\
                        (product_states[n,m,k].dag()*bare_states[l]).tr()
    labeled_states = np.empty((3,3,3), dtype=object)
    comp_state_idx = np.empty((3,3,3), dtype=int)
    for n in range(3):
        for m in range(3):
            for k in range(3):
                gram_row = gram_tensor[n,m,k,:]
                labeled_state_idx = np.argmax(gram_row)
                comp_state_idx[n,m,k] = labeled_state_idx
                labeled_states[n,m,k] = bare_states[labeled_state_idx]
    return comp_state_idx, labeled_states

def build_product_basis(a_basis, b_basis, c_basis, nlev=nlev):
    bare_comp_state_tensor = np.empty((3,3,3), dtype=object)
    for n in range(3):
        for m in range(3):
            for k in range(3):
                a_state = a_basis[n]
                b_state = b_basis[m]
                c_state = c_basis[k]
                bare_comp_state_tensor[n,m,k] = \
                    qt.tensor(a_state, b_state, c_state)
    return bare_comp_state_tensor

#Build fluxonium A and B Hamiltonians
# sub for subsytem operator
def build_subsystems(nlev=nlev):
    #build fluxonium A
    H_A_sub, A_basis, ops_A_sub = fluxonium_hamiltonian_from_ct_params(
        E_CA,
        E_JA,
        E_LA,
        phi_ex_A,
        nlev
    )
    #build fluxonium B
    H_B_sub, B_basis, ops_B_sub = fluxonium_hamiltonian_from_ct_params(
        E_CB,
        E_JB,
        E_LB,
        phi_ex_B,
        nlev
    )
    #### build transmon coupler hamiltonian in QHO basis
    a_sub = qt.destroy(nlev)
    H_C_sub = w_C*a_sub.dag()*a_sub - U*a_sub.dag()**2 * a_sub**2
    # Build Return Dictionary
    subsystems = {
        'A': (H_A_sub, A_basis, ops_A_sub),
        'B': (H_B_sub, B_basis, ops_B_sub),
        'C': (H_C_sub, qho_basis, {'a': a_sub})
    }
    return subsystems

def build_bare_systems(A, B, C, nlev):
    H_A_sub, _, ops_A_sub = A
    H_B_sub, _, ops_B_sub = B
    H_C_sub, _, ops_C_sub = C

    #raise hamiltonians and operators to joint hilbert space
    H_A, ops_A = raise_hamiltonian(H_A_sub, ops_A_sub, 0, nlev)
    H_B, ops_B = raise_hamiltonian(H_B_sub, ops_B_sub, 1, nlev)
    H_C, ops_C = raise_hamiltonian(H_C_sub, ops_C_sub, 2, nlev)
    # define bare hamiltonian
    H_bare = H_A + H_B + H_C
    subsystem_dict = {
        'A': (H_A, ops_A),
        'B': (H_B, ops_B),
        'C': (H_C, ops_C)
    }
    return H_bare, subsystem_dict

def build_dressed_system(H_bare, ops_A, ops_B, ops_C, H_int=None):
    if H_int is None:
        H_int = (g_AC*ops_A['n'] + g_BC*ops_B['n'])* \
            (ops_C['a'].dag() + ops_C['a']) + \
            g_AB*ops_A['n']*ops_B['n']
    H_dressed = H_bare + H_int
    return H_dressed, H_int

def build_static_system(nlev=nlev):
    subsystems = build_subsystems(nlev)
    H_bare, subsystems = build_bare_systems(**subsystems, nlev=nlev)

    ops_A = subsystems['A'][1]
    ops_B = subsystems['B'][1]
    ops_C = subsystems['C'][1]
    ops = (ops_A, ops_B, ops_C)
    H_dressed, H_int = build_dressed_system(H_bare, *ops)
    static_system = {
        'H_dressed': H_dressed,
        'H_bare': H_bare,
        'subsystems': subsystems
    }
    return static_system



##raise hamiltonians and operators to joint hilbert space
#H_A, ops_A = raise_hamiltonian(H_A_sub, ops_A_sub, 0)
#H_B, ops_B = raise_hamiltonian(H_B_sub, ops_B_sub,1)
#H_C, ops_C = raise_hamiltonian(H_C_sub, {'a': a_sub}, 2)
#
## define interaction hamiltonian
#
##define dressed static hamiltonian
#H_0 = H_A + H_B + H_C + H_int
#
####### Define Time-dependent driven hamiltonian
#
#
## Time dependent pulse
# define time-path polynomial, first and second derivatives (eq A3)
polynom = lambda x: 6*(2*x)**5 - 15*(2*x)**4 + 10*(2*x)**3
d_polynom = lambda x: 60*(2*x)**4 - 120*(2*x)**3 + 60*(2*x)**2
d2_polynom = lambda x: 480*(2*x)**3 - 720*(2*x)**2 + 480*x

def theta(tlist:np.ndarray, deriv=0, tg=tg):
    """Function defines time-path theta(t) as defined in
    eq A2. Depending on value of deriv parameter, function will
    return theta(t), theta'(t), or theta''(t).
    """
    #dictionary storing terms of different time-derivatives
    theta_derivs = {
        0 : [(np.pi/2)* polynom(tlist/tg),
             (np.pi/2)*(1-polynom(tlist/tg - 0.5))],
        1 : [(np.pi/(2*tg))*d_polynom(tlist/tg),
            -(np.pi/(2*tg))*d_polynom(tlist/tg - 0.5)],
        2 : [(np.pi/(2*tg**2))*d2_polynom(tlist/tg),
             -(np.pi/(2*(tg**2)))*d2_polynom(tlist/tg -0.5)]
    }
    #pick correct degree of time-derivative
    theta_d = theta_derivs[deriv]
    #pick elements out of t that satisfy each of the 2
    # sections of piecewize function in eq A2
    cond_1 = np.argwhere(np.logical_and(0<=tlist, tlist<=tg/2))
    cond_2 = np.argwhere(np.logical_and(tg/2<tlist, tlist<=tg))

    #build return array with elements corresponding with
    # correct time-path value at every time in tlist
    ret_array = np.zeros(tlist.shape)
    ret_array[cond_1] = theta_d[0][cond_1]
    ret_array[cond_2] = theta_d[1][cond_2]
    return ret_array

#define time-dependent pulse-amplitude (eq 10)
def pulse_env(t:np.ndarray, geo_phase, omega_0 =omega_0, tg=tg):
    #get time derivatives for theta
    theta_d0 = theta(t,tg=tg)
    theta_d1 = theta(t,deriv=1, tg=tg)
    theta_d2 = theta(t, deriv=2, tg=tg)
    #define pulse phase (eq 5)
    gamma = geo_phase* np.heaviside(t-tg/2, 1)
    #define pulses for fluxonium A and B
    omega_A = omega_0*(
        np.sin(theta_d0) +
        4*np.cos(theta_d0)*theta_d2/(omega_0**2 + 4*theta_d1**2)
        )
    omega_B = omega_0*np.exp(j*gamma)*(
        np.cos(theta_d0) -
        4*np.sin(theta_d0)*theta_d2/(omega_0**2 + 4*theta_d1**2)
    )
    return omega_A, omega_B

def g_ac(tlist, geo_phase, H_0, states=None, omega_0=omega_0, tg=tg):
    # eqs 32a & 32b
    # compute numerators
    omega_A, omega_B = pulse_env(tlist, geo_phase, omega_0, tg)
    #compute denominators
    a = ops_C['a']
    if states is None:
       _, states = find_comp_states(H_0)
    ge1 = states[0,1,1]
    ee0 = states[1,1,0]
    gf0 = states[1,2,0]
    g_A = omega_A/(ge1.dag()*a.dag()*a*ee0)
    g_B = omega_B/(ge1.dag()*a.dag()*a*gf0)
    return g_A, g_B

def delta_k(tlist, geo_phase, H_0, omega_0=omega_0, tg=tg):
    e_k = 0
    _, states = find_comp_states(H_0)
    g_A, g_B = g_ac(tlist, geo_phase, H_0, states)
    delta_kls = np.empty((2,2,2))
    eta_kl = {((0,1,1),(1,1,0)): -0.84*np.pi,
              ((0,1,1),(0,2,0)): -2.43*np.pi,
              ((1,1,0),(0,1,1)): 0.84*np.pi,
              ((0,2,0),(0,1,1)): 2.43*np.pi}
    w_j = {'A': w_A,
           'B': w_B}
    k = [(0,1,1), (1,1,0), (0,2,0)]
    l = [()]
