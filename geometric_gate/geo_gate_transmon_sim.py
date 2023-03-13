#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.constants import hbar, pi
from scipy.integrate import quad
from functools import partial
###### Define qubit topology and unitful constants
#UNITS: GHz
N_lev = 5
g = 2*np.pi*0.103
anharm =0.291
w_01 = 7.529
td = 6.5 # parameter sets pulse duration as t_pulse=4*td
#######Define Control pulses
#Define pulse normalization factor OMEGA_2PI
def omega_integrand(u):
    return np.exp(-0.5*u**4)
OMEGA_2PI = 2*np.pi/(quad(omega_integrand, -2, 2)[0]*td)
# Define geometric pulse
def geo_pulse(t,
              td=td,
              t0=0
              ):
    env = OMEGA_2PI*np.exp(-0.5*((t-t0)/td)**4)
    return env
#define formula for a,b complex variables
def ab_coef(theta,phi):
    a = np.sin(theta/2)*np.exp(complex(0,phi))
    b = -np.cos(theta/2)
    return (a,b)

########### Define system hamiltonian
def H_tot(theta,
          phi,
          H0=None,
          anharm=anharm,
          g=g,
          w_01=w_01,
          N_lev=5,
          td=td,
          t0=0,
          return_pulse=True
          ):
    #basis states
    s = list([qt.basis(N_lev, j) for j in range(N_lev)])
    #static qubit hamiltonian
    if H0 is None: #if no static hamiltonian is provided, assume rotating frame
        H0 = qt.Qobj(np.zeros((N_lev, N_lev)))
    # build interaction hamiltonian
    a = np.sin(theta/2)*np.exp(complex(0,phi))
    b = -np.cos(theta/2)
    s_B = np.conj(a)*s[0] + np.conj(b)*s[2]
    sigma_x_1B = s[1]*s_B.dag() + s_B*s[1].dag()
    H_int = 0.5*sigma_x_1B # static part of interaction hamiltonian
    # time-dependent coefficient
    args = {
        'td':td,
        't0':t0,
    }
    def int_coef(t,args):
        return geo_pulse(t,**args)
    H = [H0,[H_int, int_coef]]
    if return_pulse:
        return H, args, partial(geo_pulse, **args)
    else:
        return H, args

############# Define simulation procedures
def run_sim(theta,
            phi,
            H0=None,
            s0=None,
            anharm=anharm,
            g=g,
            w_01=w_01,
            N_lev=5,
            td=td,
            t0=0,
            n_timestep=int(1e5),
            return_pulse=True
            ):
    if s0 is None:
        s0 = qt.basis(N_lev,0)
    args =  (theta,
             phi,
             H0,
             anharm,
             g,
             w_01,
             N_lev,
             td,
             t0,
             return_pulse)
    if return_pulse:
        H, args, pulse = H_tot(*args)
    else:
        H, args = H_tot(*args)
    tlist = np.linspace(-2*td, 2*td, num=n_timestep)
    results = qt.mesolve(H, s0, tlist, args=args)
    if return_pulse:
        return results, tlist, pulse(tlist)
    else:
        return results, tlist

##################### Post-simulation analysis functions
def get_pop_evolution(sim_res, subspace_dim=3):
    n_timestep = len(sim_res.states)
    rhos = np.array([psi*psi.dag() for psi in sim_res.states])
    populations = np.zeros((n_timestep, subspace_dim))
    for i, rho in enumerate(rhos):
        pops = []
        for j in range(subspace_dim):
            pops.append(rho[j][j])
        populations[i] = np.array(pops)
    return populations

def plot_results(sim_res,
                 tlist,
                 pulse=None,
                 subspace_dim=3,
                 pulse_path='./plots/pulse.pdf',
                 pop_path='./plots/populations.pdf'):
    populations = get_pop_evolution(sim_res, subspace_dim)
    fig, ax = plt.subplots()
    for i in range(subspace_dim):
        ax.plot(tlist, populations[:,i], label='s{}'.format(i))
    ax.legend()
    fig.savefig(pop_path)
    if pulse is not None:
        fig2, ax2 = plt.subplots()
        ax2.plot(tlist,pulse)
        fig2.savefig(pulse_path)
    return None


# END FUNC DEFS
#
# Sims
#Hadamard Gate
results, tlist, pulse = run_sim(theta=np.pi/4, phi=0, td=td)
plot_results(results,
             tlist,
             pulse=pulse,
             pulse_path='./plots/hadamard_pulse.pdf',
             pop_path='./plots/hadamard_evolution.pdf')
print('######################## Done ########################3')

# Not gate
results, tlist, pulse = run_sim(theta=np.pi/2, phi=0, td=td)
plot_results(results,
             tlist,
             pulse=pulse,
             pulse_path='./plots/not_pulse.pdf',
             pop_path='./plots/not_evolution.pdf')
print('######################## Done ########################3')
