#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.constants import hbar, pi
from scipy.integrate import quad
from functools import partial
###### Define qubit topology
N_lev = 5
g = 2*np.pi*103e6
anharm = 291e6
w_01 = 7.529e9

#######Define Control pulses
td = 6.5e-9 # parameter sets pulse duration as t_pulse=4*td
#define constant prefactor to 01 and 12 geometric pulses
def omega_integrand(u):
    return np.exp(-0.5*u*4)/td

OMEGA_2PI = 2*np.pi/quad(omega_integrand, 2*td, -2*td)[0]
# Define geometric pulse
def geo_pulse(t,
              a,
              b,
              phi,
              w_01,
              w_12,
              td=td,
              t0=0
              ):
    env = OMEGA_2PI*np.exp(-0.5*((t-t0)/td)**4)
    drive_01 = abs(a)*np.cos(w_01*t-phi)
    drive_12 = abs(b)*np.cos(w_12*t)
    return env*(drive_01 + drive_12)
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
    if H0 is None: #if no hamiltonian is provided, build standard transmon hamiltonian
        H0=0
        wd = w_01
        for s_j in s[1:]:
            H0 += wd*s_j*s_j.dag()
            wd -= anharm
        H0 = qt.qeye(N_lev)
    # build interaction hamiltonian
    a = np.sin(theta/2)*np.exp(complex(0,phi))
    b = -np.cos(theta/2)
    s_B = np.conj(a)*s[0] + np.conj(b)*s[2]
    sigma_z_1B = s[0]*s_B.dag() + s_B*s[0].dag()
    H_int = 0.5*sigma_z_1B # static part of interaction hamiltonian
    # time-dependent part
    args = {
        'a':a,
        'b':b,
        'phi':phi,
        'td':td,
        't0':t0,
        'w_01':w_01,
        'w_12':w_01-anharm
    }
    def int_coef(t,args):
        return geo_pulse(t,**args)
    H = [H0,[H_int, int_coef]]
    if return_pulse:
        return H, args, partial(geo_pulse, **args)
    return H, args

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
    print(H)
    print('############################################')
    print(s0)
    print('############################################33')
    print(tlist)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(args)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print((H[1][1](tlist,args)==geo_pulse(tlist, **args)).all())
    results = qt.mesolve(H, s0, tlist, args=args)
    if return_pulse:
        return results, tlist, pulse(tlist)
    else:
        return results, tlist

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
    #fig.clf()
    for i in range(subspace_dim):
        ax.plot(tlist, populations[:,i], label='s{}'.format(i))
    ax.legend()
    fig.savefig(pop_path)
    if pulse is not None:
        print(pulse)
        fig2, ax2 = plt.subplots()
        #fig2.clf()
        ax2.plot(tlist,pulse)
        fig2.savefig(pulse_path)
    return None

# Run parameters corresponding to Hadamard Gate

results, tlist, pulse = run_sim(theta=np.pi/4, phi=0)
plot_results(results,
             tlist,
             pulse=pulse,
             pulse_path='./plots/hadamard_pulse.pdf',
             pop_path='./plots/hadamard_evolution.pdf')
print('done')






## Define qubit topology
#N_lev = 5
#comp_subspace = [0,2]
######### Define system hamiltonian
## energy basis
#
#s0 = qt.basis(N_lev, 0)
#s1 = qt.basis(N_lev, 1)
#s2 = qt.basis(N_lev, 2)
#s3 = qt.basis(N_lev, 3)
#s4 = qt.basis(N_lev, 4)
#
#H0 = 7.529e9*s1*s1.dag() + 7.238e9*s2*s2.dag() + 6.946e9*s3*s3.dag() + 6.656e9*s4*s4.dag() # time-independent term
##interaction hamiltionian
#g = 2*np.pi*103e6
#H_int = g*(s0*s1.dag() + s1*s2.dag() - s1*s0.dag() - s2*s1.dag())
##     define (1,0) coord of time-dependent term
#def set_bright_state(a,b):
    #return np.conj(a)*s0 + np.conj(b)*s2
#def set_interaction_hamiltonian(a,b):
    #s_B = set_bright_state(a,b)
    #return (1/2)*g*(s0*s_B.dag() + s_B*s0.dag())
#
   #
##H1 = s1*s0.dag()
##def H1_coef(t, args):
    ##a = args['a']
    ##b = args['b']
    ##phase_ij = args['phase_ij']
    ##return pulse(t, a, b, phase_ij)# * a
##define (1,2) coord of time-dependent term
##H2 = s1*s2.dag()#*(1/2)
##def H2_coef(t, args):
    ##a = args['a']
    ##b = args['b']
    ##phase_ij = args['phase_ij']
    ##return pulse(t, a, b, phase_ij)# * b
#
#def pulse(t,
          #a,
          #b,
          #phi,
          #td=6.5e-9,
          #t0=0,
          #w_01=7.529e9,
          #w_12=7.238e9):
    #env = OMEGA_2PI*np.exp(-0.5*((t-t0)/td)**4)
    #drive_01 = abs(a)*np.cos(w_01*t-phi)
    #drive_12 = abs(b)*np.cos(w_12*t)
    #return env*(drive_01 + drive_12)
#
#def interaction_coef(t, args):
    #a = args['a']
    #b = args['b']
    #phase_ij = args['phase_ij']
    #return pulse(t,a,b,phase_ij)
#
#td = 6.5e-9
#def omega_integrand(u):
    #return np.exp(-0.5*u**4)/td
#
#OMEGA_2PI = 2*np.pi/quad(omega_integrand, 2*td, -2*td)[0]
#
#
#def ab_coef(theta, phi):
    #a = np.sin(theta/2)*np.exp(complex(0,phi))
    #b = -np.cos(theta/2)
    #return (a,b)
####### Define pulses
##Geometric pulse
############
#
##HADAMARD
#n_timestep = int(1e5)
#theta = np.pi/4
#phi_01 = 0
#a, b = ab_coef(theta, phi_01)
#tlist = np.linspace(-2*td, 2*td, num=n_timestep)
#args = {'a': a,
        #'b': b,
        #'phase_ij': phi_01
        #}
#H_int = set_interaction_hamiltonian(a,b)
#H = [H0, [H_int, interaction_coef]]
##H = [H0,[H1, H1_coef], [H2, H2_coef]]
#output = qt.mesolve(H, s0, tlist,args=args)
#density_matrices = [x*x.dag() for x in output.states]
#populations = np.zeros((n_timestep,3))
#print(H)
#for i, rho in enumerate(density_matrices):
    #g_pop = rho[0,0]
    #e_pop = rho[1,1]
    #f_pop = rho[2,2]
    #populations[i] = np.array([g_pop, e_pop, f_pop])
#populations
#pulse_t = pulse(tlist, args['a'], args['b'], phi_01, )
#plt.plot(tlist, pulse_t)
#plt.savefig('./plots/pulse.pdf')
#plt.clf()
#plt.plot(tlist, populations[:,0], label='s0')
#plt.plot(tlist, populations[:,1], label='s1')
#plt.plot(tlist, populations[:,2], label='s2')
#plt.legend()
#plt.savefig('./plots/populations.pdf')
