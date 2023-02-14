#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.constants import hbar, pi
from scipy.integrate import quad
# Define qubit topology
N_lev = 5
comp_subspace = [0,2]
######## Define system hamiltonian
# energy basis

s0 = qt.basis(N_lev, 0)
s1 = qt.basis(N_lev, 1)
s2 = qt.basis(N_lev, 2)
s3 = qt.basis(N_lev, 3)
s4 = qt.basis(N_lev, 4)

H0 = 7.529e9*s1*s1.dag() + 7.238e9*s2*s2.dag() + 6.946e9*s3*s3.dag() + 6.656e9*s4*s4.dag() # time-independent term

#interaction hamiltionian
g = 2*np.pi*103e6
#H_int = g*(s0*s1.dag() + s1*s2.dag() - s1*s0.dag() - s2*s1.dag())
# define (1,0) coord of time-dependent term
#H1 = s1*s0.dag()#*(1/2)
#def H1_coef(t, args):
    #a = args['a']
    #b = args['b']
    #phase_ij = args['phase_ij']
    #return pulse(t, a, b, phase_ij)# * a
## define (1,2) coord of time-dependent term
#H2 = s1*s2.dag()#*(1/2)
#def H2_coef(t, args):
    #a = args['a']
    #b = args['b']
    #phase_ij = args['phase_ij']
    #return pulse(t, a, b, phase_ij)# * b
def interaction_coef(t, args):
    a = args['a']
    b = args['b']
    phase_ij = args['phase_ij']
    return pulse(t,a,b,phase_ij)
###### Define pulses
#Geometric pulse
td = 6.5e-9
def omega_integrand(u):
    return np.exp(-0.5*u**4)/td

OMEGA_2PI = 2*np.pi/quad(omega_integrand, 2*td, -2*td)[0]

def pulse(t,
          a,
          b,
          phi,
          td=6.5e-9,
          t_0=0,
          wd_01=7.529e9,
          wd_12=7.238e9):
    env = OMEGA_2PI*np.exp(-0.5*((t-t_0)/td)**4)
    drive_01 = abs(a)*np.cos(wd_01*t-phi)
    drive_12 = abs(b)*np.cos(wd_12*t)
    return env*(drive_01 + drive_12)

def ab_coef(theta, phi):
    a = np.sin(theta/2)*np.exp(complex(0,phi))
    b = -np.cos(theta/2)
    return (a,b)
###########
def integrand(t):
    return np.exp(-0.5*((t/td)**4))*\
        (abs(np.sin(np.pi/4))*\
         np.cos(7.529e9*t+ np.pi)+\
         abs(np.cos(np.pi/4))*np.cos(7.238e9*t))
OMEGA_2PI*quad(integrand,-2*td, 2*td)[0]

#HADAMARD
n_timestep = int(1e5)
theta = np.pi/2
phi_01 = 0
a, b = ab_coef(theta, phi_01)
tlist = np.linspace(-2*td, 2*td, num=n_timestep)
args = {'a': a,
        'b': b,
        'phase_ij': phi_01
        }

H = [H0, [H_int, interaction_coef]]
#H = [H0,[H1, H1_coef], [H2, H2_coef]]
output = qt.mesolve(H, s0, tlist,args=args)
density_matrices = [x*x.dag() for x in output.states]
populations = np.zeros((n_timestep,3))
print(H)
for i, rho in enumerate(density_matrices):
    g_pop = rho[0,0]
    e_pop = rho[1,1]
    f_pop = rho[2,2]
    populations[i] = np.array([g_pop, e_pop, f_pop])
populations
pulse_t = pulse(tlist, args['a'], args['b'], phi_01, )
plt.plot(tlist, pulse_t)
plt.savefig('./plots/pulse.pdf')
plt.plot(tlist, populations[:,0], label='s0')
plt.plot(tlist, populations[:,1], label='s1')
plt.plot(tlist, populations[:,2], label='s2')
plt.legend()
plt.savefig('./plots/populations.pdf')
