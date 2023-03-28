"""
Transmon and Fluxonium definitions

Document contains function definitions for constructing operators on
fluxonia qudits and transmon coupler as described in Setiawan et. al. 2022
in the literature folder.

All operators are defined in the Quantum Harmonic Oscillator basis.
"""
from typing import TypeAlias
import qutip as qt
import numpy as np

Qobj: TypeAlias = qt.Qobj
j: complex = complex(0,1)

def build_fluxonium_operators(nlev:int,
                              E_C:float,
                              E_J:float,
                              E_L:float,
                              phi_ext:float) -> dict[str,Qobj]:
    """Method defines operators for a fluxonium circuit as defined in
    Setiawan et. al. 2022.

    Args:
        nlev (int): Number of energy levels to consider in the system
        E_C (float): Circuit parameter, capacitance energy coefficient
        E_J (float): Circuit parameter, josephenson energy coefficient
        E_L (float): Circuit parameter, linear inductive energy coefficient
        phi_ext (float): External flux threading  loop formed by josephenson
         junction and linear inductor.

    Returns:
        dict[str,Qobj]: _Dictionary of operators acting on fluxonium state vector.
           Includes Hamiltonian operator
    """
    a:Qobj = qt.destroy(nlev)
    n_zpf:float = (E_L/(32*E_C))**(1/4)
    phi_zpf:float = (2*E_C/E_L)**(1/4)
    n:Qobj = n_zpf*(a + a.dag())
    phi:Qobj = phi_zpf*j*(a-a.dag())
    H:Qobj = build_fluxonium_hamiltonian(n,phi,E_C, E_J, E_L, phi_ext)
    return {'n':n, 'phi':phi, 'H': H}

def build_fluxonium_hamiltonian(n:Qobj, 
                                phi:Qobj, 
                                E_C:float, 
                                E_J:float,
                                E_L:float,
                                phi_ext:float) -> Qobj:
    """Method defines the hamiltonian operator for a fluxonium circuit as described
    in Setiawan et. al. 2022.

    Args:
        n (Qobj): Charge operator for the fluxonium circuit.
        phi (Qobj): Phase operator for the fluxonium circuit.
        E_C (float): Circuit parameter. Capacitive Energy Coefficient.
        E_J (float): Circuit parameter. Josephenson Energy Coefficient.
        E_L (float): Circuit parameter. Inductive Energy Coefficient.
        phi_ext (float): External flux threading  loop formed by josephenson
         junction and linear inductor.

    Returns:
        Qobj: Hamiltonian operator for fluxonium circuit with charge operator `n` and 
        phase operator `phi`.
    """
    nlev:int = np.array(n.dims).ravel()[0]
    Id:Qobj = qt.qeye(nlev)
    capacitor_energy:Qobj = 4*E_C*n**2 # type: ignore
    josephenson_energy:Qobj = E_J*phi.cosm()
    inductor_energy:Qobj = 0.5*E_L*(phi - 2*np.pi*phi_ext*Id)
    H = capacitor_energy + josephenson_energy + inductor_energy
    return H

def build_transmon_operators(nlev:int, w:float, U:float) -> dict[str,Qobj]:
    """Method defines the hamiltonian operator for a transmon circuit as described
    in Setiawan et. al. 2022

    Args:
        nlev (int): Number of energy levels to consider for transmon circuit.
        w (float): Circuit parameter, frequency of 0 -> 1 transition.
        U (float): Circuit parameter, transmon anharmonicity

    Returns:
        dict[str,Qobj]: Dictionary of operators acting on transmon state vector. 
        Includes hamiltonian.
    """
    a:Qobj = qt.destroy(nlev)
    H:Qobj = w*a.dag()*a - U*a.dag()**2 * a**2
    return {'a': a, 'H': H}