# Unit tests for optimize_subsystem.py
import qutip as qt
import numpy as np
import pytest
import yaml
from typing import TypeAlias, Callable

Qobj: TypeAlias = qt.Qobj
Qsys: TypeAlias = dict[str,Qobj]
j: complex = complex(0,1)
if __name__== '__main__':
    import sys
    sys.path.append('../src')
    yaml_path = '../config/circuit_parameters.yaml'
else:
     yaml_path = './config/circuit_parameters.yaml'    
import optimize_subsystem as opt
import subsystems


#extract circuit parameters
with open(yaml_path,'r') as stream:
    try:
        ct_params:dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Subsystem constructor parameters
param_lbls = ['E_C', 'E_J', 'E_L', 'phi_ext']
transmon_param_lbls = ['w','U']
flux_A_params:dict[str,float] = {lbl:ct_params['A'][lbl]\
                                 for lbl in param_lbls}
flux_B_params:dict[str,float] = {lbl:ct_params['B'][lbl]\
                                 for lbl in param_lbls}
transmon_params:dict[str,float] = {lbl:ct_params['C'][lbl] \
                                   for lbl in transmon_param_lbls}

# fluxonium and transmon operator labels
flux_ops = ['n', 'phi', 'H']
transmon_ops = ['a','H']

#helper functions
def msg(err:str='', n:int=0)->str:
        err +="\n Simulated with {} energy levels".format(n)

def is_diag(op:Qobj)->bool:
     diags = op.diag()
     diag_op = qt.qdiags(diags,0)
     return op==diag_op

def test_stabilize_nlev():
    flux_constr:Callable['...',Qsys] = subsystems.build_fluxonium_operators
    transmon_constr:Callable['...',Qsys] = subsystems.build_transmon_operators

    #test on fluxonium
    #default params
    flux_A, nlev_A = opt.stabilize_nlev(flux_constr, flux_A_params)
    flux_B, nlev_B = opt.stabilize_nlev(flux_constr, flux_B_params)

    assert all([isinstance(flux_A[op], Qobj) for op in flux_ops])
    assert all([isinstance(flux_B[op], Qobj) for op in flux_ops])
    assert not is_diag(flux_A['H']), 'Fluxonium A hamiltonian should not\
        be diagonal in QHO basis'
    assert not is_diag(flux_B['H']), 'Fluxonium B hamiltonian should not \
        be diagonal in QHO basis'
    assert all([flux_A[op].dims==[[nlev_A],[nlev_A]] for op in flux_ops]),\
    'Fluxonium A operators of incorrect shape.'
    assert all([flux_B[op].dims==[[nlev_B],[nlev_B]] for op in flux_ops]),\
    'Fluxonium B operators of incorrect shape.'
    #test assertion error for min_nlev< stable_levels
    with pytest.raises(AssertionError):
        opt.stabilize_nlev(flux_constr, flux_A_params, min_nlev=2)
    with pytest.raises(AssertionError):
        opt.stabilize_nlev(flux_constr, flux_A_params,min_nlev=20, max_nlev=5)
    #test warnings about exceeding max_nlev
    with pytest.warns(UserWarning):
        opt.stabilize_nlev(flux_constr, flux_A_params, max_nlev=6)
    
    # test on transmon
    transmon, nlev_T = opt.stabilize_nlev(transmon_constr, transmon_params)
    assert all([isinstance(transmon[op], Qobj) for op in transmon_ops])
    assert is_diag(transmon['H']), 'Transmon hamiltonian should \
        be diagonal in QHO basis'
    assert all([transmon[op].dims==[[nlev_T],[nlev_T]] for op in transmon_ops]),\
        'Transmon operators of incorrect shape.'

def test_diagonalize_Qsys():
     flux_constr:Callable['...',Qsys] = subsystems.build_fluxonium_operators
     transmon_constr:Callable['...',Qsys] = subsystems.build_transmon_operators

     flux_A:Qsys = opt.stabilize_nlev(flux_constr, flux_A_params)[0]
     flux_B:Qsys = opt.stabilize_nlev(flux_constr, flux_B_params)[0]
     transmon:Qsys = opt.stabilize_nlev(transmon_constr, transmon_params)[0]

     flux_A_diag:Qsys = opt.diagonalize_Qsys(flux_A)[0]
     flux_B_diag:Qsys = opt.diagonalize_Qsys(flux_B)[0]
     transmon_diag:Qsys = opt.diagonalize_Qsys(transmon)[0]
     
     diag_systems:dict[str:Qsys] = {'flux_A': flux_A_diag,
                               'flux_B': flux_B_diag,
                               'transmon': transmon_diag}
     systems:dict[str:Qsys] = {'flux_A': flux_A,
                               'flux_B': flux_B,
                               'transmon': transmon}
     
     for lbl, system in diag_systems.items():
        #check type
        if 'flux' in lbl:
            assert all([isinstance(system[op],Qobj) for op in flux_ops])
        elif 'transmon' in lbl:
            assert all([isinstance(system[op],Qobj) for op in transmon_ops])
           
        #check systems are diagonal
        assert opt.is_diag(system['H']),'{} is not diagonal after running'.format(lbl)\
         + ' optimize_subsystem.diagonalize_Qsys'
        
     #check that eigenvalues did not change
     for lbl in systems:
         diag_sys:Qsys = diag_systems[lbl]
         orig_sys:Qsys = systems[lbl]
         assert (diag_sys['H'].eigenenergies()==orig_sys['H'].eigenenergies()).all(),\
         'Eigenenergies changed during diagonalization for {}'.format(lbl)
     # check that transmon system was left invariant
     for lbl in transmon:
         assert transmon[lbl]==transmon_diag[lbl],\
         'transmon operator {} not left invariant by diagonlization'.format(lbl)
    

def test_truncate_Qsys():
    nlev=5
    flux_constr:Callable['...',Qsys] = subsystems.build_fluxonium_operators
    transmon_constr:Callable['...',Qsys] = subsystems.build_transmon_operators

    flux_A:Qsys = opt.stabilize_nlev(flux_constr, flux_A_params)[0]
    flux_B:Qsys = opt.stabilize_nlev(flux_constr, flux_B_params)[0]
    transmon:Qsys = opt.stabilize_nlev(transmon_constr, transmon_params)[0]
    flux_A_diag, basis_A = opt.diagonalize_Qsys(flux_A)
    flux_B_diag, basis_B = opt.diagonalize_Qsys(flux_B)
    transmon_diag, basis_T = opt.diagonalize_Qsys(transmon)
     
    diag_systems:dict[str:Qsys] = {'flux_A': flux_A_diag,
                               'flux_B': flux_B_diag,
                               'transmon': transmon_diag}
    bases:dict[str:np.ndarray[Qobj]] = {'flux_A': basis_A,
                                        'flux_B': basis_B,
                                        'transmon': basis_T}
    trunc_systems:dict[str:Qsys] = {}
    trunc_bases:dict[str:Qsys] = {}
  
   
    #build truncated systems
    for lbl in diag_systems:
        trunc_sys, trunc_basis = opt.truncate_Qsys(diag_systems[lbl],
                                                   nlev,
                                                   bases[lbl])
        trunc_systems[lbl] = trunc_sys
        trunc_bases[lbl] = trunc_basis
    #check bases are truncated as expected:
    for lbl, trunc_basis in trunc_bases.items():
        full_basis = bases[lbl]
        for idx, trunc_eigenstate in enumerate(trunc_basis):
            full_eigenstate = full_basis[idx]
            assert trunc_eigenstate.dims==[[nlev],[1]],\
            'basis for {} truncated to incorrect shape'.format(lbl)
            assert (trunc_eigenstate.full()==full_eigenstate.full()[:nlev]).all(),\
            'basis for {} truncated incorrectly'.format(lbl)
    #check dims of truncated ops
    for lbl, system in trunc_systems.items():
        for op_name, op in system.items():
            assert op.dims==[[nlev],[nlev]],\
            'Operators for {} truncated to incorrect shape'.format(lbl)
    #check that untruncated eigenvalues of diagonal systems did not change
    for lbl in trunc_systems:
        trunc_system = trunc_systems[lbl]
        orig_system = diag_systems[lbl]
        assert opt.has_equal_energies(trunc_system,orig_system,nlev)

def test_build_optimized_system():
    flux_constr:Callable['...',Qsys] = subsystems.build_fluxonium_operators
    transmon_constr:Callable['...',Qsys] = subsystems.build_transmon_operators

    #build transmon system
    transmon, basis_T = opt.build_optimized_system(transmon_constr, transmon_params)
    #build fluxonia
    flux_A, basis_A = opt.build_optimized_system(flux_constr, flux_A_params)
    flux_B, basis_B = opt.build_optimized_system(flux_constr, flux_B_params)
    



        


     


