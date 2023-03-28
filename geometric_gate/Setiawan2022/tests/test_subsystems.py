# Unit tests for subsystems.py
import qutip as qt
import numpy as np
import yaml
import subsystems
from typing import TypeAlias

Qobj: TypeAlias = qt.Qobj
Qsys: TypeAlias = dict[str,Qobj]
j: complex = complex(0,1)

#extract circuit parameters
with open('./config/circuit_parameters.yaml','r') as stream:
    try:
        ct_params:dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


#fluxonium A constructor parameters
param_lbls = ['E_C', 'E_J', 'E_L', 'phi_ext']
transmon_param_lbls = ['w','U']
flux_A_params:dict[str,float] = {lbl:ct_params['A'][lbl]\
                                 for lbl in param_lbls}
flux_B_params:dict[str,float] = {lbl:ct_params['B'][lbl]\
                                 for lbl in param_lbls}
transmon_params:dict[str,float] = {lbl:ct_params['C'][lbl] \
                                   for lbl in transmon_param_lbls}

def msg(err:str='', n:int=0)->str:
        err +="\n Simulated with {} energy levels".format(n)

def is_diag(op:Qobj)->bool:
     diags = op.diag()
     diag_op = qt.qdiags(diags,0)
     return op==diag_op

def test_build_fluxonium_operators():
    nlist = range(3,40+1)
    ops = ['n','phi','H']
    for nlev in nlist:
        flux_A_params['nlev']:int = nlev
        flux_B_params['nlev']:int = nlev
        flux_A:Qsys = subsystems.build_fluxonium_operators(**flux_A_params)
        flux_B:Qsys = subsystems.build_fluxonium_operators(**flux_B_params)
        assert all([op in flux_A for op in ops]), msg(n=nlev)
        assert all([op in flux_B for op in ops]), msg(n=nlev)
        assert all([isinstance(flux_A[op], Qobj) for op in ops]),\
        msg('Non qt.Qobj value in flux_A', nlev)
        assert all([isinstance(flux_B[op], Qobj) for op in ops]),\
        msg('Non qt.Qobj value in flux_B', nlev)
        assert flux_A['H'].isherm,\
        msg('Hamiltonian for Fluxonium A not hermitian',nlev)
        assert flux_B['H'].isherm,\
        msg('Hamiltonian for Fluxonium B not hermitian',nlev)
        assert all([flux_A[op].dims==[[nlev],[nlev]] for op in ops]),\
        msg('Operators in flux_A of incorrect shape')
        assert all([flux_B[op].dims==[[nlev],[nlev]] for op in ops]),\
        msg('Operators in flux_B of incorrect shape')
        assert not is_diag(flux_A['H']),\
        msg('Fluxonium A Hamiltonian should not be diagonal in QHO basis')
        assert not is_diag(flux_B['H']),\
        msg('Fluxonium B Hamiltonian should not be diagonal in QHO basis')
        

def test_build_transmon_operators():
     nlist = range(3,40+1)
     ops = ['a','H']
     for nlev in nlist:
          transmon_params['nlev']:int = nlev
          transmon:Qsys = subsystems.build_transmon_operators(**transmon_params)
          assert all([op in transmon for op in ops]), msg(n=nlev)
          assert all([isinstance(transmon[op], Qobj) for op in ops]),\
          msg('non qt.Qobj value in transmon',nlev)
          assert transmon['H'].isherm, \
          msg('Hamiltonian for transmon is not hermitian')
          assert all([transmon[op].dims ==[[nlev],[nlev]] for op in ops]),\
          msg('Operators in transmon of incorrect shape',nlev)
          assert is_diag(transmon['H']),\
          msg('Transmon Hamiltonian not Diagonal in QHO basis',nlev)


        
