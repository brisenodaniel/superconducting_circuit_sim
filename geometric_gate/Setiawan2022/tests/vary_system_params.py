# helper module used to vary simulation parameters
# in other test files. This file should not be used
# interactively
import sys 
sys.path.append('../src')
import numpy as np
import qutip as qt
import static_system 
from typing import Any, Callable
from subsystems import Subsystem
from composite_systems import CompositeSystem

def vary_mod_params(kwargs:dict[str,tuple[list[int],dict[str,Any]]],
                    constr:Callable['...',CompositeSystem]
                    )->dict[str,dict[str,list[int]|list[CompositeSystem]]]:
    varied_systems = {}
    for param_lbl, param_spec in kwargs.items():
        varied_param_range:list[int] = param_spec[0]
        static_params:dict[str,Any] = param_spec[1]
        systems = vary_mod_param(param_lbl, 
                                 varied_param_range,
                                 static_params,
                                 constr)
        systems_dict:dict[str, list[int]|list[CompositeSystem]] = {
            'param_range': varied_param_range,
            'systems': systems
        }
        varied_systems[param_lbl] = systems_dict
    return varied_systems

def vary_mod_param(param_lbl:str,
                   param_range:list[int],
                   static_params:dict[str,Any],
                   constr:Callable['...',CompositeSystem])->list[CompositeSystem]:
    varied_systems = []
    ct_params = static_system.get_params()
    for arg in param_range:
        kwargs = {param_lbl:arg,
                  **static_params}
        system = constr(ct_params, **kwargs)
        varied_systems.append(system)
    return varied_systems

def vary_property_params(kwargs:dict[str,tuple[list[int],dict[str,Any]]],
                         constr:Callable['...',CompositeSystem],
                         property_spec:Callable['...', list[float|int]],
                         property_lbl:str):
    varied_systems = vary_mod_params(kwargs,constr)
    for varied_param, varied_system in varied_systems.items():
        param_range:list[int] = varied_system['param_range']
        systems:list[CompositeSystem] = varied_system['systems']
        property_0 = property_spec(systems[0])
        properties:np.ndarray[np.ndarray[int|float]] = \
        np.empty((len(param_range),len(property_0)), dtype=type(property_0[0]))
        properties[0] = property_0
        for i, system in enumerate(systems,start=1):
            properties[i] = property_spec(system)
        varied_system[property_lbl] = properties
        varied_systems[varied_param] = varied_system
    return varied_systems

