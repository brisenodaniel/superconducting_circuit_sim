PROFILE=True
import cProfile
import pstats
import sys
sys.path.append('../')
from pulse import Pulse
import static_system
from composite_systems import CompositeSystem
import matplotlib.pyplot as plt
import numpy as np


ct_params = static_system.get_params('../../config/circuit_parameters.yaml')
p_params = static_system.get_params('../../config/pulse_parameters.yaml')
ct = static_system.build_static_system(ct_params)
pulse = Pulse(p_params, ct)
if PROFILE:
    with cProfile.Profile() as pr:
        pulse.delta_wC(1, np.pi)
        ps = pstats.Stats(pr).sort_stats('cumtime')
        pr.print_stats()

else:
    tlist = np.linspace(0, p_params['tg'], 500)
    res_list = pulse.get_integrand_func(tlist,'ge0',np.pi/3)
    print(res_list)
    plt.plot(tlist,res_list)
    plt.savefig('integrand_shape_new.pdf')

