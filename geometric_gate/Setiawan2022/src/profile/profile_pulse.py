import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import pstats
import cProfile
PROFILE = True
# setup Path
project_root = Path(__file__).parents[2]
config_dir = os.path.join(project_root, 'config')
sys.path.append(f'{project_root}/src')
if 1:  # hacky, but prevents linter from moving this befor path reconfig
    from pulse import Pulse
    import static_system
    from composite_systems import CompositeSystem
# set wd to src

ct_params = static_system.get_params(
    f'{config_dir}/circuit_parameters.yaml')['pulse_gen_ct']
p_params = static_system.get_params(
    f'{config_dir}/pulse_parameters.yaml')
ct = static_system.build_static_system(ct_params)
pulse = Pulse(p_params, ct)
if PROFILE:
    with cProfile.Profile() as pr:
        tlist = np.arange(0, 140, 0.1)
        pulse.delta_wC(1, np.pi)
        ps = pstats.Stats(pr).sort_stats('cumtime')
        pr.print_stats()

else:
    tlist = np.linspace(0, p_params['tg'], 500)
    res_list = pulse.get_integrand_func(tlist, 'ge0', np.pi/3)
    print(res_list)
    plt.plot(tlist, res_list)
    plt.savefig('integrand_shape_new.pdf')
