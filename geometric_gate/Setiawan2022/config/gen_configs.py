import yaml
import numpy as np

# To add a new pulse to simulation, add a dictionary entry
# of the form
#  {gate_label}:
#     geo_phase: {float}
#     omega_0: {float}
#     tg:{float}
#     dt: {float}
#     s0: {list of 3-char strings with chars g,e,f}
#     save_components:
#      {pulse_method_name}: {dictionary of method arg name keys and arg values, excluding the time argument `t`}
# The items omega_0, tg, dt, s0, and save_components are optional.

# gate_label must be a unique string identifier of the pulse
# omega_0 is the max amplitude of the pulse
# tg is the gate-time in ns
# dt is the size of the time-step in the simulation
# s0 lists the initial states to evolve under the pulse. One simulation will
#   be run per item in s0.
# If a method and its arguments are placed under save_components, that pulse
# method will be evaluated separately for each timestep. The method's value
# for all time t will be saved in
#     ../output/pulses/{gate_label}_{pulse_method_name}_{tg}ns_{omega_0}GHz_{dt}dt.npy
# for all entries in save_components, if a pulse method takes no arguments other than the
# current timestep `t`, then the arg dictionary may be left blank as:
#           save_components:
#               {pulse_method_name}:

# If any of the optional parameters are not provided, they will be taken from the default
# configuration in ./pulse_parameters.yaml, except for s0 which will be set to 'ggg'


def make_czs():
    base_configs: dict[str, float | dict[str, float]] =\
        {'dt': 0.001,
         'geo_phase': np.pi,
         'omega_0': 1.135,
         'tg': 130,
         't_ramp': 1.3,
         'save_components': {
             '_Pulse__omega_A': {},
             '_Pulse__omega_B': {},
             '_Pulse__delta_wmod': {'preprocess_t':
                                    {'_Pulse__g_ac':
                                     {'geo_phase': np.pi}}},
             '_Pulse__g_ac': {'geo_phase': np.pi}
         }}
    pulse_configs = {'CZ': base_configs}
    tg_list = np.arange(140, 150, 1)
    for tg in tg_list:
        t_ramp = tg * 0.01
        name = f'CZ-{tg}ns_tg-{t_ramp}ns_tramp'
        custom_config = base_configs.copy()
        custom_config['tg'] = tg
        custom_config['t_ramp'] = t_ramp
        pulse_configs[name] = custom_config
    return pulse_configs


if __name__ == '__main__':
    configs = make_czs()
    saved_configs = {}
    with open('pulses.yaml', 'r') as config_file:
        saved_configs = yaml.safe_load(config_file)
    saved_configs.update(configs)
    with open('pulses.yaml', 'w') as config_file:
        yaml.dump(config_file, saved_configs)
