import numpy as np

to_rad_freq = 2*np.pi

circuit_conversions = {
    'A': {'E_C': to_rad_freq,
          'E_J': to_rad_freq,
          'E_L': to_rad_freq,
          'w': to_rad_freq},
    'B': {'E_C': to_rad_freq,
          'E_J': to_rad_freq,
          'E_L': to_rad_freq,
          'w': to_rad_freq},
    'C': {
        # 'U': to_rad_freq,
        'w': to_rad_freq
    },
    'interaction': {'g_AC': to_rad_freq,
                    'g_BC': to_rad_freq,
                    'g_AB': to_rad_freq}
}

pulse_conversions = {
    'A': {'w_mod': to_rad_freq},
    'B': {'w_mod': to_rad_freq},
    'omega_0': to_rad_freq
}


def get_conversions():
    return circuit_conversions, pulse_conversions
