import gate
import sys


if __name__ == '__main__':
    argv = sys.argv
    use_gate_cache, use_pulse_cache, cache_gate =\
        list([kwrd not in sys.argv for kwrd in
              ['gate-nocache', 'pulse-nocache', 'gate-nosave']])
    gate.run_sim_from_configs(use_gate_cache=use_gate_cache,
                              use_pulse_cache=use_pulse_cache,
                              cache_gates=cache_gate)
