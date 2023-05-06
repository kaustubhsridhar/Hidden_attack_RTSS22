# Fail-Safe: Securing Cyber-Physical Systems against Hidden Sensor Attacks
This repository conatins the code for the paper available at https://ieeexplore.ieee.org/abstract/document/9984726

Set system name and short_window size in run.py. Then,
```
python run.py
```

# For CSR
```
from utils import short_mem_opt_attack_for_CSR, long_mem_opt_attack_for_CSR

max_deviation_float = short_mem_opt_attack_for_CSR(sys, start_state, short_window = 20)

max_deviation_float = long_mem_opt_attack_for_CSR(sys, start_state)

```

# Trajectories

Long
```
sys = RLC() # or vt() or dc()
optimal_attack, _, _, _ = long_mem_opt_attack(sys)
sys.optimal_attack = optimal_attack
t_arr, optimal_attacked_states = attacked_state(sys, short_window = short_window, type = 'optimal')
plt.plot(t_arr, optimal_attacked_states
```

Short
```
short_window = 10
sys = RLC() # or vt() or dc()
optimal_attack, _, _, _ = short_mem_opt_attack(sys, short_window=short_window)
sys.optimal_attack = optimal_attack
t_arr, optimal_attacked_states = attacked_state(sys, short_window = short_window, type = 'optimal')
plt.plot(t_arr, optimal_attacked_states
```


# Hidden_attack_RTSS22

Optimization files:
1. vt_optimization is the formulation and solving for vehicle turning, and it can output a series of sensor measurements for an
 optimal attack against long-memory detector.
2. short_memory_optimization is trying to against short-memory detector.

Visualization files:
1. short_memory_x: In these files, 2 common hiddden attacks have been implemented and compared with our optimal attack.
2. vt_motivation: Same implementation with long-memory detector.

Helper:
1.PID
2.state_estimation

Attacks formula can be found here: http://feihu.eng.ua.edu/NSF_CPS/year1/w8_3.pdf
