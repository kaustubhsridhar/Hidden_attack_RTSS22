# General code running instructions
Set system name and short_window size in run.py. Then,
```
python run.py
```

# FOr CSR
```
from utils import short_mem_opt_attack_for_CSR
max_deviation_float = short_mem_opt_attack_for_CSR(sys, start_state, short_window = short_window)
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
