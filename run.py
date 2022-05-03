import cvxopt
from cvxopt import matrix, solvers
from sim3 import vt, dc, ap, qd, RLC, DCS, dcspeed
import numpy as np
import math
import control
import scipy.linalg as linalg
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib import markers
from control.matlab import lsim

from utils import *

# Compute optimal hidden attack against long memory CUSUM detector
# Input: a sys, in [vt(), dc(), ap(), qd(), RLC(), DCS(), dcspeed()]
name_of_sys = 'dc'
sys = eval(f'{name_of_sys}()')

optimal_attack = short_mem_opt_attack(sys, short_window = 25, verbose = 1)
sys.optimal_attack = optimal_attack
t_arr, optimal_attacked_states = attacked_state(sys, 'optimal')

fig, ax = plt.subplots(figsize=(5, 3))
ax.set_ylim(0.9,1.35)
ax.set_xlim(7.8,10)
t_arr, geo = attacked_state(eval(f'{name_of_sys}()'), 'geo')
t_arr, bias = attacked_state(eval(f'{name_of_sys}()'), 'bias')
t_arr, surge = attacked_state(eval(f'{name_of_sys}()'), 'surge')
ax.plot(t_arr, geo, label='geometric', color = 'blue', linewidth=2, marker=markers.CARETDOWNBASE)
ax.plot(t_arr, surge, label='surge', color = 'r', linewidth=2, marker='p')
ax.plot(t_arr, bias, label='bias', color = 'brown', linewidth=2, marker='^')

ax.plot(t_arr, optimal_attacked_states, label='optimal', color = 'Orange', linewidth=2, marker='v')

reference_states = [sys.x_ref[0,0]]*len(t_arr)
ax.plot(t_arr, reference_states, label='reference', color='black', ls='--')
ax.axhspan(2, 3, color='r', alpha=0.5)

if name_of_sys == 'vt':
    y_label = 'Speed Difference'
elif name_of_sys == 'dc':
    y_label = 'Rotation Angle'
elif name_of_sys == 'ap':
    y_label = 'Pitch'

ax.set_ylabel(y_label, fontsize=14)
ax.legend()
plt.savefig(f'./plot.png')
# fig.show()
