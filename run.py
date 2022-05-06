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
name_of_sys = 'RLC'
short_window = 100
sys = eval(f'{name_of_sys}()')

t_arr, geo = attacked_state(eval(f'{name_of_sys}()'), short_window = short_window, type = 'geo')
t_arr, bias = attacked_state(eval(f'{name_of_sys}()'), short_window = short_window, type = 'bias')
t_arr, surge = attacked_state(eval(f'{name_of_sys}()'), short_window = short_window, type = 'surge')
# exit()

optimal_attack, Big_A, Big_B, small_c = short_mem_opt_attack(sys, steps = 100, short_window = short_window, verbose = 1)

# orig_Big_A = np.load(f'Big_A_{short_window}_orig.npy')
# orig_Big_B = np.load(f'Big_B_{short_window}_orig.npy')
# orig_small_c = np.load(f'small_c_{short_window}_orig.npy')
# print(f'\n\n\nBig_A analysis:')
# non_zero_ct = 0
# idxs = []
# for i in range(Big_A.shape[0]):
#     for j in range(Big_A.shape[1]):
#         if Big_A[i, j] - orig_Big_A[i, j] != 0:
#             idxs.append((i, j))
#             non_zero_ct += 1
# print(f'non-zero elements: {non_zero_ct}/{Big_A.shape[0] * Big_A.shape[1]}')
# print(idxs)
# print(f'\n\n\nBig_B analysis:')
# non_zero_ct = 0
# idxs = []
# for i in range(Big_B.shape[0]):
#     if Big_B[i, 0] - orig_Big_B[i] != 0:
#         print(i, Big_B[i, 0], orig_Big_B[i])
#         non_zero_ct += 1
# print(f'non-zero elements: {non_zero_ct}/{Big_B.shape[0]}')
# print(f'\n\n\nsmall_c analysis:')
# non_zero_ct = 0
# idxs = []
# for i in range(small_c.shape[0]):
#     if small_c[i, 0] - orig_small_c[i] != 0:
#         print(i, small_c[i, 0], orig_small_c[i])
#         non_zero_ct += 1
# print(f'non-zero elements: {non_zero_ct}/{small_c.shape[0]}')
# exit()

sys.optimal_attack = optimal_attack
t_arr, optimal_attacked_states = attacked_state(sys, short_window = short_window, type = 'optimal')

fig, ax = plt.subplots(figsize=(5, 3))

if name_of_sys == 'vt':
    y_label = 'Speed Difference'
    ax.set_ylim(0.9,1.35)
    ax.set_xlim(7.8,10) 
    ax.axhspan(2, 3, color='r', alpha=0.5)
elif name_of_sys == 'dc':
    y_label = 'Rotation Angle'
    # ax.set_ylim(0.9,1.35)
    # ax.set_xlim(99,121)
    ax.set_xlim(99, 121) 
elif name_of_sys == 'RLC':
    y_label = 'Voltage'
    ax.set_xlim(7.8,10) 
elif name_of_sys == 'ap':
    y_label = 'Pitch'
    ax.set_xlim(7.8,10) 

ax.plot(t_arr, geo, label='geometric', color = 'blue', linewidth=2, marker=markers.CARETDOWNBASE)
ax.plot(t_arr, surge, label='surge', color = 'r', linewidth=2, marker='p')
ax.plot(t_arr, bias, label='bias', color = 'brown', linewidth=2, marker='^')

ax.plot(t_arr, optimal_attacked_states, label='optimal', color = 'Orange', linewidth=2, marker='v')

reference_states = [sys.x_ref[sys.attacked_element_idx, 0]]*len(t_arr)
ax.plot(t_arr, reference_states, label='reference', color='black', ls='--')

ax.set_ylabel(y_label, fontsize=14)
ax.legend()
plt.savefig(f'./plot_{name_of_sys}_{short_window}.png')
