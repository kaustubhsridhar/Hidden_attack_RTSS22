from gettext import translation
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
import os 
from utils import *

translator = {'vt': 'Vehicle Turning', 'RLC': 'RLC Circuit', 'dc': 'DC Motor', 'ap': 'Aircraft Pitch'}

# Compute optimal hidden attack against long memory CUSUM detector
# Input: a sys, in [vt(), dc(), ap(), qd(), RLC(), DCS(), dcspeed()]
for name_of_sys in ['vt', 'RLC', 'dc']:
    for short_window in ['long']:
        sys = eval(f'{name_of_sys}()')

        t_arr, geo = attacked_state(eval(f'{name_of_sys}()'), short_window = short_window, type = 'geo')
        t_arr, bias = attacked_state(eval(f'{name_of_sys}()'), short_window = short_window, type = 'bias')
        t_arr, surge = attacked_state(eval(f'{name_of_sys}()'), short_window = short_window, type = 'surge')
        # exit()

        optimal_attack, Big_A, Big_B, small_c = long_mem_opt_attack(sys)

        sys.optimal_attack = optimal_attack
        t_arr, optimal_attacked_states = attacked_state(sys, short_window = short_window, type = 'optimal')

        fig, ax = plt.subplots(figsize=(8, 3))

        if name_of_sys == 'vt':
            y_label = 'Speed Diff. (m/s)'
            ax.set_ylim(0.9,1.35)
            ax.set_xlim(7.8,10) 
            ax.axhspan(2, 3, color='r', alpha=0.5)
        elif name_of_sys == 'dc':
            y_label = 'Rotation Angle\n(radians)'
            # ax.set_ylim(0.9,1.35)
            # ax.set_xlim(99,121)
            ax.set_xlim(99, 121) 
        elif name_of_sys == 'RLC':
            y_label = 'Voltage (V)'
            ax.set_xlim(7.8,10) 
        elif name_of_sys == 'ap':
            y_label = 'Pitch (radians)'
            ax.set_xlim(7.8,10) 


        sys.change_x_ref(   )
        ax.plot(t_arr, optimal_attacked_states, label='optimal', color = 'Orange', linewidth=2, marker='v') # need to have 3 of these lines for three different xrefs # for each line, you need to change x_ref in sim3_for_part3.py

        reference_states = [sys.x_ref[sys.attacked_element_idx, 0]]*len(t_arr) # 3 of these reference/uuper bound lines
        ax.plot(t_arr, reference_states, label='reference', color='black', ls='--')

        ax.set_xlabel('time (s)', fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        ax.set_title(translator[name_of_sys], fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)

        os.makedirs(f'./plots', exist_ok=True)
        plt.savefig(f'./plots/plot_{name_of_sys}_{short_window}.png', bbox_inches='tight')
