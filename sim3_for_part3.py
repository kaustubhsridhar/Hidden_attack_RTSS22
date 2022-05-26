from control.matlab import ss, lsim, linspace, c2d
from functools import partial
from state_estimation import Estimator
import numpy as np
import math

class Exp:
    count = 0

    def __init__(self, sysc, Ts, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 150, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None


class vt:
    count = 0

    def __init__(self, epsilon=1e-7):
        # for utils.py>short_mem_opt_attack()
        # change only below!
        self.A = np.array([[-25 / 3]])
        self.A_dim = len(self.A)
        self.B = np.array([[5]])

        self.attacked_element_idx = 0
        self.C = np.array([[1]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.02
        self.x_ref = np.array([[1]]) # 0.8, 1.0, 1.2 ### TODO

        self.CUSUM_thresh = np.array([[5]])
        self.CUSUM_drift = np.array([[0]])
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[2.35]])

        self.total_time = 10

        self.alpha = 0.75 # for geo attack
        self.beta = 0.85 # for geo attack
        self.steps = 100
        self.u_lowbound = -1

        # no need to change
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.sysd = c2d(self.sysc, self.dt)
        self.Ad = np.array(self.sysd.A)
        self.Bd = np.array(self.sysd.B)
        self.Bd_inv = np.linalg.inv(self.Bd)
        self.u = -1 * self.Bd_inv @ (self.Ad - np.eye(self.A_dim)) @ self.x_ref
        self.I = np.eye(self.A_dim)
        self.Zero = np.zeros((self.A_dim, self.A_dim))

        # for utils.py>attacked_state()
        # check / change
        self.total_time_slots = int(self.total_time / self.dt) # = 10/0.02 = 500
        self.slot_for_start_of_attack = self.total_time_slots - self.steps
        self.t_arr = linspace(0, self.total_time, self.total_time_slots + 1)

class dc:
    count = 0

    def __init__(self, epsilon=1e-7):
        # for utils.py>short_mem_opt_attack()
        # change only below!
        J = 0.01
        b = 0.1
        K = 0.01
        R = 1
        L = 0.5
        self.A = np.array([[0, 1, 0],
         [0, -b / J, K / J],
         [0, -K / L, -R / L]])
        self.A_dim = len(self.A)
        self.B = np.array([[0], [0], [1 / L]])

        self.attacked_element_idx = 0
        self.C = np.array([[1, 0, 0]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.2
        self.x_ref = np.array([[math.pi * 1 / 2.0], [0], [0]]) # * 1, * 1.05, *0.9

        self.CUSUM_thresh = np.array([[16]])
        self.CUSUM_drift = np.array([[0]])
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[1.66]])

        self.total_time = 120

        self.alpha = 0.8 # for geo attack
        self.beta = 0.7 # for geo attack
        self.steps = 100
        self.u_lowbound = None

        # no need to change
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.sysd = c2d(self.sysc, self.dt)
        self.Ad = np.array(self.sysd.A)
        self.Bd = np.array(self.sysd.B)
        self.Bd_inv = np.linalg.pinv(self.Bd)
        self.u = -1 * self.Bd_inv @ (self.Ad - np.eye(self.A_dim)) @ self.x_ref
        # print(self.Bd_inv, (self.Ad - np.eye(self.A_dim)), self.x_ref)
        # print(self.u)
        # exit()
        self.I = np.eye(self.A_dim)
        self.Zero = np.zeros((self.A_dim, self.A_dim))

        # for utils.py>attacked_state()
        # check / change
        self.total_time_slots = int(self.total_time / self.dt) # = 120/0.2 = 600
        self.slot_for_start_of_attack = self.total_time_slots - self.steps 
        self.t_arr = linspace(0, self.total_time, self.total_time_slots + 1)

class RLC:
    count = 0

    def __init__(self, epsilon=1e-7):
        # for utils.py>short_mem_opt_attack()
        # change only below!
        R = 10000
        L = 0.5
        C = 0.0001

        self.A = np.array([[0, 1 / C], [-1 / L, -R / L]])
        self.A_dim = len(self.A)
        self.B = np.array([[0], [1 / L]])

        self.attacked_element_idx = 0
        self.C = np.array([[1, 0]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.02
        self.x_ref = np.array([[3], [0]]) # 2.8, 3, 3.2

        self.CUSUM_thresh = np.array([[5]])
        self.CUSUM_drift = np.array([[0]])
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[4.6]])

        self.total_time = 10

        self.alpha = 0.8 # for geo attack
        self.beta = 0.7 # for geo attack
        self.steps = 100
        self.u_lowbound = None

        # no need to change
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.sysd = c2d(self.sysc, self.dt)
        self.Ad = np.array(self.sysd.A)
        self.Bd = np.array(self.sysd.B)
        self.Bd_inv = np.linalg.pinv(self.Bd)
        self.u = -1 * self.Bd_inv @ (self.Ad - np.eye(self.A_dim)) @ self.x_ref
        self.I = np.eye(self.A_dim)
        self.Zero = np.zeros((self.A_dim, self.A_dim))

        # for utils.py>attacked_state()
        # check / change
        self.total_time_slots = int(self.total_time / self.dt) # = 10/0.02 = 500
        self.slot_for_start_of_attack = self.total_time_slots - self.steps
        self.t_arr = linspace(0, self.total_time, self.total_time_slots + 1)

class ap:
    count = 0

    def __init__(self, epsilon=1e-7):
        # for utils.py>short_mem_opt_attack()
        # change only below!

        self.A = np.array([[-0.313, 56.7, 0],
         [-0.0139, -0.426, 0],
         [0, 56.7, 0]])
        self.B = np.array([[0.232], [0.0203], [0]])
        self.C = np.array([[0, 0, 1]])
        self.D = np.array([[0]])

        self.A_dim = len(self.A)
        self.attacked_element_idx = 2
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.02
        self.x_ref = np.array([[0], [0], [0.7]])

        self.CUSUM_thresh = np.array([[3]])
        self.CUSUM_drift = np.array([[0.]])
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[0.7]])

        self.total_time = 10

        self.alpha = 0.8 # for geo attack
        self.beta = 0.7 # for geo attack
        self.steps = 100
        self.u_lowbound = None

        # no need to change
        self.sysc = ss(self.A, self.B, self.C, self.D)
        self.sysd = c2d(self.sysc, self.dt)
        self.Ad = np.array(self.sysd.A)
        self.Bd = np.array(self.sysd.B)
        self.Bd_inv = np.linalg.pinv(self.Bd)
        self.u = -1 * self.Bd_inv @ (self.Ad - np.eye(self.A_dim)) @ self.x_ref
        self.I = np.eye(self.A_dim)
        self.Zero = np.zeros((self.A_dim, self.A_dim))

        # for utils.py>attacked_state()
        # check / change
        self.total_time_slots = int(self.total_time / self.dt) # = 10/0.02 = 500
        self.slot_for_start_of_attack = self.total_time_slots - self.steps
        self.t_arr = linspace(0, self.total_time, self.total_time_slots + 1)