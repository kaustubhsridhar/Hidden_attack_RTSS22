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
        self.C = np.array([[1]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.02
        self.x_ref = np.array([[1]])
        self.attacked_element_idx = 0

        self.CUSUM_thresh = np.array([[5]])
        self.CUSUM_drift = np.array([[0]])
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[2.2]])

        self.total_time = 10

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
        # no need to change
        self.total_time_slots = int(self.total_time / self.dt) # = 10/0.02 = 500
        self.slot_for_start_of_attack = int(4/5 * self.total_time_slots) # = 400
        self.t_arr = linspace(0, self.total_time, self.total_time_slots + 1)

    def controller(self, x_measured, x_ref):
        return (-0.27698396)*(x_measured - x_ref)+self.u

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
        self.C = np.array([[1, 0, 0]])
        self.D = np.array([[0]])
        self.Q = np.eye(self.A_dim)
        self.R = np.array([[1]])
        self.dt = 0.2
        self.attacked_element_idx = 0
        self.x_ref = np.array([[math.pi / 2.0], [0], [0]])

        self.CUSUM_thresh = np.array([[0.2]])
        self.CUSUM_drift = np.array([[0]])
        self.x0 = np.copy(self.x_ref)
        self.u_upbound_tuned = np.array([[2.2]])

        self.total_time = 120

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
        # no need to change
        self.total_time_slots = int(self.total_time / self.dt) # = 120/0.2 = 600
        self.slot_for_start_of_attack = int(2/3 * self.total_time_slots) # = 200
        self.t_arr = linspace(0, self.total_time, self.total_time_slots + 1)

    def controller(self, x_measured, x_ref):
        return (-0.27698396)*(x_measured[self.attacked_element_idx, 0] - x_ref[self.attacked_element_idx, 0])+self.u
