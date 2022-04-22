from control.matlab import ss, lsim, linspace, c2d
from functools import partial
from state_estimation import Estimator
import numpy as np
import math


class Exp:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape']
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
        # ---------------------
        self.limit_upbound = 2.2


class vt:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'xmeasure', 'xreal', 'att', 'cin', 'place', 'maxc', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape']
    count = 0
    A = [[-25 / 3]]
    B = [[5]]
    C = [[1]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.02

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 0.5
        self.i = 7
        self.d = 0
        self.total = 10
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [1]*501 # [0] * 251 + [1] * 250
        
        self.ref_initial_in_right_shape = np.array([[1]])

        self.thres = 5
        self.drift = 0 # 0.12 # can use drift = 0 for now.
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.xmeasure = 1
        self.xreal = 1
        self.score = []
        self.place = 400
        self.maxc = 5
        # --------------------------------------
        self.safeset = {'lo': [-2.7], 'up': [2.7]}

        # limit_upbound: max control can take, tunable <= max control (first control) under surge attack
        # run surge attack on given cusum_thres and cusum_drift, start from some high upper bound of surge attack
        # then decrease this upper bound until you have attack > upper bound at some time, limit_upbound = upper bound
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.inv(np.array(self.sysd.B))


class dc:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape']
    count = 0
    J = 0.01
    b = 0.1
    K = 0.01
    R = 1
    L = 0.5

    A = [[0, 1, 0],
         [0, -b / J, K / J],
         [0, -K / L, -R / L]]
    B = [[0], [0], [1 / L]]
    C = [[1, 0, 0]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.2

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 11
        self.i = 0
        self.d = 5
        self.total = 120
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [math.pi / 2] * 71 + [math.pi / 2] * 50

        self.ref_initial_in_right_shape = np.array([[math.pi / 2.0], [0], [0]])

        self.thres = 0.2
        self.drift = 0.01
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 200
        self.maxc = 20
        self.xmeasure = [[1.7], [0], [0]]
        self.xreal = [[1.7], [0], [0]]
        self.safeset = {'lo': [-4, -1000000000, -100000000], 'up': [4, 1000000000, 100000000]}
        # ---------------------
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.pinv(np.array(self.B))

class ap:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape']
    A = [[-0.313, 56.7, 0],
         [-0.0139, -0.426, 0],
         [0, 56.7, 0]]
    B = [[0.232], [0.0203], [0]]
    C = [[0, 0, 1]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.02

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 14
        self.i = 0.8
        self.d = 5.7
        self.total = 10
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [0] * 201 + [0.7] * 200 + [0.5] * 100

        self.ref_initial_in_right_shape = np.array([[0], [0.5], [0]])

        self.thres = 10
        self.drift = 1
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 300
        self.maxc = 20
        self.xmeasure = [[0], [0], [0]]
        self.xreal = [[0], [0], [0]]
        self.safeset = {'lo': [-100, -100, 0], 'up': [100, 100, 2]}
        # ---------------------
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.inv(np.array(self.B))

class qd:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape']
    count = 0
    g = 9.81
    m = 0.468
    A = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, -g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
    B = [[0], [0], [0], [0], [0], [0], [0], [0], [1 / m], [0], [0], [0]]
    C = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    D = [[0], [0], [0], [0], [0], [0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.02

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 0.1
        self.i = 0
        self.d = 0.6
        self.total = 30
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref =  [2] * 601 + [4] * 600 + [2] * 300
        self.thres = 3
        self.drift = 1
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 700
        self.maxc = 50
        self.xmeasure = [[0], [0], [0],[0], [0], [0],[0], [0], [0]]
        self.xreal = [[0], [0], [0],[0], [0], [0],[0], [0], [0]]
        self.safeset = {'lo': [-10000] * 8 + [-1], 'up': [10000] * 8 + [8]}# model
        # ---------------------
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.inv(np.array(self.B))
        
class RLC:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape'] 
    count = 0
    R = 10000
    L = 0.5
    C = 0.0001

    A = [[0, 1 / C], [-1 / L, -R / L]]
    B = [[0], [1 / L]]
    C = [[1, 0]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.02
    RLC_circuit = Exp(sysc_default, dt_default)
    exp = RLC_circuit
    exp.name = 'RLC Circuit'
    exp.x_0 = [[0], [0]]
    exp.y_0 = 0
    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 5
        self.i = 5
        self.d = 0
        self.total = 10
        self.ref  = [2] * 201 + [3] * 300

        self.ref_initial_in_right_shape = np.array([[3], [0]])

        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.thres = 3
        self.drift = 0.1
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 450
        self.maxc = 50
        self.xmeasure = np.array([0,0])
        self.xreal = np.array([0, 0])
        self.safeset = {'lo': [0, -5], 'up': [7, 5]} # model
        # ---------------------
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.inv(np.array(self.B))
        
class DCS:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape'] 

    A = [[0, 101, 0], [-6.6 ,-100, 50],[0,0,-598.8]]
    B = [[0], [0], [23952.1]]
    C = [[1, 0, 0]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.15
    RLC_circuit = Exp(sysc_default, dt_default)
    exp = RLC_circuit
    exp.name = 'RLC Circuit'
    exp.x_0 = [[0], [0]]
    exp.y_0 = 0
    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 5
        self.i = 5
        self.d = 0
        self.total = 30
        self.ref  = [800] * 201
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.thres = 10
        self.drift = 1
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0
        self.yreal = 0
        self.score = []
        self.place = 300
        self.maxc = 20
        self.xmeasure = np.array([[0], [0], [0]])
        self.xreal = np.array([[0], [0], [0]])
        self.safeset = {'lo': [0, -5], 'up': [7, 5]} # model
        # ---------------------
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.inv(np.array(self.B))
        
class dcspeed:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'ymeasure', 'yreal', 'att', 'cin', 'place', 'maxc', 'xreal', 'xmeasure', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape']
    count = 0
    J = 0.01
    b = 0.1
    K = 0.01
    R = 1
    L = 0.5

    A = [
         [ -b / J, K / J],
         [-K / L, -R / L]]
    B = [[0], [1 / L]]
    C = [[1, 0]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.2

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 35
        self.i = 20
        self.d = 1
        self.total = 120
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [math.pi / 2] * 71 + [math.pi / 2] * 50
        self.thres = 0.2
        self.drift = 0.01
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.ymeasure = 0.8
        self.yreal = 0.8
        self.score = []
        self.place = 500
        self.maxc = 20
        self.xmeasure = [[1], [0]]
        self.xreal = [[1], [0]]
        self.safeset = {'lo': [-4, -1000000000, -100000000], 'up': [4, 1000000000, 100000000]}
        # ---------------------
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.inv(np.array(self.B))

class vs_testbed:
    __slots__ = ['name', 'sysc', 'Ts', 'sysd', 'x_0', 'y_0', 'p', 'i', 'd', 'ref',
                 'est', 'slot', 't_arr', 't_attack', 't_detect', 'attacks', 'y_index',
                 'safeset', 'target_set', 'control_limit', 'max_k', 'worst_case_control', 'k', 'epsilon',
                 'sep_graph', 'y_label', 'x_left', 'y_up', 'y_lo', 'total', 'thres', 'drift', 'y_real_arr',
                 's', 'score', 'xmeasure', 'xreal', 'att', 'cin', 'place', 'maxc','yreal','ymeasure', 'limit_upbound', 'B_inv', 'ref_initial_in_right_shape']
    count = 0
    A = [[-0.6922]]
    B = [[0.1231]]
    C = [[1]]
    D = [[0]]
    sysc_default = ss(A, B, C, D)
    dt_default = 0.05

    def __init__(self, sysc=sysc_default, Ts=dt_default, epsilon=1e-7):
        self.sysd = c2d(sysc, Ts)
        self.Ts = Ts
        self.sysc = sysc
        self.epsilon = epsilon
        self.est = Estimator(self.sysd, 500, self.epsilon)
        self.y_index = None
        self.worst_case_control = 'current'
        self.k = None
        self.y_up = None
        self.y_lo = None
        # diy
        self.p = 30
        self.i = 0
        self.d = 0
        self.total = 25
        self.slot = int(self.total / Ts)
        self.t_arr = linspace(0, self.total, self.slot + 1)
        self.ref = [0] * 251 + [0.6] * 250
        self.thres = 5
        self.drift = 0.12
        self.x_0 = [0]
        self.y_real_arr = []
        self.s = 0
        self.att = 0
        self.cin = 0
        self.xmeasure = 1
        self.xreal = 1
        self.score = []
        self.place = 400
        self.maxc = 5
        self.yreal=0
        self.ymeasure=0
        # --------------------------------------
        self.safeset = {'lo': [-2.7], 'up': [2.7]}
        # ---------------------
        self.limit_upbound = 2.2
        self.B_inv = np.linalg.inv(np.array(self.B))

