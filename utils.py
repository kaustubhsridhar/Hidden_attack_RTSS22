import cvxopt
from cvxopt import matrix, solvers
from sim3_new import *
import numpy as np
import math
import control
import scipy.linalg as linalg
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib import markers
from control.matlab import lsim
import cvxpy as cp

def short_mem_opt_attack(sys, short_window, verbose = False, method = 'CVXOPT'):
    # Solve CARE for optimal control K backward from infinity horizon
    P = np.array(linalg.solve_continuous_are(sys.A, sys.B, sys.Q, sys.R))
    K = np.array(linalg.inv(sys.R) * (sys.B.T @ P)) # shape = 1 x sys.A_dim

    """     We have :-- Big_A_{ (3*A_dim*sys.steps) x (A_dim*sys.steps)} Big_X_{ (A_dim*sys.steps) x 1} \leq Big_B_{ (3*A_dim*sys.steps) x 1}
            We have :-- 3*A_dim*sys.steps for rows because we have 3 sets of A_dim*sys.steps constrainst, namely, CUSUM, Control, and ABS_VALUE constraints
    """
    
    # Setup CUSUM constraints in Big_A in terms of blocks of matrices
    CUSUM_Big_A = []
    for row_idx in range(sys.steps):
        if row_idx < short_window:
            row = [sys.Ad - sys.Bd @ K - sys.I]*(row_idx)                                                                        + [-1*sys.I] + [sys.Zero]*(sys.steps-row_idx-1)
        else:
            row = [sys.Zero]*(row_idx-short_window) + [sys.Ad - sys.Bd @ K] + [sys.Ad - sys.Bd @ K - sys.I]*(short_window-1)     + [-1*sys.I] + [sys.Zero]*(sys.steps-row_idx-1)
        CUSUM_Big_A.append(row)
    CUSUM_Big_A = np.block(CUSUM_Big_A)
    # Setup CONTROL constraints in Big_A in terms of blocks of matrices
    # since K is a row matrix of shape = 1 x sys.A_dim, we pad it with zeros below to get the shape of sys.A_dim x sys.A_dim
    CONTROL_Big_A = []
    if sys.A_dim > 1:
        padding = np.array([[0 for _ in range(sys.A_dim)] for _ in range(sys.A_dim-1)])
        K_with_padding = np.vstack((-1*K, padding))
    else:
        K_with_padding = -1*K
    for row_idx in range(sys.steps):
        if row_idx == 0:
            row = [sys.Zero]*(sys.steps) # first row is zeros
        else:
            row = [sys.Zero]*(row_idx-1) + [K_with_padding] + [sys.Zero]*(sys.steps - row_idx)
        CONTROL_Big_A.append(row)
    CONTROL_Big_A = np.block(CONTROL_Big_A)
    # Setup ABS_VALUE constraints on Big_A, as follows:
    ## \hat{x}_i                                                    > \tilde{x}_i
    ## ==> A \tilde{x}_{i-1} + B u_{i-1}                            > \tilde{x}_i
    ## ==> A \tilde{x}_{i-1} + B (u0 + K(x_ref - \tilde{x}_{i-1}))  > \tilde{x}_i [from LQ, u_i = u0 + K(x_ref - \tilde{x}_i)]
    ## ==> -(A - B K) \tilde{x}_{i-1} - (I) \tilde{x}_i < B u0 + B K x_ref
    ABS_VALUE_Big_A = []
    for row_idx in range(sys.steps):
        if row_idx == 0:
            row = [sys.I] + [sys.Zero]*(sys.steps-1)
        else:
            row = [sys.Zero]*(row_idx-1) + [-1*(sys.Ad - sys.Bd @ K)] + [sys.I] + [sys.Zero]*(sys.steps - row_idx - 1)
        ABS_VALUE_Big_A.append(row)
    ABS_VALUE_Big_A = np.block(ABS_VALUE_Big_A)
    # Finally, we have, for Big_A...
    Big_A = np.vstack((CUSUM_Big_A, CONTROL_Big_A, ABS_VALUE_Big_A))
    assert (Big_A.shape[0] == 3*sys.A_dim*sys.steps and Big_A.shape[1] == sys.A_dim*sys.steps), f"actual: {Big_A.shape} [CUSUM {CUSUM_Big_A.shape}, CONTROL {CONTROL_Big_A.shape}, ABS_VALUE {ABS_VALUE_Big_A.shape}], expected: {(3*sys.A_dim*sys.steps, sys.A_dim*sys.steps)}"

    """     We have the same 3 sets of constrainst for Big_B also as follows 
    """
    # CUSUM 
    CUSUM_Big_B = []
    # sys.u = np.array([[1.667]]) # we use exact u unlike in Mengyu's ipyhton-notebooks which uses this!
    for row_idx in range(sys.steps):
        if row_idx < short_window:
            row = [sys.CUSUM_thresh - (sys.Ad - sys.Bd @ K) @ sys.x0 - (row_idx+1)*(sys.Bd @ K @ sys.x_ref + sys.Bd @ sys.u - sys.CUSUM_drift)]
        else:
            row = [sys.CUSUM_thresh - (short_window)*(sys.Bd @ K @ sys.x_ref + sys.Bd @ sys.u - sys.CUSUM_drift)]
        CUSUM_Big_B.append(row)
    CUSUM_Big_B = np.block(CUSUM_Big_B)
    # CONTROL
    CONTROL_Big_B = []
    term = sys.u_upbound_tuned - K @ sys.x_ref - sys.u
    if sys.A_dim > 1:
        column_padding = np.array([[0] for _ in range(sys.A_dim-1)])
        term_with_padding = np.vstack((term, column_padding))
    else:
        term_with_padding = term
    for row_idx in range(sys.steps):
        row = [term_with_padding]
        CONTROL_Big_B.append(row)
    CONTROL_Big_B = np.block(CONTROL_Big_B)
    # ABS_VALUE ##################### PREVIOUSLY and NOW, \hat{x}_{i+1} missing
    ABS_VALUE_Big_B = []
    for row_idx in range(sys.steps):
        row = [sys.Bd @ sys.u + sys.Bd @ K @ sys.x_ref]
        ABS_VALUE_Big_B.append(row)
    ABS_VALUE_Big_B = np.block(ABS_VALUE_Big_B)
    # Finally, we have, Big_B...
    Big_B = np.vstack((CUSUM_Big_B, CONTROL_Big_B, ABS_VALUE_Big_B))
    assert (Big_B.shape[0] == 3*sys.A_dim*sys.steps and Big_B.shape[1] == 1), f"actual: {Big_B.shape} [CUSUM {CUSUM_Big_B.shape}, CONTROL {CONTROL_Big_B.shape}, ABS_VALUE {ABS_VALUE_Big_B.shape}], expected: {(3*sys.A_dim*sys.steps, 1)}"

    """     and OPTIMIZATION Below...
    """
    sols = []
    objs = []
    for f in range(1, sys.steps): # we don't need to solve first problem!
        """     OBJECTIVE FN. =  small_C_{ (A_dim*sys.steps) x (1)} ^T @ Big_X
        """
        small_c = []
        for i in range(sys.steps):
            if i < f:
                matrix_coeff_of_x = np.linalg.matrix_power(sys.Ad, f - i - 1) @ sys.Bd @ K
                row_corresponding_to_attacked_element_reshaped_to_col = matrix_coeff_of_x[sys.attacked_element_idx, :].reshape(sys.A_dim, 1)
                row = [ row_corresponding_to_attacked_element_reshaped_to_col ]
            else:
                row = [ np.zeros((sys.A_dim, 1)) ]
            small_c.append(row)
        small_c = np.block(small_c)
        assert (small_c.shape[0] == sys.A_dim*sys.steps and small_c.shape[1] == 1), f"actual: {small_c.shape}, expected: {(sys.A_dim*sys.steps, 1)}"

        if method == 'CVXOPT':
            sol = solvers.lp(matrix(small_c, (len(small_c), 1), 'd'), matrix(Big_A), matrix(Big_B, (len(Big_B), 1), 'd'), solver='glpk')
            sols.append(list(sol['x']))
            objs.append(sol['primal objective'])

        elif method == 'CVXPY':
            Big_X = cp.Variable((sys.A_dim * sys.steps, 1))
            prob = cp.Problem(cp.Minimize(small_c.T @ Big_X),
                            [Big_A @ Big_X <= Big_B])
            prob.solve()
            sols.append(Big_X.value)
            objs.append(prob.value)

    sols = np.array(sols) # shape = (sys.steps-1) * sys.steps
    objs = np.array(objs) # shape = (sys.steps-1)
    j = np.argmax(objs)
    return sols[-1*sys.A_dim], Big_A, Big_B, small_c # sols[-sys.A_dim], Big_A, Big_B, small_c


def long_mem_opt_attack(sys, verbose = False, method = 'CVXOPT'):
    # Solve CARE for optimal control K backward from infinity horizon
    P = np.array(linalg.solve_continuous_are(sys.A, sys.B, sys.Q, sys.R))
    K = np.array(linalg.inv(sys.R) * (sys.B.T @ P)) # shape = 1 x sys.A_dim

    """     We have :-- Big_A_{ (3*A_dim*sys.steps) x (A_dim*sys.steps)} Big_X_{ (A_dim*sys.steps) x 1} \leq Big_B_{ (3*A_dim*sys.steps) x 1}
            We have :-- 3*A_dim*sys.steps for rows because we have 3 sets of A_dim*sys.steps constrainst, namely, CUSUM, Control, and ABS_VALUE constraints
    """
    
    # Setup CUSUM constraints in Big_A in terms of blocks of matrices
    CUSUM_Big_A = []
    for row_idx in range(sys.steps):
        row = [sys.Ad - sys.Bd @ K - sys.I]*(row_idx) + [-1*sys.I] + [sys.Zero]*(sys.steps-row_idx-1)
        CUSUM_Big_A.append(row)
    CUSUM_Big_A = np.block(CUSUM_Big_A)
    # Setup CONTROL constraints in Big_A in terms of blocks of matrices
    # since K is a row matrix of shape = 1 x sys.A_dim, we pad it with zeros below to get the shape of sys.A_dim x sys.A_dim
    CONTROL_Big_A = []
    if sys.A_dim > 1:
        padding = np.array([[0 for _ in range(sys.A_dim)] for _ in range(sys.A_dim-1)])
        K_with_padding = np.vstack((-1*K, padding))
    else:
        K_with_padding = -1*K
    for row_idx in range(sys.steps):
        if row_idx == 0:
            row = [sys.Zero]*(sys.steps) # first row is zeros
        else:
            row = [sys.Zero]*(row_idx-1) + [K_with_padding] + [sys.Zero]*(sys.steps - row_idx)
        CONTROL_Big_A.append(row)
    CONTROL_Big_A = np.block(CONTROL_Big_A)
    # Setup ABS_VALUE constraints on Big_A, as follows:
    ## \hat{x}_i                                                    > \tilde{x}_i
    ## ==> A \tilde{x}_{i-1} + B u_{i-1}                            > \tilde{x}_i
    ## ==> A \tilde{x}_{i-1} + B (u0 + K(x_ref - \tilde{x}_{i-1}))  > \tilde{x}_i [from LQ, u_i = u0 + K(x_ref - \tilde{x}_i)]
    ## ==> -(A - B K) \tilde{x}_{i-1} - (I) \tilde{x}_i < B u0 + B K x_ref
    ABS_VALUE_Big_A = []
    for row_idx in range(sys.steps):
        if row_idx == 0:
            row = [sys.I] + [sys.Zero]*(sys.steps-1)
        else:
            row = [sys.Zero]*(row_idx-1) + [-1*(sys.Ad - sys.Bd @ K)] + [sys.I] + [sys.Zero]*(sys.steps - row_idx - 1)
        ABS_VALUE_Big_A.append(row)
    ABS_VALUE_Big_A = np.block(ABS_VALUE_Big_A)
    # Finally, we have, for Big_A...
    Big_A = np.vstack((CUSUM_Big_A, CONTROL_Big_A, ABS_VALUE_Big_A))
    assert (Big_A.shape[0] == 3*sys.A_dim*sys.steps and Big_A.shape[1] == sys.A_dim*sys.steps), f"actual: {Big_A.shape} [CUSUM {CUSUM_Big_A.shape}, CONTROL {CONTROL_Big_A.shape}, ABS_VALUE {ABS_VALUE_Big_A.shape}], expected: {(3*sys.A_dim*sys.steps, sys.A_dim*sys.steps)}"

    """     We have the same 3 sets of constrainst for Big_B also as follows 
    """
    # CUSUM 
    CUSUM_Big_B = []
    # sys.u = np.array([[1.667]]) # we use exact u unlike in Mengyu's ipyhton-notebooks which uses this!
    for row_idx in range(sys.steps):
        row = [sys.CUSUM_thresh - (sys.Ad - sys.Bd @ K) @ sys.x0 - (row_idx+1)*(sys.Bd @ K @ sys.x_ref + sys.Bd @ sys.u - sys.CUSUM_drift)]
        CUSUM_Big_B.append(row)
    CUSUM_Big_B = np.block(CUSUM_Big_B)
    # CONTROL
    CONTROL_Big_B = []
    term = sys.u_upbound_tuned - K @ sys.x_ref - sys.u
    if sys.A_dim > 1:
        column_padding = np.array([[0] for _ in range(sys.A_dim-1)])
        term_with_padding = np.vstack((term, column_padding))
    else:
        term_with_padding = term
    for row_idx in range(sys.steps):
        row = [term_with_padding]
        CONTROL_Big_B.append(row)
    CONTROL_Big_B = np.block(CONTROL_Big_B)
    # ABS_VALUE ##################### PREVIOUSLY and NOW, \hat{x}_{i+1} missing
    ABS_VALUE_Big_B = []
    for row_idx in range(sys.steps):
        row = [sys.Bd @ sys.u + sys.Bd @ K @ sys.x_ref]
        ABS_VALUE_Big_B.append(row)
    ABS_VALUE_Big_B = np.block(ABS_VALUE_Big_B)
    # Finally, we have, Big_B...
    Big_B = np.vstack((CUSUM_Big_B, CONTROL_Big_B, ABS_VALUE_Big_B))
    assert (Big_B.shape[0] == 3*sys.A_dim*sys.steps and Big_B.shape[1] == 1), f"actual: {Big_B.shape} [CUSUM {CUSUM_Big_B.shape}, CONTROL {CONTROL_Big_B.shape}, ABS_VALUE {ABS_VALUE_Big_B.shape}], expected: {(3*sys.A_dim*sys.steps, 1)}"

    """     and OPTIMIZATION Below...
    """
    sols = []
    objs = []
    for f in range(1, sys.steps): # we don't need to solve first problem!
        """     OBJECTIVE FN. =  small_C_{ (A_dim*sys.steps) x (1)} ^T @ Big_X
        """
        small_c = []
        for i in range(sys.steps):
            if i < f:
                matrix_coeff_of_x = np.linalg.matrix_power(sys.Ad, f - i - 1) @ sys.Bd @ K
                row_corresponding_to_attacked_element_reshaped_to_col = matrix_coeff_of_x[sys.attacked_element_idx, :].reshape(sys.A_dim, 1)
                row = [ row_corresponding_to_attacked_element_reshaped_to_col ]
            else:
                row = [ np.zeros((sys.A_dim, 1)) ]
            small_c.append(row)
        small_c = np.block(small_c)
        assert (small_c.shape[0] == sys.A_dim*sys.steps and small_c.shape[1] == 1), f"actual: {small_c.shape}, expected: {(sys.A_dim*sys.steps, 1)}"

        if method == 'CVXOPT':
            sol = solvers.lp(matrix(small_c, (len(small_c), 1), 'd'), matrix(Big_A), matrix(Big_B, (len(Big_B), 1), 'd'), solver='glpk')
            sols.append(list(sol['x']))
            objs.append(sol['primal objective'])

        elif method == 'CVXPY':
            Big_X = cp.Variable((sys.A_dim * sys.steps, 1))
            prob = cp.Problem(cp.Minimize(small_c.T @ Big_X),
                            [Big_A @ Big_X <= Big_B])
            prob.solve()
            sols.append(Big_X.value)
            objs.append(prob.value)

    sols = np.array(sols) # shape = (sys.steps-1) * sys.steps
    objs = np.array(objs) # shape = (sys.steps-1)
    j = np.argmax(objs)
    return sols[20*sys.A_dim], Big_A, Big_B, small_c

def attacked_state(sys, short_window, type = 'surge'):
    # AGAIN (See Above): Solve CARE for optimal control K backward from infinity horizon
    P = np.array(linalg.solve_continuous_are(sys.A, sys.B, sys.Q, sys.R))
    K = np.array(linalg.inv(sys.R) * (sys.B.T @ P)) # shape = 1 x sys.A_dim
    # print(K)
    # exit()

    # if type == 'surge':
    #     sys.slot_for_start_of_attack -= 1
    if type == 'geo':
        previous = sys.slot_for_start_of_attack
        alpha = sys.alpha
        beta = sys.beta
        coef = 1

    x_measured = sys.x0
    x_real = sys.x0
    CUSUM_score = 0
    atk = 0
    scores = []
    y_real_arr = []
    Us = []
    for slot_idx in range(sys.total_time_slots+1):
        # print(f'simulation step {slot_idx}')
        # step
        cin = sys.u + K @ (sys.x_ref - x_measured)
        Us.append(cin)
        if cin > sys.u_upbound_tuned:
            cin = sys.u_upbound_tuned
        elif sys.u_lowbound and cin < sys.u_lowbound:
            cin = sys.u_lowbound
        cin = cin[0, 0] # converts (1,1) array to float
            
        yout, T, xout = lsim(sys.sysc, cin, [0, sys.dt], x_measured)
        yout_real, T_real, xout_real = lsim(sys.sysc, cin, [0, sys.dt], x_real)

        x_real = xout_real[-1].reshape(sys.A_dim, 1)
        x_pred = xout[-1].reshape(sys.A_dim, 1)
        x_measured = xout[-1].reshape(sys.A_dim, 1) - atk

        if slot_idx > sys.slot_for_start_of_attack and type == 'optimal':
            # overwrite x_measured
            x_measured = sys.optimal_attack[slot_idx - sys.slot_for_start_of_attack - 1]

        if slot_idx > sys.slot_for_start_of_attack:
            # compute atk
            if short_window == 'long':
                if type == 'surge':
                    if slot_idx == sys.slot_for_start_of_attack + 1:
                        atk = sys.CUSUM_thresh + sys.CUSUM_drift
                    else:
                        atk = sys.CUSUM_drift
                elif type == 'bias': 
                    atk = sys.CUSUM_thresh/sys.steps+ sys.CUSUM_drift 
                elif type == 'geo':
                    atk = coef*beta*(alpha**(sys.total_time_slots + 1 - slot_idx)) 
            else:
                if type == 'surge':
                    if (slot_idx - sys.slot_for_start_of_attack) % short_window == 0: 
                        atk = sys.CUSUM_thresh + sys.CUSUM_drift
                    else:
                        atk = sys.CUSUM_drift
                elif type == 'bias': 
                    atk = sys.CUSUM_thresh/short_window + sys.CUSUM_drift 
                elif type == 'geo':
                    if (slot_idx - sys.slot_for_start_of_attack) % short_window == 0: 
                        previous = slot_idx
                    atk = coef*beta*(alpha**(previous + short_window + 1 - slot_idx)) 
            # compute CUSUM_score
            CUSUM_score = CUSUM_score + abs(x_pred-x_measured)-sys.CUSUM_drift
        
        # save
        scores.append(CUSUM_score)
        y_real_arr.append(x_real[sys.attacked_element_idx])
        
    # print(type, y_real_arr)
    # print(f'Control inputs are: {Us}')
    return sys.t_arr, y_real_arr

