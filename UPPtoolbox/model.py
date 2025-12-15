from collections.abc import Callable
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import scipy.io
from scipy.signal import butter, filtfilt
import warnings
import pandas
import seaborn as sns
import warnings
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import math
from scipy.stats import zscore
from scipy.signal import savgol_filter
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import minimize
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


'''
Parent class "UPPmodel" will contain everything common to the following three classes of models: 
    1. Ornstein-Uhlenbeck (linear)
    2. Non-linear with fixed gain
    3. Non-linear with modulated gain (SNR-sensitive)
    4. Non-linear with modulated gain (naive to SNR)

For this purpose, now "gain" is a function of (input, category) with parameters that can be chosen (for example, a 2-degrees polynomial of the input).

UPPmodel can run simulations based on its parameters with the Euler-Maruyama algorithm.
'''

#####################################################################################################################
#####################################################################################################################
#######################################               root class              #######################################
#####################################################################################################################
#####################################################################################################################


class UPP_abstract(ABC):

    _param_names = ["tau", "process_noise", "measure_noise"]

    
    def __init__(self, tau: float, process_noise: float, measure_noise: float):
        self.t = 0 # model clock
        self.dt = 1
        self.tau = tau
        self.process_noise = process_noise
        self.measure_noise = measure_noise
        

    # ------------------------- PARAMETERS HANDLING -------------------------

    def get_params(self) -> dict[str, float]:
        """Return dict of parameters for this model."""
        return {name: getattr(self, name) for name in self._param_names}

    def set_params(self, new_params: dict[str, float]) -> None:
        """Update parameters from dict."""
        for name in self._param_names:
            if name in new_params:
                setattr(self, name, new_params[name])

    def save_params(self, save_path: str) -> None:
        with open(save_path, "w") as f:
            json.dump(self.get_params(), f)

    def load_params(self, save_path: str) -> None:
        with open(save_path, "r") as f:
            params = json.load(f)
        self.set_params(params)

    # convenience for fitting
    def set_params_from_list(self, params_list: list) -> None:
        """Update parameters from list, order follows _param_names."""
        for name, val in zip(self._param_names, params_list):
            setattr(self, name, val)

    
    # ------------------------- MODEL STRUCTURE -------------------------
    
    @abstractmethod
    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        pass

    @abstractmethod
    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        pass
        
    # ------------------------- CORE DEFINITION -------------------------
    
    # core model: f(x(t)) = x(t+1)
    def core(self, state: float | np.ndarray, input_value: float | np.ndarray, signal_category: int) -> np.ndarray:
        '''
        State and input_value are arrays of trials and inputs sampled at one time-point, 
            ex : state is the value of the signal for each trial at time 100 ms, input_value is array of inputs at the corresponding time.
            Returns the "state" values for the next time-step (e.g. at time 110 ms).
        '''
        # assert state.shape == input_value.shape, f'State and input must have same shape per time-step, but are resp. {state.shape} and {input_value.shape}.'
        linear_component = - state / self.tau
        input_component = self.input_function(input_value, signal_category)
        nonlinear_component = self.nonlinearity(state, input_value, signal_category)
        return state + (linear_component + input_component + nonlinear_component) *self.dt


    # ------------------------- SIMULATION TOOLS -------------------------
    
    # simulate trajectories with Euler-Maruyama
    def euler_maruyama(self, initial_state: np.ndarray, input_series: np.ndarray, signal_category: int) -> np.ndarray:
        trajectories = np.zeros_like(input_series)
        trajectories[:, 0] = initial_state
        for t, input_value in enumerate(input_series[:,:-1].T):
            trajectories[:, t+1] = self.core(trajectories[:, t], input_value, signal_category) + np.random.normal(0, self.process_noise*np.sqrt(self.dt), size = input_value.shape[0])
        return trajectories

    # add measurement noise
    def measure_simulations(self, initial_state: np.ndarray, input_series: np.ndarray, signal_category: int) -> np.ndarray:
        trajectories = self.euler_maruyama(initial_state, input_series, signal_category)
        return trajectories +  np.random.normal(0, self.measure_noise, size = trajectories.shape)

    
    # ------------------------- FITTING TOOLS -------------------------

    # to compute the loglikelihood with a specific method (e.g., Kalman, Unscented Kalman, Gaussian-Mixture Unscented Kalman, ...)
    @abstractmethod
    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]) -> float:
        pass

    # maximize likelihood to fit the model parameters
    def fit(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray], init_params: list, bounds: list, fixed_params: list=[], l2_tau: float=0, l1_tau: float=0, feedback: bool=False) -> None:
    
        # Identify which parameters to optimize
        free_indices = [i for i, name in enumerate(self._param_names) if name not in fixed_params]
        free_names = [self._param_names[i] for i in free_indices]
    
        def to_minimize(p_free):
            full_params = init_params.copy() # the values of the fixed parameters must be in init_params
            for i, idx in enumerate(free_indices):
                full_params[idx] = p_free[i]
            self.set_params_from_list(full_params)
            return -self.loglikelihood(state_series, input_series) + l1_tau * self.tau + l2_tau * self.tau**2

        init_free = [init_params[i] for i in free_indices]
        bounds_free = [bounds[i] for i in free_indices]
        result = minimize(to_minimize, init_free, bounds=bounds_free)
    
        # Apply the fitted parameters
        full_params = init_params.copy()
        for i, idx in enumerate(free_indices):
            full_params[idx] = result.x[i]
        self.set_params_from_list(full_params)
        
        print(f'Optimisation success : {result.success}. \nFinal log-likelihood evaluation : {result.fun}.') if feedback else None


    # ------------------------- MODEL VISUALIZATION -------------------------

    def plot_model(self, states: np.ndarray, conditions: list, colormap: list, labels: list, save_path=None) -> None:
        """ conditions is a list [(input_1, signal_category_1), (input_2, signal_category_2), ..., (input_n, signal_category_n)] """
        plt.figure(figsize=(10,5))
        plt.plot([states[0], states[-1]], [0,0], linestyle='--', color='black')
        for cond_ind, (input_value, signal_category) in enumerate(conditions):
            plt.plot(states, self.core(states, input_value*np.ones_like(states), signal_category) - states, color=colormap[cond_ind], label=labels[cond_ind])
        plt.xlabel('state')
        plt.ylabel('d states / dt')
        plt.savefig(save_path) if save_path != None else None
        plt.show()
    


#####################################################################################################################
#####################################################################################################################
#######################################           Linear models               #######################################
#####################################################################################################################
#####################################################################################################################

class Linear(UPP_abstract):
    """ dx/dt = -x + input_weight * input """

    _param_names = UPP_abstract._param_names + ['input_weight']
    
    def __init__(self, tau: float, process_noise: float, measure_noise: float, input_weight: float):
        self.input_weight = input_weight
        super().__init__(tau, process_noise, measure_noise)

    # ------------------------- MODEL STRUCTURE -------------------------

    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return self.input_weight * input_value

    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return np.zeros_like(state)

    # ------------------------- FITTING TOOLS -------------------------

    # to compute the loglikelihood with regular Kalman filtering
    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]) -> float:
        # not categorical
        state_series = state_series[list(state_series.keys())[0]]
        input_series = input_series[list(input_series.keys())[0]]
        
        A = 1 - self.dt / self.tau                          # discrete version of np.exp(- self.dt / self.tau)
        B = self.input_weight
        Q = self.process_noise**2 / self.dt                 # discrete version of (self.tau*self.process_noise**2 / 2) * (1 - np.exp(-2 * self.dt / self.tau))
        R = self.measure_noise**2
        
        total_log_likelihood = 0.0
        for trial_id in range(state_series.shape[0]):
            states = state_series[trial_id]
            inputs = input_series[trial_id]
            
            x_pred = states[0]
            P_pred = 1.0                                    # initial variance estimate
            
            for t in range(1, len(states)):
                input_value = inputs[t-1]
                x_pred = A * x_pred + B * input_value       # Predict state x
                P_pred = A**2 * P_pred + Q                  # Predict uncertainty on state P
                S = P_pred + R                              # Update residual variance S
                K = P_pred / S                              # Optimal Kalman gain
                innovation = states[t] - x_pred              # Residual
                x_pred = x_pred + K * innovation            # Update state x
                P_pred = (1 - K) * P_pred                   # Update uncertainty on state P
                total_log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation**2) / S)
        return total_log_likelihood




class StratifiedLinear(UPP_abstract):
    """ dx/dt = -x + w_{category} * input """

    _param_names = UPP_abstract._param_names + ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    
    def __init__(self, tau: float, process_noise: float, measure_noise: float, w0: float=0, w1: float=0, w2: float=0, w3: float=0, w4: float=0, w5: float=0, w6: float=0):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        super().__init__(tau, process_noise, measure_noise)

    # ------------------------- MODEL STRUCTURE -------------------------

    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        input_weight = getattr(self, 'w' + str(signal_category))
        return input_weight * input_value

    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return np.zeros_like(state)

    # ------------------------- FITTING TOOLS -------------------------
    # to compute the loglikelihood with UKF
    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]):
        n_jobs = 8
        batch_size = 20

        # ---- Parameters ----
        dt, tau = self.dt, self.tau
        R = self.measure_noise**2
        process_noise = self.process_noise
        Q = self.process_noise**2 / self.dt                 # discrete version of (self.tau*self.process_noise**2 / 2) * (1 - np.exp(-2 * self.dt / self.tau))
        eps = 0 # used to be 1e-12

        # ---- Sigma points ----
        SigmaPoints = MerweScaledSigmaPoints
        sig_alpha, sig_beta, sig_kappa = 0.1, 2.0, 1.0
        UKF_class = UKF
        core_func = self.core

        # ---- Worker for one batch of one category ----
        def process_batch(batch_states, batch_inputs, signal_category):
            total_ll = 0.0
            for states, inputs in zip(batch_states, batch_inputs):
                sigmas = SigmaPoints(n=1, alpha=sig_alpha, beta=sig_beta, kappa=sig_kappa)
                ukf = UKF_class(
                    dim_x=1, dim_z=1,
                    fx=lambda x, dt_local, inp=inputs[0]: core_func(x, inp, signal_category)[0],
                    hx=lambda x: x,
                    dt=dt, points=sigmas
                )
                ukf.x = np.array([states[0]])
                ukf.P = np.eye(1)
                ukf.Q = np.eye(1) * Q
                ukf.R = np.eye(1) * R

                # ---- iterate through time ----
                for t in range(1, len(states)):
                    ukf.fx = lambda x, dt_local, inp=inputs[t - 1]: core_func(x, inp, signal_category)[0]
                    ukf.predict()
                    ukf.update(states[t])

                    r = ukf.y[0]
                    S = ukf.S[0, 0]
                    total_ll += -0.5 * (np.log(2 * np.pi * S + eps) + (r ** 2) / (S + eps))

            return total_ll

        # ---- Build category-wise batches ----
        tasks = []
        for signal_category in input_series.keys():
            states_cat = state_series[signal_category]
            inputs_cat = input_series[signal_category]
            n_trials = states_cat.shape[0]
            batches = [
                (states_cat[i:i + batch_size], inputs_cat[i:i + batch_size], signal_category)
                for i in range(0, n_trials, batch_size)
            ]
            tasks.extend(batches)

        # ---- Parallel execution ----
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_batch)(bs, bi, cat) for bs, bi, cat in tasks
        )

        return float(np.sum(results))


    # to compute the loglikelihood with regular Kalman filtering
    def loglikelihood_kalman(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]) -> float:
        
        A = 1 - self.dt / self.tau                          # discrete version of np.exp(- self.dt / self.tau)
        Q = self.process_noise**2 / self.dt                 # discrete version of (self.tau*self.process_noise**2 / 2) * (1 - np.exp(-2 * self.dt / self.tau))
        R = self.measure_noise**2
        
        total_log_likelihood = 0.0
        for signal_category in list(input_series.keys()):
            B = getattr(self, 'w' + str(signal_category))   # input_weight for this category
            state_series_this_category = state_series[signal_category]
            input_series_this_category = input_series[signal_category]
            
            for trial_id in range(state_series_this_category.shape[0]):
                states = state_series_this_category[trial_id]
                inputs = input_series_this_category[trial_id]
                
                x_pred = states[0]
                P_pred = 1.0                                    # initial variance estimate
                
                for t in range(1, len(states)):
                    input_value = inputs[t-1]
                    x_pred = A * x_pred + B * input_value       # Predict state x
                    P_pred = A**2 * P_pred + Q                  # Predict uncertainty on state P
                    S = P_pred + R                              # Update residual variance S
                    K = P_pred / S                              # Optimal Kalman gain
                    innovation = states[t] - x_pred             # Residual
                    x_pred = x_pred + K * innovation            # Update state x
                    P_pred = (1 - K) * P_pred                   # Update uncertainty on state P
                    total_log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation**2) / S)
        return total_log_likelihood




class StratifiedLinear_kalman(UPP_abstract):
    """ dx/dt = -x + w_{category} * input """

    _param_names = UPP_abstract._param_names + ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6']
    
    def __init__(self, tau: float, process_noise: float, measure_noise: float, w0: float=0, w1: float=0, w2: float=0, w3: float=0, w4: float=0, w5: float=0, w6: float=0):
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        super().__init__(tau, process_noise, measure_noise)

    # ------------------------- MODEL STRUCTURE -------------------------

    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        input_weight = getattr(self, 'w' + str(signal_category))
        return input_weight * input_value

    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return np.zeros_like(state)

    # ------------------------- FITTING TOOLS -------------------------
    # to compute the loglikelihood with UKF
    def loglikelihood_ukf(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]):
        n_jobs = 8
        batch_size = 20

        # ---- Parameters ----
        dt, tau = self.dt, self.tau
        R = self.measure_noise**2
        process_noise = self.process_noise
        Q = self.process_noise**2 / self.dt                 # discrete version of (self.tau*self.process_noise**2 / 2) * (1 - np.exp(-2 * self.dt / self.tau))
        eps = 0 # used to be 1e-12

        # ---- Sigma points ----
        SigmaPoints = MerweScaledSigmaPoints
        sig_alpha, sig_beta, sig_kappa = 0.1, 2.0, 1.0
        UKF_class = UKF
        core_func = self.core

        # ---- Worker for one batch of one category ----
        def process_batch(batch_states, batch_inputs, signal_category):
            total_ll = 0.0
            for states, inputs in zip(batch_states, batch_inputs):
                sigmas = SigmaPoints(n=1, alpha=sig_alpha, beta=sig_beta, kappa=sig_kappa)
                ukf = UKF_class(
                    dim_x=1, dim_z=1,
                    fx=lambda x, dt_local, inp=inputs[0]: core_func(x, inp, signal_category)[0],
                    hx=lambda x: x,
                    dt=dt, points=sigmas
                )
                ukf.x = np.array([states[0]])
                ukf.P = np.eye(1)
                ukf.Q = np.eye(1) * Q
                ukf.R = np.eye(1) * R

                # ---- iterate through time ----
                for t in range(1, len(states)):
                    ukf.fx = lambda x, dt_local, inp=inputs[t - 1]: core_func(x, inp, signal_category)[0]
                    ukf.predict()
                    ukf.update(states[t])

                    r = ukf.y[0]
                    S = ukf.S[0, 0]
                    total_ll += -0.5 * (np.log(2 * np.pi * S + eps) + (r ** 2) / (S + eps))

            return total_ll

        # ---- Build category-wise batches ----
        tasks = []
        for signal_category in input_series.keys():
            states_cat = state_series[signal_category]
            inputs_cat = input_series[signal_category]
            n_trials = states_cat.shape[0]
            batches = [
                (states_cat[i:i + batch_size], inputs_cat[i:i + batch_size], signal_category)
                for i in range(0, n_trials, batch_size)
            ]
            tasks.extend(batches)

        # ---- Parallel execution ----
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_batch)(bs, bi, cat) for bs, bi, cat in tasks
        )

        return float(np.sum(results))


    # to compute the loglikelihood with regular Kalman filtering
    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]) -> float:
        
        A = 1 - self.dt / self.tau                          # discrete version of np.exp(- self.dt / self.tau)
        Q = self.process_noise**2 / self.dt                 # discrete version of (self.tau*self.process_noise**2 / 2) * (1 - np.exp(-2 * self.dt / self.tau))
        R = self.measure_noise**2
        
        total_log_likelihood = 0.0
        for signal_category in list(input_series.keys()):
            B = getattr(self, 'w' + str(signal_category))   # input_weight for this category
            state_series_this_category = state_series[signal_category]
            input_series_this_category = input_series[signal_category]
            
            for trial_id in range(state_series_this_category.shape[0]):
                states = state_series_this_category[trial_id]
                inputs = input_series_this_category[trial_id]
                
                x_pred = states[0]
                P_pred = 1.0                                    # initial variance estimate
                
                for t in range(1, len(states)):
                    input_value = inputs[t-1]
                    x_pred = A * x_pred + B * input_value       # Predict state x
                    P_pred = A**2 * P_pred + Q                  # Predict uncertainty on state P
                    S = P_pred + R                              # Update residual variance S
                    K = P_pred / S                              # Optimal Kalman gain
                    innovation = states[t] - x_pred             # Residual
                    x_pred = x_pred + K * innovation            # Update state x
                    P_pred = (1 - K) * P_pred                   # Update uncertainty on state P
                    total_log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation**2) / S)
        return total_log_likelihood



#####################################################################################################################
#####################################################################################################################
###############################           Non-linear models (fixed gain)               ##############################
#####################################################################################################################
#####################################################################################################################



class NonLinear1(UPP_abstract):
    """ dx/dt = -x + input_weight * input + gain * sigmoid(threshold, sharpness, x)
        Parallelized UKF (single-component) log-likelihood computation.
    """

    _param_names = UPP_abstract._param_names + ['input_weight', 'gain', 'threshold']

    def __init__(self, tau: float, process_noise: float, measure_noise: float,
                 input_weight: float, gain: float, threshold: float, sharpness: float):
        self.input_weight = input_weight
        self.gain = gain
        self.threshold = threshold
        self.sharpness = sharpness
        super().__init__(tau, process_noise, measure_noise)

    # ------------------------- MODEL STRUCTURE -------------------------

    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return self.input_weight * input_value

    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return self.gain / (1 + np.exp(self.sharpness * (self.threshold - state)))

    # ------------------------- FITTING TOOLS -------------------------

    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]) -> float:
        
        n_jobs = 8
        batch_size = 20
        
        # ---- Extract arrays (assume single condition) ----
        state_arr = state_series[list(state_series.keys())[0]]
        input_arr = input_series[list(input_series.keys())[0]]

        # ---- Model parameters ----
        dt, tau = self.dt, self.tau
        Q = self.process_noise**2 / dt
        R = self.measure_noise**2
        eps = 0 # used to be 1e-12

        # ---- Local references ----
        SigmaPoints = MerweScaledSigmaPoints
        UKF_class = UKF
        core_func = self.core

        sig_alpha, sig_beta, sig_kappa = 0.1, 2.0, 1.0

        # ---- Worker for one batch ----
        def process_batch(batch_states, batch_inputs):
            batch_ll = 0.0
            for states, inputs in zip(batch_states, batch_inputs):
                sigmas = SigmaPoints(n=1, alpha=sig_alpha, beta=sig_beta, kappa=sig_kappa)

                def fx(x, dt_local):
                    input_value = inputs[self.t - 1]
                    return core_func(x, input_value, 0)[0]

                def hx(x):
                    return x

                ukf = UKF_class(dim_x=1, dim_z=1, fx=fx, hx=hx, dt=dt, points=sigmas)
                ukf.Q = np.eye(1) * Q
                ukf.R = np.eye(1) * R
                ukf.x = np.array([states[0]])
                ukf.P = np.eye(1)

                logL = 0.0
                for t in range(1, len(states)):
                    self.t = t
                    ukf.predict()
                    ukf.update(states[t])
                    r = ukf.y[0]
                    S = ukf.S[0, 0]
                    logL += -0.5 * (np.log(2 * np.pi * S + eps) + (r**2) / (S + eps))

                batch_ll += logL
            return batch_ll

        # ---- Build batches ----
        n_trials = state_arr.shape[0]
        batches = [
            (state_arr[i:i + batch_size], input_arr[i:i + batch_size])
            for i in range(0, n_trials, batch_size)
        ]

        # ---- Run in parallel ----
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_batch)(bs, bi) for bs, bi in batches
        )

        return float(np.sum(results))




class StratifiedNonLinear1(UPP_abstract):
    """ dx/dt = -x + w_category * input + gain * sigmoid(threshold, sharpness, x)
        Parallelized Unscented Kalman Filter (UKF) likelihood with stratified categories.
    """

    _param_names = (
        UPP_abstract._param_names
        + ['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'gain', 'threshold']
    )

    def __init__(self, tau: float, process_noise: float, measure_noise: float,
                 gain: float, threshold: float, sharpness: float,
                 w0=0, w1=0, w2=0, w3=0, w4=0, w5=0, w6=0):
        self.gain = gain
        self.threshold = threshold
        self.sharpness = sharpness
        self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = w0, w1, w2, w3, w4, w5, w6
        super().__init__(tau, process_noise, measure_noise)

    # ------------------------- MODEL STRUCTURE -------------------------

    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        w = getattr(self, f"w{signal_category}")
        return w * input_value

    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return self.gain / (1 + np.exp(self.sharpness * (self.threshold - state)))

    # ------------------------- FITTING TOOLS -------------------------

    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]):
        """
        Parallelized UKF log-likelihood (no Gaussian mixture).
        """
        n_jobs = 8
        batch_size = 20

        # ---- Parameters ----
        dt, tau = self.dt, self.tau
        R = self.measure_noise**2
        process_noise = self.process_noise
        Q = self.process_noise**2 / self.dt                 # discrete version of (self.tau*self.process_noise**2 / 2) * (1 - np.exp(-2 * self.dt / self.tau))
        eps = 0 # used to be 1e-12

        # ---- Sigma points ----
        SigmaPoints = MerweScaledSigmaPoints
        sig_alpha, sig_beta, sig_kappa = 0.1, 2.0, 1.0
        UKF_class = UKF
        core_func = self.core

        # ---- Worker for one batch of one category ----
        def process_batch(batch_states, batch_inputs, signal_category):
            total_ll = 0.0
            for states, inputs in zip(batch_states, batch_inputs):
                sigmas = SigmaPoints(n=1, alpha=sig_alpha, beta=sig_beta, kappa=sig_kappa)
                ukf = UKF_class(
                    dim_x=1, dim_z=1,
                    fx=lambda x, dt_local, inp=inputs[0]: core_func(x, inp, signal_category)[0],
                    hx=lambda x: x,
                    dt=dt, points=sigmas
                )
                ukf.x = np.array([states[0]])
                ukf.P = np.eye(1)
                ukf.Q = np.eye(1) * Q
                ukf.R = np.eye(1) * R

                # ---- iterate through time ----
                for t in range(1, len(states)):
                    ukf.fx = lambda x, dt_local, inp=inputs[t - 1]: core_func(x, inp, signal_category)[0]
                    ukf.predict()
                    ukf.update(states[t])

                    r = ukf.y[0]
                    S = ukf.S[0, 0]
                    total_ll += -0.5 * (np.log(2 * np.pi * S + eps) + (r ** 2) / (S + eps))

            return total_ll

        # ---- Build category-wise batches ----
        tasks = []
        for signal_category in input_series.keys():
            states_cat = state_series[signal_category]
            inputs_cat = input_series[signal_category]
            n_trials = states_cat.shape[0]
            batches = [
                (states_cat[i:i + batch_size], inputs_cat[i:i + batch_size], signal_category)
                for i in range(0, n_trials, batch_size)
            ]
            tasks.extend(batches)

        # ---- Parallel execution ----
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_batch)(bs, bi, cat) for bs, bi, cat in tasks
        )

        return float(np.sum(results))


        
#####################################################################################################################
#####################################################################################################################
#####################################             Gain Modulation                ####################################
#####################################################################################################################
#####################################################################################################################

class GainModulation(UPP_abstract):
    """ dx/dt = -x + input_weight * input + gain * input * sigmoid(threshold, sharpness, x)"""

    _param_names = UPP_abstract._param_names + ['input_weight', 'gain', 'threshold']
    
    def __init__(self, tau: float, process_noise: float, measure_noise: float, input_weight: float, gain: float, threshold: float, sharpness: float):
        self.input_weight = input_weight
        self.gain = gain
        self.threshold = threshold
        self.sharpness = sharpness
        super().__init__(tau, process_noise, measure_noise)

    # ------------------------- MODEL STRUCTURE -------------------------

    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return self.input_weight * input_value

    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        return self.gain * input_value / (1 + np.exp(self.sharpness * (self.threshold - state)))

    # ------------------------- FITTING TOOLS -------------------------

    # to compute the loglikelihood with Unscented Kalman filtering
    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]) -> float:
        
        n_jobs = 8
        batch_size = 20
        
        # ---- Extract arrays (assume single condition) ----
        state_arr = state_series[list(state_series.keys())[0]]
        input_arr = input_series[list(input_series.keys())[0]]

        # ---- Model parameters ----
        dt, tau = self.dt, self.tau
        Q = self.process_noise**2 / dt
        R = self.measure_noise**2
        eps = 0 # used to be 1e-12

        # ---- Local references ----
        SigmaPoints = MerweScaledSigmaPoints
        UKF_class = UKF
        core_func = self.core

        sig_alpha, sig_beta, sig_kappa = 0.1, 2.0, 1.0

        # ---- Worker for one batch ----
        def process_batch(batch_states, batch_inputs):
            batch_ll = 0.0
            for states, inputs in zip(batch_states, batch_inputs):
                sigmas = SigmaPoints(n=1, alpha=sig_alpha, beta=sig_beta, kappa=sig_kappa)

                def fx(x, dt_local):
                    input_value = inputs[self.t - 1]
                    return core_func(x, input_value, 0)[0]

                def hx(x):
                    return x

                ukf = UKF_class(dim_x=1, dim_z=1, fx=fx, hx=hx, dt=dt, points=sigmas)
                ukf.Q = np.eye(1) * Q
                ukf.R = np.eye(1) * R
                ukf.x = np.array([states[0]])
                ukf.P = np.eye(1)

                logL = 0.0
                for t in range(1, len(states)):
                    self.t = t
                    ukf.predict()
                    ukf.update(states[t])
                    r = ukf.y[0]
                    S = ukf.S[0, 0]
                    logL += -0.5 * (np.log(2 * np.pi * S + eps) + (r**2) / (S + eps))

                batch_ll += logL
            return batch_ll

        # ---- Build batches ----
        n_trials = state_arr.shape[0]
        batches = [
            (state_arr[i:i + batch_size], input_arr[i:i + batch_size])
            for i in range(0, n_trials, batch_size)
        ]

        # ---- Run in parallel ----
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_batch)(bs, bi) for bs, bi in batches
        )

        return float(np.sum(results))




class StratifiedGainModulation(UPP_abstract):
    """ dx/dt = -x + w_category * input + g_category * sigmoid(threshold, sharpness, x) """

    _param_names = (
        UPP_abstract._param_names
        + ['threshold']
        + [f"w{i}" for i in range(7)]
        + [f"g{i}" for i in range(7)]
    )

    def __init__(
        self, tau: float, process_noise: float, measure_noise: float,
        threshold: float, sharpness: float,
        w0=0, w1=0, w2=0, w3=0, w4=0, w5=0, w6=0,
        g0=0, g1=0, g2=0, g3=0, g4=0, g5=0, g6=0
    ):
        self.threshold = threshold
        self.sharpness = sharpness
        self.w0, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = w0, w1, w2, w3, w4, w5, w6
        self.g0, self.g1, self.g2, self.g3, self.g4, self.g5, self.g6 = g0, g1, g2, g3, g4, g5, g6
        super().__init__(tau, process_noise, measure_noise)

    # ------------------------- STANDARDIZED METHODS -------------------------

    def input_function(self, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        input_weight = getattr(self, f"w{signal_category}")
        return input_weight * input_value

    # def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
    #     gain = getattr(self, f"g{signal_category}")
    #     return gain * input_value / (1 + np.exp(self.sharpness * (self.threshold - state)))

    def nonlinearity(self, state: np.ndarray, input_value: np.ndarray, signal_category: int) -> np.ndarray:
        if type(input_value)==np.ndarray: # degueu, à rendre plus lisible
            input_on = input_value[0]
        else:
            input_on = input_value
        gain = getattr(self, f"g{int(signal_category*input_on)}") # here "input_value" is supposed to always be 0 or 1.
        return gain / (1 + np.exp(self.sharpness * (self.threshold - state)))

    # ------------------------- FITTING TOOLS -------------------------

    def loglikelihood(self, state_series: dict[int, np.ndarray], input_series: dict[int, np.ndarray]):
        """
        Parallelized UKF log-likelihood (no Gaussian mixture).
        """
        n_jobs = 8
        batch_size = 20

        # ---- Parameters ----
        dt, tau = self.dt, self.tau
        R = self.measure_noise**2
        process_noise = self.process_noise
        Q = self.process_noise**2 / self.dt                 # discrete version of (self.tau*self.process_noise**2 / 2) * (1 - np.exp(-2 * self.dt / self.tau))
        eps = 0 # used to be 1e-12

        # ---- Sigma points ----
        SigmaPoints = MerweScaledSigmaPoints
        sig_alpha, sig_beta, sig_kappa = 0.1, 2.0, 1.0
        UKF_class = UKF
        core_func = self.core

        # ---- Worker for one batch of one category ----
        def process_batch(batch_states, batch_inputs, signal_category):
            total_ll = 0.0
            for states, inputs in zip(batch_states, batch_inputs):
                sigmas = SigmaPoints(n=1, alpha=sig_alpha, beta=sig_beta, kappa=sig_kappa)
                ukf = UKF_class(
                    dim_x=1, dim_z=1,
                    fx=lambda x, dt_local, inp=inputs[0]: core_func(x, inp, signal_category)[0],
                    hx=lambda x: x,
                    dt=dt, points=sigmas
                )
                ukf.x = np.array([states[0]])
                ukf.P = np.eye(1)
                ukf.Q = np.eye(1) * Q
                ukf.R = np.eye(1) * R

                # ---- iterate through time ----
                for t in range(1, len(states)):
                    ukf.fx = lambda x, dt_local, inp=inputs[t - 1]: core_func(x, inp, signal_category)[0]
                    ukf.predict()
                    ukf.update(states[t])

                    r = ukf.y[0]
                    S = ukf.S[0, 0]
                    total_ll += -0.5 * (np.log(2 * np.pi * S + eps) + (r ** 2) / (S + eps))

            return total_ll

        # ---- Build category-wise batches ----
        tasks = []
        for signal_category in input_series.keys():
            states_cat = state_series[signal_category]
            inputs_cat = input_series[signal_category]
            n_trials = states_cat.shape[0]
            batches = [
                (states_cat[i:i + batch_size], inputs_cat[i:i + batch_size], signal_category)
                for i in range(0, n_trials, batch_size)
            ]
            tasks.extend(batches)

        # ---- Parallel execution ----
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_batch)(bs, bi, cat) for bs, bi, cat in tasks
        )

        return float(np.sum(results))
