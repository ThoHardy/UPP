import numpy as np
from . import model


def clever_fit_gainmodul(pinit: dict[str, float] | None, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, tau_bias: float=0.0, t1: int=75, t2: int=100, t3: int=150) -> model.StratifiedGainModulation:

    n_categories = len(list(state_train.keys()))
    
    if pinit == None:
        pinit = {'tau': 10, 'process_noise': 0.5, 'measure_noise': 0.5}
        
    # pre-fit with linear
    linear_gm = model.Linear(tau=10, process_noise=0.2, measure_noise=0.2, input_weight=0)
    linear_gm.fit(state_series = {0: state_train[0]}, 
               input_series = {0: input_train[0]}, 
               init_params = [pinit['tau'], pinit['process_noise'], pinit['measure_noise'], 0], 
               bounds = [(1,25), (0.01, 1), (0.01, 1), (0,1)],
               fixed_params = ['input_weight'],
               feedback = False)

    # fit gainmodul (different initializations for the threhold)
    whole_GMmodels, ll_GMmodels = [], []
    for th in np.linspace(0, 1.5, 10):
        gainmodul = model.StratifiedGainModulation(tau=linear_gm.tau + tau_bias, process_noise=linear_gm.process_noise, measure_noise=linear_gm.measure_noise, threshold=th, sharpness=5, 
                                                   **{f"w{cat}": 0.1 for cat in range(1, n_categories)}, 
                                                   **{f"g{cat}": 0.1 for cat in range(1, n_categories)})
        for _ in range(n_loops):
            # Focus on the (w, g) couples for all categories except 0
            for cat in range(1, n_categories):
                gainmodul_one_cat = model.GainModulation(tau=gainmodul.tau, process_noise=gainmodul.process_noise, measure_noise=gainmodul.measure_noise, 
                                                             input_weight=getattr(gainmodul, f'w{cat}'), 
                                                             gain=getattr(gainmodul, f'g{cat}'), 
                                                             threshold=gainmodul.threshold, sharpness=5)
                gainmodul_one_cat.fit(state_series = {0: state_train[cat][:,t1-10:t2]}, 
                                      input_series = {0: input_train[cat][:,t1-10:t2]}, 
                                      init_params = [getattr(gainmodul_one_cat, pname) for pname in gainmodul_one_cat._param_names], 
                                      bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 2)], 
                                      fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold'],
                                      feedback = False)
                gainmodul.set_params({f'w{cat}': gainmodul_one_cat.input_weight, f'g{cat}': gainmodul_one_cat.gain})
            # Focus on the g0
            gainmodul.fit(state_series = state_train, # {cat: state_train[cat][:,t2:t3] for cat in range(n_categories)}, 
                                 input_series = input_train, # {cat: input_train[cat][:,t2:t3] for cat in range(n_categories)}, 
                                 init_params = [getattr(gainmodul, pname) for pname in gainmodul._param_names], 
                                 bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + 14*[(0, 1)], 
                                 fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(1,7)],
                                 feedback = False)
            # Focus on the threshold
            gainmodul.fit(state_series = state_train, # {cat: state_train[cat][:,t1:t3] for cat in range(n_categories)}, 
                                 input_series = input_train, # {cat: input_train[cat][:,t1:t3] for cat in range(n_categories)}, 
                                 init_params = [getattr(gainmodul, pname) for pname in gainmodul._param_names], 
                                 bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + 14*[(0, 1)], 
                                 fixed_params = ['tau', 'process_noise', 'measure_noise'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(7)],
                                 feedback = False)
            # Focus on tau
            gainmodul.fit(state_series={0: state_train[0]},
                                 input_series={0: input_train[0]},
                                 init_params = [getattr(gainmodul, pname) for pname in gainmodul._param_names], 
                                 bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + 14*[(0, 1)], 
                                 fixed_params = ['process_noise', 'measure_noise', 'threshold'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(7)],
                                 feedback = False)
        whole_GMmodels.append(gainmodul)
        ll_GMmodels.append(gainmodul.loglikelihood(state_train, input_train))
    gainmodul_fitted = whole_GMmodels[np.argmax(ll_GMmodels)]

    return gainmodul_fitted




def fit_nonlinear_from_gainmodul(gainmodul: model.StratifiedGainModulation, pinit: dict[str, float] | None, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, tau_bias: float=0.0, t1: int=75, t2: int=100, t3: int=150) -> model.StratifiedNonLinear1:

    n_categories = len(list(state_train.keys()))
    to_keep_fixed = ['w6'] if n_categories < 7 else []
    
    mean_gain = np.mean([getattr(gainmodul, f"g{i}") for i in range(n_categories)])
    nonlinear = upp.model.StratifiedNonLinear1(tau=gainmodul.tau, process_noise=gainmodul.process_noise, measure_noise=gainmodul.measure_noise, gain=mean_gain, threshold=gainmodul.threshold, sharpness=5, **{f"w{i}": getattr(gainmodul, f"w{i}") for i in range(7)})

    for _ in range(n_loops):
        # Focus on the input_weights and the fixed gain
        nonlinear.fit(state_series=state_train,
                    input_series=input_train,
                    init_params=[nonlinear.tau, nonlinear.process_noise, nonlinear.measure_noise] + [getattr(nonlinear, f'w{i}')/2 for i in range(7)] + [mean_gain, nonlinear.threshold],
                    bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7 + [(0, 0.5), (0, 2)],
                    fixed_params=['tau', 'process_noise', 'measure_noise', 'w0'] + to_keep_fixed,
                    feedback=False)
        
        # Focus on the threshold
        nonlinear.fit(state_series=state_train,
                    input_series=input_train,
                    init_params=[nonlinear.tau, nonlinear.process_noise, nonlinear.measure_noise] + [getattr(nonlinear, f'w{i}')/2 for i in range(7)] + [mean_gain, nonlinear.threshold],
                    bounds=[(1, 25), (0.01, 1), (0.01, 1)] + [(0, 1)]*7 + [(0, 0.5), (0, 2)],
                    fixed_params=['tau', 'process_noise', 'measure_noise', 'gain'] + [f'w{i}' for i in range(7)],
                    feedback=False)

    return nonlinear






    