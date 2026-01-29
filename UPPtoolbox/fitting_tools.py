import numpy as np
from . import model


def clever_fit_linear(state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, input_start_index: int=75, input_stop_index: int=100) -> model.Linear:

    n_categories = len(list(state_train.keys()))

    # fit the OU process on resting-state dynamics
    linear_resting_state = model.Linear(tau=10, process_noise=0.1, measure_noise=0.1, input_weight=0)
    linear_resting_state.fit(
        state_series={0: state_train[0]},
        input_series={0: input_train[0]},
        init_params=[10, 0.1, 0.1, 0],
        bounds=[(1, 25), (0.01, 1), (0.01, 1), (0, 1)],
        fixed_params=['input_weight'])
    

    # isolate segments where stimulation is supposed constant
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}

    # fit linear (different initializations for the ws)
    linear = model.StratifiedLinear_kalman(tau=linear_resting_state.tau, process_noise=linear_resting_state.process_noise, measure_noise=linear_resting_state.measure_noise)
    for cat in range(1, n_categories):
        ws, lls = [], []
        for w_init in [0, 0.1]:
            linear_one_cat = model.Linear(tau=linear.tau, process_noise=linear.process_noise, measure_noise=linear.measure_noise, input_weight=w_init)
            linear_one_cat.fit(
                state_series={0: state_train_constant_stim[cat]},
                input_series={0: input_train_constant_stim[cat]},
                init_params=[linear.tau, linear.process_noise, linear.measure_noise, w_init],
                bounds=[(1, 25), (0.01, 1), (0.01, 1), (0, 1)],
                fixed_params=['tau', 'measure_noise', 'process_noise'])
            ws.append(linear_one_cat.input_weight)
            lls.append(linear_one_cat.loglikelihood(state_train_constant_stim, input_train_constant_stim))
        linear.set_params({f'w{cat}': ws[np.argmax(lls)]})
    return linear




def clever_fit_gainmodul(linear_prefitted: model.StratifiedLinear_kalman, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, input_start_index: int=75, input_stop_index: int=100, feedback=False) -> model.StratifiedGainModulation:

    n_categories = len(list(state_train.keys()))

    # fit the OU process on resting-state dynamics
    linear = model.Linear(tau=10, process_noise=0.1, measure_noise=0.1, input_weight=0)
    linear.fit(
        state_series={0: state_train[0]},
        input_series={0: input_train[0]},
        init_params=[10, 0.1, 0.1, 0],
        bounds=[(1, 25), (0.01, 1), (0.01, 1), (0, 1)],
        fixed_params=['input_weight'])

    # isolate segments where stimulation is supposed constant
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}

    # fit gainmodul (different initializations for the threhold)
    whole_GMmodels, ll_GMmodels = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(2):
            init_gain_multiplier = init_gain_index/2       # make gain_init range from 0 to w5_linear/2
            w_multiplier = 1 - init_gain_index/2           # make w5_init range from w5_linear to w5_linear/2
            gainmodul = model.StratifiedGainModulation(tau=linear.tau, process_noise=linear.process_noise, measure_noise=linear.measure_noise, threshold=th, sharpness=5, 
                                                       **{f"w{cat}": getattr(linear_prefitted, f"w{cat}")*w_multiplier for cat in range(1, n_categories)}, 
                                                       **{f"g{cat}": getattr(linear_prefitted, f"w{cat}")*init_gain_multiplier for cat in range(1, n_categories)})
            for _ in range(n_loops):
                # Focus on the (w, g) couples for all categories except 0
                for cat in range(1, n_categories):
                    gainmodul_one_cat = model.GainModulation(tau=gainmodul.tau, process_noise=gainmodul.process_noise, measure_noise=gainmodul.measure_noise, 
                                                                 input_weight=getattr(gainmodul, f'w{cat}'), 
                                                                 gain=getattr(gainmodul, f'g{cat}'), 
                                                                 threshold=gainmodul.threshold, sharpness=5)
                    gainmodul_one_cat.fit(state_series = {0: state_train_constant_stim[cat]}, 
                                          input_series = {0: input_train_constant_stim[cat]}, 
                                          init_params = [getattr(gainmodul_one_cat, pname) for pname in gainmodul_one_cat._param_names], 
                                          bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 2)], 
                                          fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold'],
                                          feedback = False)
                    gainmodul.set_params({f'w{cat}': gainmodul_one_cat.input_weight, f'g{cat}': gainmodul_one_cat.gain})
                # Focus on the threshold
                gainmodul.fit(state_series = state_train_constant_stim, 
                                     input_series = input_train_constant_stim, 
                                     init_params = [getattr(gainmodul, pname) for pname in gainmodul._param_names], 
                                     bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + 14*[(0, 1)], 
                                     fixed_params = ['tau', 'process_noise', 'measure_noise'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(7)],
                                     feedback = False)
            whole_GMmodels.append(gainmodul)
            ll_GMmodels.append(gainmodul.loglikelihood(state_train_constant_stim, input_train_constant_stim))
    gainmodul_fitted = whole_GMmodels[np.argmax(ll_GMmodels)]
    if feedback:
        print(f'Best model selected among: {[model.get_params() for model in whole_GMmodels]}, \nwith respective log-likelihoods {ll_GMmodels}')
    return gainmodul_fitted



def clever_fit_nonlinear1(linear_prefitted: model.StratifiedLinear_kalman, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, input_start_index: int=75, input_stop_index: int=100, feedback=False) -> model.StratifiedNonLinear1:

    n_categories = len(list(state_train.keys()))

    # fit the OU process on resting-state dynamics
    linear = model.Linear(tau=10, process_noise=0.1, measure_noise=0.1, input_weight=0)
    linear.fit(
        state_series={0: state_train[0]},
        input_series={0: input_train[0]},
        init_params=[10, 0.1, 0.1, 0],
        bounds=[(1, 25), (0.01, 1), (0.01, 1), (0, 1)],
        fixed_params=['input_weight'])

    # isolate segments where stimulation is supposed constant
    state_train_constant_stim = {cat: state_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    state_train_constant_stim[0] = state_train[0]
    input_train_constant_stim = {cat: input_train[cat][:,input_start_index:input_stop_index] for cat in range(1, n_categories)}
    input_train_constant_stim[0] = input_train[0]

    # fit gainmodul (different initializations for the threhold)
    all_models, ll_models = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(2):
            init_gain_multiplier = init_gain_index/2       # make gain_init range from 0 to w5_linear/2
            w_multiplier = 1 - init_gain_index/2           # make w5_init range from w5_linear to w5_linear/2
            nonlinear = model.StratifiedNonLinear1(tau=linear.tau, process_noise=linear.process_noise, measure_noise=linear.measure_noise, threshold=th, gain=getattr(linear_prefitted, f"w{n_categories-1}")*init_gain_multiplier, sharpness=5, 
                                                       **{f"w{cat}": getattr(linear_prefitted, f"w{cat}")*w_multiplier for cat in range(1, n_categories)})
            for _ in range(n_loops):
                # Focus on the gain and the threshold
                nonlinear.fit(state_series = state_train_constant_stim,
                             input_series = input_train_constant_stim,
                             init_params = [getattr(nonlinear, pname) for pname in nonlinear._param_names], 
                             bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + [(0, 1)]*7 + [(0, 0.5), (0, 2)], 
                             fixed_params = ['tau', 'process_noise', 'measure_noise'] + [f'w{cat}' for cat in range(7)],
                             feedback = False)
                # Focus on the w for all categories except 0
                for cat in range(1, n_categories):
                    gainmodul_one_cat = model.GainModulation(tau=nonlinear.tau, process_noise=nonlinear.process_noise, measure_noise=nonlinear.measure_noise, 
                                                             input_weight=getattr(nonlinear, f'w{cat}'), 
                                                             gain=nonlinear.gain, 
                                                             threshold=nonlinear.threshold, sharpness=5)
                    gainmodul_one_cat.fit(state_series = {0: state_train_constant_stim[cat]}, 
                                          input_series = {0: input_train_constant_stim[cat]}, 
                                          init_params = [getattr(gainmodul_one_cat, pname) for pname in gainmodul_one_cat._param_names], 
                                          bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 2)], 
                                          fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold', 'gain'],
                                          feedback = False)
                    nonlinear.set_params({f'w{cat}': gainmodul_one_cat.input_weight})
            all_models.append(nonlinear)
            ll_models.append(nonlinear.loglikelihood(state_train_constant_stim, input_train_constant_stim))
    nonlinear_fitted = all_models[np.argmax(ll_models)]
    if feedback:
        print(f'Best model selected among: {[model.get_params() for model in whole_GMmodels]}, \nwith respective log-likelihoods {ll_GMmodels}')
    return nonlinear_fitted
    


def fit_gainmodul_from_linear(linear_gm: model.StratifiedLinear_kalman, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, tau_bias: float=0.0, l2_tau: float=0, input_start_index: int=75, input_stop_index: int=100, fit_g0=False) -> model.StratifiedGainModulation:

    n_categories = len(list(state_train.keys()))

    # fit gainmodul (different initializations for the threhold)
    whole_GMmodels, ll_GMmodels = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(3):
            init_gain_multiplier = init_gain_index/3       # make gain_init range from 0 to w5_linear
            w_multiplier = 1 - init_gain_index/3           # make w5_init range from w5_linear to 0
            gainmodul = model.StratifiedGainModulation(tau=linear_gm.tau + tau_bias, process_noise=linear_gm.process_noise, measure_noise=linear_gm.measure_noise, threshold=th, sharpness=5, 
                                                       **{f"w{cat}": getattr(linear_gm, f"w{cat}")*w_multiplier for cat in range(1, n_categories)}, 
                                                       **{f"g{cat}": getattr(linear_gm, f"w{cat}")*init_gain_multiplier for cat in range(1, n_categories)})
            for _ in range(n_loops):
                # Focus on the (w, g) couples for all categories except 0
                for cat in range(1, n_categories):
                    gainmodul_one_cat = model.GainModulation(tau=gainmodul.tau, process_noise=gainmodul.process_noise, measure_noise=gainmodul.measure_noise, 
                                                                 input_weight=getattr(gainmodul, f'w{cat}'), 
                                                                 gain=getattr(gainmodul, f'g{cat}'), 
                                                                 threshold=gainmodul.threshold, sharpness=5)
                    gainmodul_one_cat.fit(state_series = {0: state_train[cat][:,max(0, input_start_index-10):input_stop_index]}, 
                                          input_series = {0: input_train[cat][:,max(0, input_start_index-10):input_stop_index]}, 
                                          init_params = [getattr(gainmodul_one_cat, pname) for pname in gainmodul_one_cat._param_names], 
                                          bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 2)], 
                                          fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold'],
                                          feedback = False)
                    gainmodul.set_params({f'w{cat}': gainmodul_one_cat.input_weight, f'g{cat}': gainmodul_one_cat.gain})
                # Focus on the g0 (not by default)
                if fit_g0:
                    gainmodul.fit(state_series = {0: state_train[0]},
                                         input_series = {0: input_train[0]},
                                         init_params = [getattr(gainmodul, pname) for pname in gainmodul._param_names], 
                                         bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + 14*[(0, 1)], 
                                         fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(1,7)],
                                         feedback = False)
                # Focus on the threshold
                gainmodul.fit(state_series = {cat: state_train[cat][:,max(0, input_start_index-10):input_stop_index] for cat in range(n_categories)}, 
                                     input_series = {cat: input_train[cat][:,max(0, input_start_index-10):input_stop_index] for cat in range(n_categories)}, 
                                     init_params = [getattr(gainmodul, pname) for pname in gainmodul._param_names], 
                                     bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + 14*[(0, 1)], 
                                     fixed_params = ['tau', 'process_noise', 'measure_noise'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(7)],
                                     feedback = False)
                # Focus on tau
                gainmodul.fit(state_series = state_train,
                                     input_series = input_train,
                                     init_params = [getattr(gainmodul, pname) for pname in gainmodul._param_names], 
                                     bounds = [(1,linear_gm.tau), (0.01, 1), (0.01, 1), (0, 2)] + 14*[(0, 1)], 
                                     fixed_params = ['process_noise', 'measure_noise', 'threshold'] + [f'w{cat}' for cat in range(7)] + [f'g{cat}' for cat in range(7)],
                                     feedback = False,
                                     l2_tau = l2_tau)
            whole_GMmodels.append(gainmodul)
            ll_GMmodels.append(gainmodul.loglikelihood(state_train, input_train))
    gainmodul_fitted = whole_GMmodels[np.argmax(ll_GMmodels)]

    return gainmodul_fitted




def fit_nonlinear1_from_linear(linear_gm: model.StratifiedLinear_kalman, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, tau_bias: float=0.0, l2_tau: float=0, input_start_index: int=75, input_stop_index: int=100) -> model.StratifiedNonLinear1:

    n_categories = len(list(state_train.keys()))

    # fit gainmodul (different initializations for the threhold)
    all_models, ll_models = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(3):
            init_gain_multiplier = init_gain_index/3       # make gain_init range from 0 to w5_linear
            w_multiplier = 1 - init_gain_index/3           # make w5_init range from w5_linear to 0
            nonlinear = model.StratifiedNonLinear1(tau=linear_gm.tau + tau_bias, process_noise=linear_gm.process_noise, measure_noise=linear_gm.measure_noise, threshold=th, gain=getattr(linear_gm, f"w{n_categories-1}")*init_gain_multiplier, sharpness=5, 
                                                       **{f"w{cat}": getattr(linear_gm, f"w{cat}")*w_multiplier for cat in range(1, n_categories)})
            for _ in range(n_loops):
                # Focus on the gain
                nonlinear.fit(state_series = {cat: state_train[cat][:,:input_stop_index] for cat in range(n_categories)},
                                     input_series = {cat: input_train[cat][:,:input_stop_index] for cat in range(n_categories)},
                                     init_params = [getattr(nonlinear, pname) for pname in nonlinear._param_names], 
                                     bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + [(0, 1)]*7 + [(0, 0.5), (0, 2)], 
                                     fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold'] + [f'w{cat}' for cat in range(7)],
                                     feedback = False)
                # Focus on the input weights
                for cat in range(1, n_categories):
                    gainmodul_one_cat = model.GainModulation(tau=nonlinear.tau, process_noise=nonlinear.process_noise, measure_noise=nonlinear.measure_noise, 
                                                             input_weight=getattr(nonlinear, f'w{cat}'), 
                                                             gain=nonlinear.gain, 
                                                             threshold=nonlinear.threshold, sharpness=5)
                    gainmodul_one_cat.fit(state_series = {0: state_train[cat][:,max(0, input_start_index-10):input_stop_index]}, 
                                          input_series = {0: input_train[cat][:,max(0, input_start_index-10):input_stop_index]}, 
                                          init_params = [getattr(gainmodul_one_cat, pname) for pname in gainmodul_one_cat._param_names], 
                                          bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 2)], 
                                          fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold', 'gain'],
                                          feedback = False)
                    nonlinear.set_params({f'w{cat}': gainmodul_one_cat.input_weight})
                # Focus on the threshold
                nonlinear.fit(state_series = state_train,
                                     input_series = input_train,
                                     init_params = [getattr(nonlinear, pname) for pname in nonlinear._param_names], 
                                     bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + [(0, 1)]*7 + [(0, 0.5), (0, 2)], 
                                     fixed_params = ['tau', 'process_noise', 'measure_noise', 'gain'] + [f'w{cat}' for cat in range(7)],
                                     feedback = False)
                # Focus on tau
                nonlinear.fit(state_series = state_train,
                                     input_series = input_train,
                                     init_params = [getattr(nonlinear, pname) for pname in nonlinear._param_names], 
                                     bounds = [(1,linear_gm.tau), (0.01, 1), (0.01, 1), (0, 2)] + [(0, 1)]*7 + [(0, 0.5), (0, 2)], 
                                     fixed_params = ['process_noise', 'measure_noise', 'threshold', 'gain'] + [f'w{cat}' for cat in range(7)],
                                     feedback = False,
                                     l2_tau = l2_tau)
            all_models.append(nonlinear)
            ll_models.append(nonlinear.loglikelihood(state_train, input_train))
    nonlinear_fitted = all_models[np.argmax(ll_models)]

    return nonlinear_fitted




def fit_nonlinear2_from_linear(linear_gm: model.StratifiedLinear_kalman, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, tau_bias: float=0.0, l2_tau: float=0, input_start_index: int=75, input_stop_index: int=100) -> model.StratifiedNonLinear2:

    n_categories = len(list(state_train.keys()))

    # fit gainmodul (different initializations for the threhold)
    all_models, ll_models = [], []
    for th in np.linspace(0, 2, 5):
        for init_gain_index in range(3):
            init_gain_multiplier = init_gain_index/3       # make gain_init range from 0 to w5_linear
            w_multiplier = 1 - init_gain_index/3           # make w5_init range from w5_linear to 0
            nonlinear = model.StratifiedNonLinear2(tau=linear_gm.tau + tau_bias, process_noise=linear_gm.process_noise, measure_noise=linear_gm.measure_noise, threshold=th, a=0, b=getattr(linear_gm, f"w{n_categories-1}")*init_gain_multiplier, sharpness=5, 
                                                       **{f"w{cat}": getattr(linear_gm, f"w{cat}")*w_multiplier for cat in range(1, n_categories)})
            for _ in range(n_loops):
                # Focus on a & b
                nonlinear.fit(state_series = {cat: state_train[cat][:,:input_stop_index] for cat in range(n_categories)},
                                     input_series = {cat: input_train[cat][:,:input_stop_index] for cat in range(n_categories)},
                                     init_params = [getattr(nonlinear, pname) for pname in nonlinear._param_names], 
                                     bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + [(0, 1)]*7 + [(0, 1), (0, 0.5), (0, 2)], 
                                     fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold'] + [f'w{cat}' for cat in range(7)],
                                     feedback = False)
                # Focus on the input weights
                for cat in range(1, n_categories):
                    nonlinear_one_cat = model.NonLinear2(tau=nonlinear.tau, process_noise=nonlinear.process_noise, measure_noise=nonlinear.measure_noise, 
                                                         input_weight=getattr(nonlinear, f'w{cat}'), 
                                                         a=nonlinear.a, b=nonlinear.b, 
                                                         threshold=nonlinear.threshold, sharpness=5)
                    nonlinear_one_cat.fit(state_series = {0: state_train[cat][:,input_start_index-10:input_stop_index]}, 
                                          input_series = {0: input_train[cat][:,input_start_index-10:input_stop_index]}, 
                                          init_params = [getattr(nonlinear_one_cat, pname) for pname in nonlinear_one_cat._param_names], 
                                          bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 1), (0, 1), (0, 1), (0, 2)], 
                                          fixed_params = ['tau', 'process_noise', 'measure_noise', 'threshold', 'a', 'b'],
                                          feedback = False)
                    nonlinear.set_params({f'w{cat}': nonlinear_one_cat.input_weight})
                # Focus on the threshold
                nonlinear.fit(state_series = state_train,
                                     input_series = input_train,
                                     init_params = [getattr(nonlinear, pname) for pname in nonlinear._param_names], 
                                     bounds = [(1,25), (0.01, 1), (0.01, 1), (0, 2)] + [(0, 1)]*7 + [(0, 1), (0, 0.5), (0, 2)], 
                                     fixed_params = ['tau', 'process_noise', 'measure_noise', 'a', 'b'] + [f'w{cat}' for cat in range(7)],
                                     feedback = False)
                # Focus on tau
                nonlinear.fit(state_series={0: state_train[0]},
                                     input_series={0: input_train[0]},
                                     init_params = [getattr(nonlinear, pname) for pname in nonlinear._param_names], 
                                     bounds = [(1,linear_gm.tau), (0.01, 1), (0.01, 1), (0, 2)] + [(0, 1)]*7 + [(0, 1), (0, 0.5), (0, 2)], 
                                     fixed_params = ['process_noise', 'measure_noise', 'threshold', 'a', 'b'] + [f'w{cat}' for cat in range(7)],
                                     feedback = False,
                                     l2_tau = l2_tau)
            all_models.append(nonlinear)
            ll_models.append(nonlinear.loglikelihood(state_train, input_train))
    nonlinear_fitted = all_models[np.argmax(ll_models)]

    return nonlinear_fitted

    


def fit_nonlinear_from_gainmodul(gainmodul: model.StratifiedGainModulation, pinit: dict[str, float] | None, state_train: dict[int, np.ndarray], input_train: dict[int, np.ndarray], n_loops: int=2, tau_bias: float=0.0, t1: int=75, t2: int=100, t3: int=150) -> model.StratifiedNonLinear1:

    n_categories = len(list(state_train.keys()))
    to_keep_fixed = ['w6'] if n_categories < 7 else []
    
    mean_gain = np.mean([getattr(gainmodul, f"g{i}") for i in range(n_categories)])
    nonlinear = model.StratifiedNonLinear1(tau=gainmodul.tau, process_noise=gainmodul.process_noise, measure_noise=gainmodul.measure_noise, gain=mean_gain, threshold=gainmodul.threshold, sharpness=5, **{f"w{i}": getattr(gainmodul, f"w{i}") for i in range(7)})

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






    