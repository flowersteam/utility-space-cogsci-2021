#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
from scipy import optimize
import pandas as pd
import argparse
from itertools import combinations
from loc_utils import get_sdf, ewmv, objective, objective2


def main(data_path, sid, nfits, save_path, test):
    # Set up model comparison (get paramter combinations)
    # np.random.seed(1)
    var_list = ['mean', 'var', 'diffOFmean', 'diffOFvar']
    if test == 1:
        var_list = ['mean']
    elif test == 2:
        var_list = ['mean', 'diffOFvar']
    subsets = []
    for nb_vars in range(1, len(var_list)+1):
        for subset in combinations(var_list, nb_vars):
            subsets.append(' + '.join(subset))
    if test == 3:
        subsets = var_list
    # Fit a model-form subset for each subject one-by-one
    df = get_sdf(data_path, sid, verbose=False).filter(items=['group','trial','activity','correct'])
    group = df.loc[:, 'group'].values[0]
    first = True
    for i_subset, form in enumerate(subsets):
        softmax_vars = {'tau': [0, 1000]}
        utility_vars = dict(
            zip(form.split(' + '), [[-1, 1] for _ in form.split(' + ')])
        )
        feature_vars = {'weight1': [0, 1]}
        if 'OF' in form:
            feature_vars['weight2'] = [0, 1]

        bounds = sum([list(d.values()) for d in [softmax_vars, utility_vars, feature_vars]], [])
        comps = list(utility_vars.keys())

        # Peform n fits
        all_param_keys = ['tau', 'weight1', 'weight2'] + var_list
        param_results = dict(zip(all_param_keys, [[] for i in all_param_keys]))
        results = {'sid': [], 'form': [], 'ind': [], 'aic': []}
        results.update(param_results)
        for i_fit in range(nfits):
            init_guess = [np.random.uniform(b[0], b[1]) for b in bounds]
            params, negloglik, _ = optimize.fmin_l_bfgs_b(
                func=objective2,
                x0=init_guess,
                args=[df, comps],
                bounds=bounds,
                approx_grad=True,
                disp=False
            )
            score = round(negloglik, 4)
            results['sid'].append(sid)
            results['form'].append(form)
            results['ind'].append(i_fit)
            results['aic'].append(2*score + 2*len(bounds))
            param_keys = list(dict(**softmax_vars, **utility_vars, **feature_vars).keys())
            for k in all_param_keys:
                results[k].append(params[param_keys.index(k)] if k in param_keys else np.nan)
        sid_index = f'{sid}'.zfill(3)
        pd.DataFrame(results).to_csv(f'{save_path}/fit_s{sid_index}.csv', mode='w' if first else 'a', header=first, index=False)
        first = False

    print(f'SID {sid} done')

parser = argparse.ArgumentParser()
parser.add_argument('--sid', help='subject id', type=int)
parser.add_argument('--nfits', help='numer of fits per siubject', type=int)
parser.add_argument('--save_to', help='relative path to save data', type=str)
parser.add_argument('--test', help='whether to run a minimal example', type=int)
args = parser.parse_args()

np.seterr(divide='ignore')
main(
    data_path = 'clean_data.csv',
    sid = args.sid,
    nfits = args.nfits,
    save_path = args.save_to,
    test = args.test if args.test else False
)
