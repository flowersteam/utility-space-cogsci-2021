import numpy as np
import pandas as pd


def get_sdf(data_path, sid, verbose=True):
    df = pd.read_csv(data_path, index_col='sid')
    if sid < 0:
        rsid = np.random.choice(df.index.tolist())
        sdf = df.loc[rsid, :]
    else:
        rsid = sid
        sdf = df.loc[sid, :]
    if verbose:
        print('Sampling sid: {}'.format(rsid))
    return sdf


def softmax_2d(x, shift=True):
    """Shifting prevents overflow"""
    z = x - np.max(x, axis=1)[:, np.newaxis] if shift else x
    a = np.exp(z)
    return a / np.sum(a, axis=1)[:, np.newaxis]


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def ewmv(X, step, init_mean=.5, init_var=.25, return_keys='all'):
    out_dict = {}
    out_dict['diff'] = [X[0] - init_mean]
    out_dict['incr'] = [step*(out_dict['diff'][0])]
    out_dict['mean'] = [init_mean + out_dict['incr'][0]]
    out_dict['var'] = [(1-step) * (init_var + out_dict['diff'][0] * out_dict['incr'][0])]

    for i, x in enumerate(X[1:]):
        diff = x - out_dict['mean'][i]
        incr = step * diff
        mean = out_dict['mean'][i] + incr
        vari = (1-step) * (out_dict['var'][i] + diff * incr)
        out_dict['mean'].append(mean)
        out_dict['diff'].append(diff)
        out_dict['incr'].append(incr)
        out_dict['var'].append(vari)

    for k, v in out_dict.items():
        out_dict[k] = np.array(v)

    if return_keys == 'all':
        return out_dict
    else:
        return dict((k, out_dict[k]) for k in return_keys if k in out_dict)


def objective(params, *args):
    df, comps = args
    if 'diffOF' in ''.join(comps):
        step = params[-2]
        scale = params[-1]
    else:
        step = params[-1]
        scale = 0
    choices = pd.get_dummies(df.set_index('trial').activity).reset_index()
    if df.group.values[0]<=1:
	       choices = choices.loc[choices.trial.gt(60), :].drop(columns='trial').values.astype(bool)
    else:
	       choices = choices.drop(columns='trial').values.astype(bool)

    for i, a in enumerate(sorted(df.activity.unique()), 1):
        X = df.loc[df.activity.eq(a), 'correct'].values
        features = ewmv(X, step, init_mean=.5, init_var=.25, return_keys='all')
        features_ = ewmv(X, step*scale, init_mean=.5, init_var=.25, return_keys='all')
        for k, v in features.items():
            df[f'{k}{i}'] = np.nan
            df.loc[df.activity.eq(a), f'{k}{i}'] = v

            df[f'diffOF{k}{i}'] = np.nan
            df.loc[df.activity.eq(a), f'diffOF{k}{i}'] = np.abs(v - features_[k])

    df = df.fillna(method='ffill', axis=0).dropna()
    if df.group.values[0]<=1:
	       df = df.loc[df.trial.gt(60), :].drop(columns=['trial', 'correct'])
    else:
	       df = df.drop(columns=['trial', 'correct'])
    ucomps = []
    for i, comp in enumerate(comps, 1):
        ucomps.append(
            df.loc[:, [f'{comp}{a}' for a in '1234']]*params[i]
        )

    u = sum([ucomp.values for ucomp in ucomps]) * params[0]
    try:
        p_choices = softmax_2d(u)[choices]
        loglik_trials = np.log(p_choices.astype(np.float64)+np.nextafter(np.float64(0), np.float64(1)))
        loglik_sum = np.sum(loglik_trials, axis=0)
        return -loglik_sum
    except IndexError:
	    return np.nan


def objective2(params, *args):
    priors = {
        'mean': 0.5,
        'var': 0.25,
        'meanOFmean': 0.5,
        'varOFmean': 0.0,
        'meanOFvar': .25,
        'varOFvar': 0.0,
        'meanOFdiff': 0.0,
        'varOFdiff': .25
    }
    df, comps = args
    if 'OF' in ''.join(comps):
        weight1 = params[-2]
        weight2 = params[-1]
    else:
        weight1 = params[-1]
        weight2 = None
    choices = pd.get_dummies(df.set_index('trial').activity).reset_index()
    if df.group.values[0] <= 1:
	       choices = choices.loc[choices.trial.gt(60), :].drop(columns='trial').values.astype(bool)
    else:
	       choices = choices.drop(columns='trial').values.astype(bool)

    for i, a in enumerate(sorted(df.activity.unique()), 1):
        X = df.loc[df.activity.eq(a), 'correct'].values
        features1 = ewmv(X, weight1, init_mean=priors['mean'], init_var=priors['var'], return_keys=['mean', 'var', 'diff'])
        for k, v in features1.items():
            df[f'{k}{i}'] = np.nan
            df.loc[df.activity.eq(a), f'{k}{i}'] = v
            if weight2 is not None:
                features2 = ewmv(v, weight2, init_mean=priors[f'meanOF{k}'], init_var=priors[f'varOF{k}'], return_keys=['mean', 'var', 'diff'])
                for k2, v2 in features2.items():
                    df[f'{k2}OF{k}{i}'] = np.nan
                    df.loc[df.activity.eq(a), f'{k2}OF{k}{i}'] = v2

    df = df.fillna(method='ffill', axis=0).dropna()
    if df.group.values[0] <= 1:
	       df = df.loc[df.trial.gt(60), :].drop(columns=['trial', 'correct'])
    else:
	       df = df.drop(columns=['trial', 'correct'])
    ucomps = []
    for i, comp in enumerate(comps, 1):
        cols = []
        for a in '1234':
            if f'{comp}{a}' in df.columns.tolist():
                cols.append(f'{comp}{a}')
        ucomps.append(
            df.loc[:, cols]*params[i]
        )

    u = sum([ucomp.values for ucomp in ucomps]) * params[0]
    try:
        p_choices = softmax_2d(u)[choices]
        loglik_trials = np.log(p_choices.astype(np.float64)+np.nextafter(np.float64(0), np.float64(1)))
        loglik_sum = np.sum(loglik_trials, axis=0)
        return -loglik_sum
    except IndexError:
        return np.nan
