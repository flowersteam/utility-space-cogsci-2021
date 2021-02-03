import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols, mixedlm
from scipy import stats


print(f'Using local utils in {__file__}')


def delta_aic(x):
    return x - x.min()


def akaike_w(x):
    return np.exp(-x/2) / np.sum(np.exp(-x/2))


def evidence_ratio(x):
    return x.iloc[0] / x


def gen_regplots(df, figname, cols, rows, xlabels=None, ylabels=None):    
    fig, ax = plt.subplots(ncols=len(cols), nrows=len(rows), num=figname, figsize=[3*len(cols), 2.5*len(rows)])
    if ax.ndim == 1:
        ax = ax[np.newaxis, :]
    for j, col in enumerate(cols):
        df[f'z{col}'] = stats.zscore(df.loc[:, col])
        print(f'{col.upper()}:')
        for i, row in enumerate(rows):
            r, pval = stats.pearsonr(df[row], df[col])
            sign = '*' if pval < .05 else ''
            sign = '**' if pval < .01 else sign
            print(f'{row}: r = {r:.3f}, p = {pval:.3f}  {sign}')
            report = f'r = {r:.2f}\np = {pval:.3f}{sign}'
            ax[i, j].text(.05, .98, report, va='top', ha='left', 
                          transform=ax[i, j].transAxes, fontsize=14,
                          color='black' if sign else 'gray')
            
            sns.regplot(y=row, x=col, data=df, ax=ax[i, j], color='red' if sign else 'gray', scatter_kws={'s':5, 'alpha':.3}, order=1)
            if j:
                ax[i,j].set_ylabel('')
            else:
                if ylabels:
                    ax[i,j].set_ylabel(ylabels[row])
            if i<len(rows)-1:
                ax[i,j].set_xlabel('')
            else:
                if xlabels:
                    ax[i,j].set_ylabel(ylabels[col])
    return fig, ax


def gen_regplots2(df, figname, cols, rows, xlabels=None, ylabels=None):    
    fig, ax = plt.subplots(ncols=len(cols), nrows=len(rows), num=figname, figsize=[3*len(cols), 2.5*len(rows)])
    for j, col in enumerate(cols):
        df[f'z{col}'] = stats.zscore(df.loc[:, col])
        print(f'{col.upper()}:')
        for i, row in enumerate(rows):
            reg = ols(f'{row} ~ 1 + {col}', data=df).fit()
            qreg = ols(f'{row} ~ z{col} + np.power(z{col}, 2)', data=df).fit()
            aic_diff = reg.aic - qreg.aic
            model = [reg, qreg][np.argmin([reg.aic, qreg.aic])]
            
            f, fpval = model.fvalue, model.f_pvalue
            sign = '*' if fpval < .05 else ''
            sign = '**' if fpval < .01 else sign
            print(f'{row} [∆aic(L,Q)={aic_diff:.2f})]: {f:.3f}, {fpval:.3f}  {sign}')
            b2, pval = model.params.values[-1], model.pvalues.values[-1]
            report = f'r = {b2:.2f}\np = {pval:.3f}{sign}'
            ax[i, j].text(.05, .98, report, va='top', ha='left', 
                          transform=ax[i, j].transAxes, fontsize=14,
                          color='black' if sign else 'gray')
            
            sns.regplot(y=row, x=col, data=df, ax=ax[i, j], color='red' if sign else 'gray', scatter_kws={'s':5, 'alpha':.3}, order=2 if aic_diff>2 else 1)
            if j:
                ax[i,j].set_ylabel('')
            else:
                if ylabels:
                    ax[i,j].set_ylabel(ylabels[row])
            if i<len(rows)-1:
                ax[i,j].set_xlabel('')
            else:
                if xlabels:
                    ax[i,j].set_ylabel(ylabels[col])
    return fig, ax


def rename_components(df):
    df = df.replace('delta_mean', '∆M', regex=True)
    df = df.replace('delta_var', '∆V', regex=True)
    df = df.replace(['mean', 'var'], ['M', 'V'], regex=True)
    df = df.rename(columns={
        'mean': 'comp',
        'var': 'unc', 
        'delta_mean': 'dcomp',
        'delta_var': 'dunc'
    })
    return df


def delta_aic_of_top5(data_path):
    df = pd.read_csv(data_path).filter(items=['group', 'sid', 'form', 'drank', 'delta_aicc'])
    df = lut.rename_components(df)
    df.loc[:, 'drank'] += 1
    df['also_good'] = df.delta_aicc.lt(3) & df.drank.gt(1)
    also_good = df.groupby(['group','sid'])[['also_good']].sum()
    print(also_good.mean())
    
    
def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def filter_learners(df, bdata_path, n, by, which, flat=False):
    bdf = pd.read_csv(bdata_path).set_index(['sid'])
    bdf = bdf.loc[bdf.group==0, :]
    bdf['prog'] = bdf.fpc15 - bdf.ipc15
    bdf_learnable = bdf.loc[bdf.activity!='A4', :]
    grouped_bdf = bdf.groupby('sid')
    grouped_bdf_learnable = bdf_learnable.groupby('sid')
    outdf = grouped_bdf[['n']].apply(lambda x: np.std(x))
    outdf = outdf.loc[df.sid.unique(), :]

    weighted_mean = lambda x: np.sum(x.values.squeeze()*(np.array([1,2,3])/6))
    flat_mean = lambda x: np.mean(x.values.squeeze())

    func = flat_mean if flat else weighted_mean

    outdf.columns = ['alloc_bias']
    outdf['final_rPC'] = grouped_bdf_learnable[['fpc15']].apply(func)
    outdf['delta_rPC'] = grouped_bdf_learnable[['prog']].apply(func)
    outdf['PC'] = grouped_bdf_learnable[['pcall']].apply(func)
    outdf['spd'] = grouped_bdf_learnable[['speed']].apply(func)
    
    outdf.reset_index(inplace=True)
    if which=='top':
        outdf = outdf.nlargest(n, by)
    elif which=='bot':
        outdf = outdf.nsmallest(n, by)
    return df.set_index('sid').loc[outdf.sid.unique(), :].reset_index()
    

def ghost(ax):
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    return ax


def add_subplot_label(ax, x, y, label, size, preview=False):
    ax.text(x, y, label, transform=ax.transAxes,
        size=size, weight='bold')
    if preview:
        ghost(ax)
        ax.set_facecolor('gray')
    else:
        ghost(ax)
        
        
def get_fmean(x):
    y = pd.Series(x).rolling(window=15).mean().values
    return np.nan_to_num(y, nan=y[~np.isnan(y)][0])


def get_fdmean(x):
    y = pd.Series(x).rolling(window=15).apply(rlp_func).values
    return np.nan_to_num(y, nan=y[~np.isnan(y)][0])


def get_fvar(x):
    y = pd.Series(x).rolling(window=15).var().values
    return np.nan_to_num(y, nan=y[~np.isnan(y)][0])


def get_fdvar(x):
    y = pd.Series(x).rolling(window=15).apply(rlp_func2).values
    return np.nan_to_num(y, nan=y[~np.isnan(y)][0])


def rlp_func(x):
    return np.abs(np.mean(x[-9:]) - np.mean(x[:10]))


def rlp_func2(x):
    return np.abs(np.var(x[-9:]) - np.var(x[:10]))
