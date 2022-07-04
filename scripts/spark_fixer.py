import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys
from dataclasses import dataclass

from xps.xps_import import XPS_experiment, write_processed_xp


def xy_multiscan_delimiters(path: str) -> tuple:
    """Retrieve position, name and number of lines of each spectrum in a .xy file
        Each region contains N scans which is detected and stored to organize a dict of skiplines"""

    skipRows0 = []
    nrows0 = []
    names = []
    nscans = []
    scanCounter = 0

    with open(path) as infile:
        for i,line in enumerate(infile):
            if '# ColumnLabels: energy' in line:
                skipRows0.append(i)
                scanCounter += 1
            if '# Region:' in line:
                names.append(line[21:-1].replace(' ', '_'))
                nscans.append(scanCounter)
                scanCounter = 0
            if '# Values/Curve:' in line:
                if len(nrows0) == len(names)-1:
                    nrows0.append(int(line[21:-1]))

    nscans = nscans[1:]          # Drop the first 0
    nscans.append(scanCounter)   # Append the last count of scans

    skipdict = {}
    indaux = 0
    for r, scan in zip(names, nscans):
        skipdict[r] = skipRows0[indaux:indaux+scan]
        indaux += scan
    return (skipdict, nrows0, names)

def import_multiscan_df(path: str, skipdict: dict, nrows0: list, regions: list):
    """Import a file with scans stored separately, return dfx only
        dfx is a dict with the scans of each region stored as an entry"""
    dfx = {}
    for i, r in enumerate(regions):
        frames = []
        nscans = []
        for j, sk in enumerate(skipdict[r]):
            dfscan = pd.read_table(path, sep='\s+', skiprows=sk+2, nrows = nrows0[i], header=None, names=['energy_'+r, 'scan'+str(j)],
                                            decimal='.', encoding='ascii', engine='python')
            if j == 0:
                energy = dfscan['energy_'+r]
            frames.append(dfscan.drop('energy_'+r, axis=1))

        dfr = pd.concat(frames, axis=1)
        dfr.index=energy
        dfx[r] = dfr
    return dfx

def sparks_region_fix(dfr: pd.DataFrame):
    """Fix the drops in counts (sparks) by droping the whole scan
        if the minimum is lower than 85% of the baseline"""

    absmin = dfr.mean(axis=1).min()
    for c in dfr.columns:
        if dfr[c].min() < absmin*0.85:
            dfr.drop(c, axis=1, inplace=True)
            print('Dropped ', c)
    return dfr

def sparks_dfx_fix(dfx: dict):
    """Iterate over the regions of the file and fix the sparks in each of them"""
    for r in dfx.keys():
        dfx[r] = sparks_region_fix(dfx[r])
    return dfx

def scans_mean_plot(dfr: pd.DataFrame, ax=None):
    """Plot the mean of a region with N scans"""
    if ax == None: ax = plt.gca()
    dfr.mean(axis=1).plot(ax=ax)
    plt.gca().invert_xaxis()

def scans_plot_all(dfr: pd.DataFrame, ax=None):
    """Plot all the scans of a region"""
    if ax == None: ax = plt.gca()
    for c in dfr.columns:
        dfr[c].plot(ax=ax)
    plt.gca().invert_xaxis()

def scans_multi_plot(dfx: dict, regs: list, mode=['all', 'mean'],  ncols = 3):
    nrows = int(np.ceil(len(regs) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(nrows*8, ncols*4))
    for i,r in enumerate(regs):
        j, k = i//ncols, i%ncols
        if mode == 'all':
            scans_plot_all(dfx[r], ax=ax[j,k])
        else:
            scans_mean_plot(dfx[r], ax=ax[j,k])
        ax[j,k].set(xlabel=None, yticks=[])
        ax[j,k].invert_xaxis()
        ax[j,k].text(s=r.replace('_', ' '), y=0.9, x=0.1, transform=ax[j][k].transAxes)
    plt.tight_layout()

def set_fixed_dfx(dfx: dict) -> pd.DataFrame:
    """Convert dfx (dict of dfx scans) to regular dfx with the mean over all scans"""
    frames = []
    for r in dfx.keys():
        frames.append(pd.DataFrame( {'energy': dfx[r].index, 'counts': dfx[r].mean(axis=1).values}))
    dfix = pd.concat(frames, axis=1)

    index2 = np.array(['energy', 'counts'])
    mi = pd.MultiIndex.from_product([list(dfx.keys()), index2], names=['range', 'properties'])
    mi.to_frame()
    dfix.columns = mi
    return dfix

def set_fixed_xp(path, dfx, delimiters):
    """Arrange the whole fixed XPS_experiment with name etc from filename and fixed dfx"""
    relpath, filename = os.path.split(path)
    dir_name = os.path.split(relpath)[1]
    da = re.search('\d+_', filename).group(0).replace('/', '').replace('_', '')
    if da[:4] != '2020':
        date = re.sub('(\d{2})(\d{2})(\d{4})', r"\1.\2.\3", da, flags=re.DOTALL)
    else:
        date = re.sub('(\d{4})(\d{2})(\d{2})', r"\1.\2.\3", da, flags=re.DOTALL)

    other_meta = filename.replace(da, '')[1:].strip('.xy')
    name = other_meta
    label = da+'_'+other_meta

    dfix = set_fixed_dfx(dfx)

    xpfixed = XPS_experiment(path = path.replace('_scans', ''), dfx = dfix, delimiters = delimiters, color = None,
                              name = name, label = label, date = date, other_meta = other_meta, fit={})
    return xpfixed

def main(path: str):
    delimiters = xy_multiscan_delimiters(path)
    skipdict, nrows0, regions = delimiters[:]

    dfspark = import_multiscan_df(path, skipdict, nrows0, regions)
    regs = list(dfspark.keys())
    scans_multi_plot(dfspark, regs)

    dffixed = sparks_dfx_fix(dfspark)
    scans_multi_plot(dffixed, regs)

    xpfixed = set_fixed_xp(path, dffixed, delimiters)
    write_processed_xp(xpfixed.path.replace('_scans.xy', '.uxy'), xpfixed)
    plt.show()
    
if __name__ == "__main__":
    path = sys.argv[1]
    main(path)
