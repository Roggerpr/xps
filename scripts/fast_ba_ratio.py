import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
import peakutils
import datetime
import warnings
import logging

from xps.xps_sw import *
from xps.xps_import import *
from xps.xps_analysis import *
from xps.xps_bg import *

logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

plt.rcParams['errorbar.capsize'] = 8
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 2.5
plt.rc('font', size= 18)
plt.rc('legend', fontsize= 15)
plt.rc('xtick', labelsize= 18)
plt.rc('xtick.major', size=6)
plt.rc('ytick.major', size=6)

plt.rc('ytick', labelsize= 18)
plt.rc('axes', labelsize=18)
plt.rc('axes', titlesize=18)

asf = dict({'C1s' : 0.296, 'O1s' : 0.711, 'N1s' : 0.477, 'Ba3d' : 7.49,
            'Br3p' : 1.054, 'Br3d' : 1.054, 'In3d_(2)' : 4.359, 'Na1s': 1.685,
           'In3d' : 4.359, 'Sn3d' : 4.725, 'Cl2p' : 0.891, 'Si2p': 0.339})

mfps = {'Cu2p' : 1.86, 'In3d': 3.05, 'Si2p': 3.8}

nm = 1e-6

def glob_import_unscaled(globpath: str) -> list:
    files = glob.glob(globpath+'/**/*.uxy', recursive=True)
    files.sort()

    experiments = [read_processed_xp(path=f) for f in files]
    for xp in experiments:
        print('Imported ', xp.name)
    return experiments

def bulk_integrate_areas(experiments: list, regions: list) ->list:

    #fig, ax = plt.subplots
    if len(experiments) == 1:
        experiments = [experiments[0], experiments[0]]
    for i,r in enumerate(regions):
        integrateRegions(experiments, region=r, asf=asf)#, indRef=indRefs[i])
    #integrateRegions(experiments, region='Ba3d', asf=asf, edw=775, eup=801)

def plot_coverages(experiments):
    layers, dlayers = [], []
    names = []
    for xp in experiments:
        try:
            layers.append(xp.area['layers'])
            dlayers.append(xp.area['dlayers'])
            names.append(xp.name)
        except KeyError:
            pass

    fig = plt.figure()
    plt.errorbar(x=names, y=layers, yerr=dlayers, fmt='o', label='Rate $R_0$')
    ax = plt.gca()
    ax.set_ylabel('Layers')
    ax.legend()

def store_results(exps: list):

    for xpu in exps:
        print('Stored ', xpu.path)
        write_processed_xp(xpu.path, xpu)

def area_molec(exps: list, molec: str):
    for xp in exps:
        if molec == 'fbisl':
            xp.area[molec] = xp.area['N1s'] / 4
        elif molec == 'fbi':
            xp.area[molec] = xp.area['N1s'] / 3
        elif molec == 'ann205':
            xp.area[molec] = xp.area['N1s'] / 7


def fast_ratio_main(globpath : str, molec: str = 'fbisl'):
    experiments = glob_import_unscaled(globpath)

    regions = list(experiments[0].dfx.columns.levels[0].values)
    for i,r in enumerate(regions):
        if ('_bg' in r) or ('overv' in r): regions.pop(i)

    print('Regions: ', regions, '\n')
    bulk_integrate_areas(experiments, regions)

    area_molec(experiments, molec)
    num = ['Ba3d', 'Na1s']
    denom = [molec, molec]
    print('Area molec: ', molec, experiments[0].area[molec])

    print('Stoichiometry values: \n\n\n------------------------------------------------------------------')
    make_stoichometry_table(experiments,  num=num, denom=denom, sep=' \t ')
    print('------------------------------------------------------------------\n\n\n')
    store_results(experiments)
    plt.show()

######### Main #########
if __name__ == "__main__":
    globpath = sys.argv[1]
    molec = sys.argv[2]
    if molec == None: molec = 'fbisl'
    fast_ratio_main(globpath)
