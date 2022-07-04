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

asf = dict({'C_1s' : 0.296, 'O_1s' : 0.711, 'N_1s' : 0.477, 'Ba_3d' : 7.49,
            'Br_3p' : 1.054, 'Cu_2p' : 5.321, 'Ba_4d': 2.35, 'Na_1s' : 1.685,
           'In3d' : 4.359, 'Sn3d' : 4.725, 'Cl_2p' : 0.891, 'Si2p': 0.339})

mfps = {'Cu_2p' : 1.86, 'In3d': 3.05}

nm = 1e-6

def glob_import_unscaled(globpath: str) -> list:
    files = glob.glob(globpath+'/**/*.uxy', recursive=True)
    files.sort()

    experiments = [read_processed_xp(path=f) for f in files]
    for xp in experiments:
        print('Imported ', xp.name)
    return experiments

def bulk_integrate_areas(experiments: list, regions: list) ->list:

    indRefs = [0, 0, 0, 0, 0, 0]

    for i,r in enumerate(regions):
        integrateRegions(experiments, region=r, asf=asf, indRef=indRefs[i])

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
    ax = plt.gca()
    ax.errorbar(x=names, y=layers, yerr=dlayers, fmt='o', label='Rate $R_0$')

    ax.set_ylabel('Layers')
    plt.draw()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend()

def store_results(exps: list):

    for xpu in exps:
        print('Stored ', xpu.path)
        write_processed_xp(xpu.path, xpu)


def fast_preproc_main(globpath : str):
    experiments = glob_import_unscaled(globpath)

    regions = ['C_1s', 'N_1s', 'O_1s', 'Cu_2p', 'Ba_3d', 'Cl_2p', 'Na_1s']

    bulk_integrate_areas(experiments, regions)
    print('Stoichiometry values:')

    num, denom = (('N_1s', 'C_1s', 'C_1s' ), ('O_1s', 'N_1s', 'O_1s',))
    make_stoichometry_table(experiments,  num=num, denom=denom, sep=' \t ')

    #num, denom = ('N_1s', 'Cl_2p'), ( 'Na_1s', 'Na_1s')
    #make_stoichometry_table([experiments[2]],  num=num, denom=denom, sep=' \t ')

    #inds = [[0, 1]]

    #layers_fbi = arrange_coverages(experiments, inds,
    #                           r_ml = 1.1*nm, region='Cu_2p', mfp = mfps['Cu_2p']*nm, takeoff = 10)

    #for xp in experiments:
    #    print(xp.area['layers'])
    #plot_coverages(experiments)
    store_results(experiments)
    #plt.show()

######### Main #########
if __name__ == "__main__":
    globpath = sys.argv[1]

    fast_preproc_main(globpath)
