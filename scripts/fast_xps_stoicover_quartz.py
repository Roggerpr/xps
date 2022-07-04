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

asf = dict({'C1s' : 0.296, 'O1s' : 0.711, 'N1s' : 0.477, 'Ba3d' : 7.49, 'Na1s': 1.685,
            'Br3p' : 1.054, 'Br3d' : 1.054,
           'In3d' : 4.359, 'Sn3d' : 4.725, 'Cl2p' : 0.891, 'Si2p': 0.339})

asfScof = {'O1s': 0.6613995485327314, 'Si2p': 0.18442437923250563, 'N1s': 0.4063205417607223, 'C1s': 0.2257336343115124,
           'In3d': 5.088036117381489, 'Ba3d5/2': 5.832957110609481, 'Cl2p': 0.5158013544018059, 'Ru3d5/2': 1.6681715575620766,
           'Si2s': 0.21557562076749434, 'Ru3p3/2': 1.530474040632054, 'Ir4d5/2': 2.4604966139954856}


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


    for i,r in enumerate(regions):
        integrateRegions(experiments, region=r, asf=asfScof)#, indRef=indRefs[i])
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

def fast_stoicover_main(globpath : str):
    experiments = glob_import_unscaled(globpath)

    regions = ['C1s', 'N1s', 'O1s', 'Si2p']#, 'Ba3d', 'Cl2p', 'Na1s', 'Ir4d5/2']

    bulk_integrate_areas(experiments, regions)
    print('Stoichiometry values:')

    num, denom = (('N1s', 'C1s', 'C1s'), ('O1s', 'N1s', 'O1s'))
    make_stoichometry_table(experiments,  num=num, denom=denom, sep=' \t ')

#    num, denom = (('N1s', 'Cl2p'), ('Ba3d', 'Ba3d'))
#    make_stoichometry_table(experiments[:2],  num=num, denom=denom, sep=' \t ')


    #inds = [[0, 1]]#, [3, 2]]
    #inds = [guess_clean_xp(experiments)]
    #layers_fbi = arrange_coverages(experiments, inds,
    #                           r_ml = 1.1*nm, region='Si2p', mfp = mfps['Si2p']*nm, takeoff = 10)

    #plot_coverages(experiments)
    store_results(experiments)
    plt.show()

######### Main #########
if __name__ == "__main__":
    globpath = sys.argv[1]

    fast_stoicover_main(globpath)
