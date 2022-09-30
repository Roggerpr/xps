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

def glob_import_raw(globpath: str) -> list:
    files = glob.glob(globpath+'/**/*.xy', recursive=True)
    files.sort()
    files_new = []
    for f in files:
        if ('/proc' not in f):
            files_new.append(f)

    files = files_new

    experiments = [xps_data_import(path=f) for f in files]
    for xp in experiments:
        print('Imported ', xp.name)
    return experiments

def bg_subtraction(experiments: list, regions: list) ->list:

    bg_exps = region_2bg_subtract(experiments, region='Cu_2p', xlim=938, flag_plot=False)


    bg_exps = batch_bg_subtract(bg_exps, regions, )
    return bg_exps

def compress_regions(bg_exps: list, indRef: int, region='# NOTE: 1s', flag_plot:bool = True):
    for xp in bg_exps:
        if 'FBI' not in xp.name:
            compress_noisy_region(xp=xp, xpRef=bg_exps[indRef], region=region, flag_plot=flag_plot, inplace=True)

def store_results(bg_exps: list, scaled_exps: list):

    for xpu, xps in zip(bg_exps, scaled_exps):
        filepath, filename = os.path.split(xpu.path)
        filename = os.path.splitext(filename)[0]
        newpath = filepath + '/proc/'
        try:
            os.mkdir(newpath)
        except FileExistsError: pass
        print('Stored ', newpath + filename)
        write_processed_xp(newpath + filename + '.uxy', xpu)
        write_processed_xp(newpath + filename + '.sxy', xps)

def fast_preproc_main(globpath : str):
    experiments = glob_import_raw(globpath)

    print('Imported:')
    for xp in experiments:
        print(xp.name)
    regions = ['C_1s', 'N_1s', 'O_1s', 'Ba_3d', 'Cl_2p', 'Cu_2p', 'Na_1s']
    #regions = ['C_1s', 'N_1s', 'O_1s', 'Cu_2p']

    bg_exps = bg_subtraction(experiments, regions)
    print('Background subtracted, check plots')
    compress_regions(bg_exps, indRef=0, region='N_1s', flag_plot=False)
    compress_regions(bg_exps, indRef=0, region='O_1s', flag_plot=False)

    plot_xp_regions(bg_exps, regions, ncols=3);
    scaled_exps = scale_and_plot_spectra(bg_exps, region='Cu_2p')

    plot_xp_regions(scaled_exps, regions, ncols=3);
    plot_normal_regions(scaled_exps, regions, ncols=3);

    print('Experiments scaled, check plots')

    store_results(bg_exps, scaled_exps)
    plt.show()
######### Main #########
if __name__ == "__main__":
    globpath = sys.argv[1]

    fast_preproc_main(globpath)
