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
plt.rcParams['font.size'] = 6
plt.rc('font', size= 6)
plt.rc('legend', fontsize= 4)
plt.rc('xtick', labelsize= 2)
plt.rc('xtick.major', size=2)
plt.rc('ytick.major', size=2)

plt.rc('ytick', labelsize= 2)
plt.rc('axes', labelsize=6)
plt.rc('axes', titlesize=6)


def glob_import_raw(globpath):

    files = glob.glob(globpath+'/**/*.xy', recursive=True)
    files.sort()
    files_new = []
    for f in files:
        if ('/proc' not in f):
            files_new.append(f)

    files = files_new
    print(files)
    experiments = [xps_data_import(path=f) for f in files]
    for xp in experiments:
        print('Imported ', xp.name)
    return experiments

def bg_subtraction(experiments: list, regions) ->list:

    bg_exps = batch_bg_subtract(experiments, regions, flag_plot=False)

    """bg2 = []
    fig, ax  = plt.subplots(1, 3)
    for xp in bg_exps:
        xp = subtract_linear_bg(xp, 'N_1s', ax=ax[0])
        if 'clean' not in xp.name:
            xp = subtract_shirley_bg(xp, 'N_1s', ax=ax[1])
        else: xp = subtract_als_bg(xp, 'N_1s', ax=ax[1])
        #fix_tail_bg(xp, 'O_1s', edw=535.5, ax=ax[2], inplace=True)
        bg2.append(xp)
    bg_exps = bg2
    bg_exps = region_2bg_subtract(bg_exps, region='Ba_3d', xlim=791.4, flag_plot=False)
"""
    return bg_exps

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

def compress_regions(bg_exps: list, indRef: int, region='N1s', flag_plot:bool = True):
    for xp in bg_exps:
        if 'FBI' not in xp.name:
            compress_noisy_region(xp=xp, xpRef=bg_exps[indRef], region=region, flag_plot=flag_plot, inplace=True)

def fast_preproc_main(globpath : str):
    experiments = glob_import_raw(globpath)

    regions = ['C_1s', 'O_1s', 'N_1s', 'O_1s_(2)', 'Cl_2p', 'Au_4f', 'Ba_3d']

    bg_exps = bg_subtraction(experiments, regions)
    print('Background subtracted, check plots')
    plot_xp_regions(bg_exps, regions, ncols=3);

    compress_regions(bg_exps, indRef=0, region='N_1s', flag_plot=False)
    #compress_regions(bg_exps, indRef=0, region='Ba_3d', flag_plot=False)


    scaled_exps = scale_and_plot_spectra(bg_exps+[bg_exps[0]], region='Au_4f', flag_plot=False)

    plot_xp_regions(scaled_exps, regions, ncols=3);
    #plot_normal_regions(scaled_exps, regions, ncols=3);

    print('Experiments scaled, check plots')

    store_results(bg_exps, scaled_exps)
    plt.show()
######### Main #########

if __name__ == "__main__":
    globpath = sys.argv[1]

    fast_preproc_main(globpath)
