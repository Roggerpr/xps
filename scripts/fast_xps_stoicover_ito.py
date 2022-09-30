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

asf = dict({'C1s' : 0.296, 'O1s' : 0.711, 'O1s_sub' : 0.711, 'N1s' : 0.477, 'Ba3d' : 7.49, 'Ba_3d_5/2' : 7.49, 'Ba_3d_3/2' : 5.20,
            'Br3p' : 1.054, 'Cu_2p' : 5.321, 'Ba4d': 2.35, 'Na1s' : 1.685, 'Cl2s' : 0.37, 'Ru3d' : 4.273,
           'In3d' : 4.359, 'Sn3d' : 4.725, 'Cl2p' : 0.891, 'Si2p': 0.339, 'Ag3d': 5.987, 'Zn2p': 3.576})

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
        integrateRegions(experiments, region=r, asf=asf)#, indRef=indRefs[i])
    #integrateRegions(experiments, region='Ba3d', asf=asf, edw=775, eup=801)

def filter_he_tail(subo: list, cleans: list, cleanRef: int = 0):
    """Crop spectra up to the peak of the original clean"""
    x = cleans[cleanRef].dfx['O1s'].energy.dropna()
    y = cleans[cleanRef].dfx['O1s'].counts.dropna()

    epeak = x[np.argmax(y)]
    for xp in subo:
        crop_spectrum(xp, region='O1s_sub', edw=epeak, inplace=True)

def subtract_ito_ox(exps: list, cleanRef: int = 0):

    ### Sort experiments in clean and coated
    cleans = [xp for xp in exps if 'clean' in xp.name ]
    g2 = [xp for xp in exps if 'clean' not in xp.name]

    ### Scale to O1s
    scalo = scale_and_plot_spectra([cleans[cleanRef]] + g2,
    indRef=0, region='O1s', flag_plot=False)

    ### Subtract clean (ITO) component
    subo = []
    for xp in scalo[1:]:
        xpsub = subtract_ref_region(xp, scalo[0], 'O1s')
        subo.append(xpsub)

    ### Filter and background subtraction
    filter_he_tail(subo, cleans, cleanRef)
    subo = region_bg_subtract(subo, 'O1s_sub', flag_plot=False)

    return subo

def fast_sto_main(globpath : str, cleanRef: int = 0):
    unscaled = glob_import_unscaled(globpath)
    #unscaled = unscaled[:10]
    #regions = ['C1s', 'N1s', 'O1s', 'In3d', 'Si2p']
    regions = get_all_regions(unscaled)     # Get regions of all experiments
    newregs = []

    for r in regions:                           # Filter out '_bg' or '(2), (3)' ones
        if ('(' not in r) and ('_bg' not in r):
            newregs.append(r)
    regions = newregs
    #print(regions)

    subo = subtract_ito_ox(unscaled, cleanRef = cleanRef)

    bulk_integrate_areas(subo, regions)
    print('Stoichiometry values:')

    num, denom = (('N1s', 'C1s', 'C1s', 'N1s', 'C1s'), ('O1s', 'N1s', 'O1s', 'Si2p', 'Si2p'))
    make_stoichometry_table(subo,  num=num, denom=denom, sep=' \t ')

    bas = [xp for xp in subo if '_Ba' in xp.name]
    for xp in bas:
        xp.area['G2'] = xp.area['N1s'] / 4

    num, denom = (('Ba3d', 'Ba3d'), ('N1s', 'Cl2p'))
    make_stoichometry_table(bas,  num=num, denom=denom, sep=' \t ')



    #inds = [[4, 3]]#, [3, 2]]

    #inds = [guess_clean_xp(experiments)]

    #layers_fbi = arrange_coverages(experiments, inds,
    #                           r_ml = 1.1*nm, region='In3d', mfp = mfps['In3d']*nm, takeoff = 10)


    #plot_coverages(experiments)
    store_proc_results(unscaled)
    store_proc_results(subo)

    plt.show()

######### Main #########
if __name__ == "__main__":
    globpath = sys.argv[1]
    cleanRef = int(sys.argv[2])
    if cleanRef == None: cleanRef = 0

    fast_sto_main(globpath, cleanRef)
