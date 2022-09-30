import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
from copy import deepcopy
from scipy.optimize import curve_fit

from dataclasses import dataclass
from xps.xps_import import XPS_experiment
from xps.xps_fits import *

from xps.xps_analysis import plot_region, cosmetics_plot, crop_spectrum

def find_and_plot_peaks(df : pd.DataFrame, thres : float = 0.5, min_d : int = 10, col : str = 'r'):
    leny = len(df.index)
    peaks =  peakutils.indexes(df.counts.values, thres=thres, min_dist=min_d)
    x_peaks = leny - df.index[peaks]
    y_peaks = df.counts.values[peaks]
    plt.plot(x_peaks, y_peaks, col+'o', label='Peaks at thres = %.1f' %thres)

    return peaks


def find_shift(xp : XPS_experiment, xpRef : XPS_experiment, region : str) -> float:
    """Compare maxima between two spectra and get energy shift"""
    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts.values
    xRef = xpRef.dfx[region].dropna().energy
    yRef = xpRef.dfx[region].dropna().counts.values

    imax = np.argmax(y)
    irmax = np.argmax(yRef)
    shift = xRef[irmax] - x[imax]
    return shift

def align_dfx(xp : XPS_experiment, shift : float, inplace : bool = False) -> XPS_experiment:
    """Apply energy shift to dfx and return xp aligned with reference"""
    names = list(xp.dfx.columns.levels[0])
    dfnew = pd.DataFrame()
    frames = []
    for n in names:
        frames.append( pd.DataFrame([xp.dfx[n].energy + shift, xp.dfx[n].counts]).T )
    dfnew = pd.concat(frames, axis=1)

    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
    mi.to_frame()
    dfnew.columns = mi

    if inplace:
        xp.dfx = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx = dfnew
    return xpNew

def align_double_fit(bg_exps: list, xpRef: XPS_experiment, region: str, inplace: bool = False):
    """
        Fit double voigt and align by position of fit center
        Input:
        ------------------
            bg_exps: list of experiments to align
            xpRef: xp with reference energy. MUST BE fitted already
            region: str
            inplace: bool
                if True, overwrite bg_exps with aligned regions
        Output:
            aligned_exps: list of experiments aligned to xpRef
    """
    if xpRef.fit[region].best_values['v1_center'] > xpRef.fit[region].best_values['v2_center']:
        peak_ref = xpRef.fit[region].best_values['v2_center']
    else:
        peak_ref = xpRef.fit[region].best_values['v1_center']
    fig = plt.figure()
    aligned_exps = []
    for xp in bg_exps:
        try:
            fn = XPFit(xp, region)
            fitvv = fn.double_voigt()
        except KeyError:
            inplace = False
            pass
    #     plot_fit_result(xpc, region)
        if fitvv.best_values['v1_center'] > fitvv.best_values['v2_center']:
            shiftIn = scaled[0].fit[region].best_values['v1_center'] - fitvv.best_values['v2_center']
        else:
            shiftIn = scaled[0].fit[region].best_values['v1_center'] - fitvv.best_values['v1_center']
        xpa = align_dfx(xp, shiftIn)
        plot_region(xpa, region)
        aligned_exps.append(xpa)
    if inplace:
        bg_exps = aligned_exps
    else:
        return aligned_exps

def scale_and_plot_spectra(bg_exps : list, indRef: int = 0 , region : str = 'overview_',
                           bl_deg : int = 5, flag_plot:bool = True) -> float:
    """Plot two spectra and compute average count ratio between main peaks for scaling
        Input:
        -----------------
        xp: XPS_experiment
            Experiment containing the spectrum region to scale UP
        xpRef: XPS_experiment
            Reference spectrum to compare to
        lb : tuple
            Labels for legend
        Output:
        ------------------
        y_sc : array
            scaled counts
        normAv : float
            Scale factor computed as the average ratio between peak heights. Should be > 1,
            otherwise the reference spectrum has weaker intensity than the one intended to scale up
        indmax : int
            Index position of the highest peak"""
    fig, ax = plt.subplots(len(bg_exps), 2, figsize=(16, 7 * len(bg_exps)))

    scaled_exps = []
    xpRef = bg_exps[indRef]
    for j, xp in enumerate(bg_exps):

        lb = (xp.name, xpRef.name)
        try:
            df, dfRef = xp.dfx[region].dropna(), xpRef.dfx[region].dropna()
        except KeyError as e:
            print('KeyError in %s' %e)
            pass

        scale_factor = (np.max(dfRef.counts) - np.min(dfRef.counts)) / (np.max(df.counts) - np.min(df.counts))
        y_scale = df.counts * scale_factor

        scaled_exps.append(scale_dfx(xp = xp, scale_factor = scale_factor))

        if flag_plot:
            ax[j][0].plot(df.energy, df.counts, '-b', label=lb[0])
            ax[j][0].plot(dfRef.energy, dfRef.counts, '-r', label=lb[1] + ' (ref.)')

            indmax = np.argmax(dfRef.counts.values) # Get only highest peak
            indmin = np.argmin(dfRef.counts[indmax : ]) # Get absolute minimum in near neighbourhood
            ax[j][0].axhline(dfRef.counts[indmax], color='k', ls = '--')
            ax[j][0].axhline(dfRef.counts[indmin], color='k', ls = '--')
            ax[j][0].axvline(dfRef.energy[indmax], color='k', ls = '--')

            cosmetics_plot(ax = ax[j][0])
            ax[j][0].set_title('Baseline and peak')


            ax[j][1].plot(df.energy, y_scale, '-b', label=lb[0])
            ax[j][1].plot(dfRef.energy, dfRef.counts , '-r', label=lb[1]+ ' (ref.)')
            cosmetics_plot(ax = ax[j][1])
            ax[j][1].set_title('Scaling result')
    if flag_plot: fig.tight_layout()

    return scaled_exps

def scale_dfx(xp : XPS_experiment, scale_factor : float, inplace : bool = False):
    """Rescale xp.dfx for comparison with other experiment and subtract baseline
    Returns whole XPS_experiment"""
    names = list(xp.dfx.columns.levels[0])
    dfnew = pd.DataFrame()

    frames = []
    for n in names:
        y = xp.dfx[n].dropna().counts
        ysc = y.apply(lambda c : c * scale_factor)
        frames.append( pd.DataFrame([xp.dfx[n].energy, ysc]).T )
    dfnew = pd.concat(frames, axis=1)

    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
    mi.to_frame()
    dfnew.columns = mi

    if inplace:
        xp.dfx = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx = dfnew
    return xpNew

def normalise_dfx(xp : XPS_experiment, inplace : bool = False):
    """Normalise spectrum counts to maximum peak at index position indmax"""

    names = list(xp.dfx.columns.levels[0])
    dfnew = pd.DataFrame()

    frames = []
    for n in names:
        y =  xp.dfx[n].dropna().counts
        ybase = y.apply(lambda c : c - np.min(y)  )  # If bg is subtracted this should do nothing
        ynorm = ybase.apply(lambda c :  c / np.max(ybase) )

        frames.append( pd.DataFrame([xp.dfx[n].energy, ynorm]).T )
    dfnew = pd.concat(frames, axis=1)

    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
    mi.to_frame()
    dfnew.columns = mi

    if inplace:
        xp.dfx = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx = dfnew
    return xpNew

def subtract_ref_region(xp : XPS_experiment, xpRef: XPS_experiment, region: str, inplace : bool = False):
    """Subtract the counts in a region in xpRef from xp, useful to account for clean substrate contributions"""

    df, dfRef = xp.dfx[region].dropna(), xpRef.dfx[region].dropna()
    y_sub = df.counts - dfRef.counts
    if inplace:
        xp.dfx[region+'_sub', 'energy'] = pd.Series(xp.dfx[region].energy)
        xp.dfx[region+'_sub', 'counts']  = pd.Series(y_sub)
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region+'_sub', 'energy'] = pd.Series(xp.dfx[region].energy)
        xpNew.dfx[region+'_sub', 'counts']  = pd.Series(y_sub)
        return xpNew

def clean_reg_subtract(exps: list, xpRef: XPS_experiment, region: str):
    """
        Subtract clean substrate contribution in region to whole set of exps.
        Store result in new dfx region as region+'_sub'
        input
        ------------
        exps: list of experiments to perform operation onto
        xpRef: XPS_experiment with clean region
        region: str
    """
    fig, ax = plt.subplots(len(exps), figsize=(6, 6*len(exps)))
    for i, xp in enumerate(exps):
        subexps = subtract_ref_region(xp, xpRef, region=region, )
        plot_region(xp, region, ax=ax[i])
        subexps.ls = '--'
        plot_region(subexps, region, ax=ax[i])
        plot_region(xpRef, region, ax=ax[i])
        ax[i].invert_xaxis()
        ax[i].legend()

        xp.dfx[region+'_sub', 'energy'] = subexps.dfx[region].energy
        xp.dfx[region+'_sub', 'counts'] = subexps.dfx[region].counts

def compress_noisy_region(xp: XPS_experiment, xpRef: XPS_experiment, region, lb: tuple = None, flag_plot:bool = True, inplace: bool = False):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    if lb == None: lb = (xp.name, xpRef.name)
    df, dfRef = xp.dfx[region].dropna(), xpRef.dfx[region].dropna()

    ax[0].plot(df.energy, df.counts, '-b', label=lb[0] + ' (bg.)')
    ax[0].plot(dfRef.energy, dfRef.counts, '-r', label=lb[1] + ' (sg.)')

    bl_factor = np.average(baseline_als(dfRef.counts))
    scale = bl_factor / np.max(df.counts)
    y_scale = df.counts * scale

    ax[1].plot(df.energy, y_scale, '-b', label=lb[0] + 'noise compressed')
    ax[1].plot(dfRef.energy, dfRef.counts , '-r', label=lb[1]+' (sg.)')
    ax[1].legend(); ax[0].legend()
    if not flag_plot: plt.clf(); plt.close()
    if inplace:
        xp.dfx[region] = pd.DataFrame([xp.dfx[region].energy, y_scale]).T
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region].counts = y_scale
        return xpNew

def find_integration_limits(x, y, flag_plot = False, region : str = None, ax = None):
    """Utility to locate limits for shirley bg subtraction"""
    # Locate the biggest peak.
    maxidx = np.argmax(y)

    # It's possible that maxidx will be 0 or -1. If that is the case,
    # we can't use this algorithm, we return a zero background.
    assert maxidx > 0 and maxidx < len(y) - 1, "specs.shirley_calculate: Boundaries too high for algorithm: returning a zero background."

    # Locate the minima either side of maxidx.
    lmidx = abs(y[0:maxidx] - np.min(y[0:maxidx])).argmin()
    rmidx = abs(y[maxidx:] - np.min(y[maxidx:])).argmin() + maxidx

    if flag_plot:
        if ax == None: ax = plt.gca()
        ybase = ax.get_ylim()[0]
        if ybase < 0: ybase = 0
        ind = [maxidx, lmidx, rmidx]
        for i in ind:
            ax.vlines(x = x[i], ymin=ybase, ymax=y[i], linestyles='--', color='k')
            ax.text(s= '%.2f'%x[i], x = x[i], y = y[i])

    return lmidx, rmidx

def align_subtract_clean(exps: list, xpRef: XPS_experiment, regScale: str, regSub: str,
                        bg_flag:bool = True, inplace:bool = False):
    """
        Scale and align exps in region regScale, wrt xpRef,
        then subtract clean substrate contribution in region regSub
        If bg_flag: subtract shirley background from new region_sub"""
    scaled = scale_and_plot_spectra([xpRef] + exps, region=regScale, flag_plot=False)
    aligned = align_double_fit(scaled[1:], xpRef=scaled[0],region=regScale)
    sub_exps = clean_reg_subtract(aligned, xpRef, regSub)
    if bg_flag:
        sub_exps = region_bg_subtract(sub_exps, regSub+'_sub')

    if inplace:
        exps[:] = sub_exps[:]
        return exps
    else:
        return sub_exps

"""        Shirley        """


def shirley_loop(x, y,
                 lmidx : int = None,
                 rmidx : int = None,
                 maxit : int = 10, tol : float = 1e-5,
                 DEBUG : bool = False,
                 flag_plot : bool = False,
                 ax = None):
    """Main loop for shirley background fitting"""
    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below lmidx and initially zero above.

#     x, y, is_reversed = check_arrays(x, y)

    if (lmidx == None) or (rmidx == None):
        lmidx, rmidx = find_integration_limits(x, y, flag_plot=flag_plot, ax = ax)
    xl, yl = x[lmidx], y[lmidx]
    xr, yr = x[rmidx], y[rmidx]

    B = np.zeros(x.shape)
    B[:lmidx] = yl - yr
    Bnew = B.copy()
    it = 0
    while it < maxit:
        if DEBUG:
            print ("Shirley iteration: %i" %it)

        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = np.trapz( + B[lmidx:rmidx - 1] + yr - y[lmidx:rmidx - 1] , x=x[lmidx:rmidx - 1])
        k = (yl - yr) / ksum

        # Calculate new B
        ysum = 0
        for i in range(lmidx, rmidx):
            ysum = np.trapz( B[i:rmidx - 1] + yr - y[i:rmidx - 1] , x=x[i:rmidx - 1])
            Bnew[i] = k * ysum

        # If Bnew is close to B, exit.
        if np.linalg.norm(Bnew-B) < tol:
            B = Bnew.copy()
            break
        else:
            B = Bnew.copy()
        it += 1

    assert it < maxit, "shirley_loop: Max iterations exceeded before convergence."
    if it >= maxit:
        print("specs.shirley_calculate: Max iterations exceeded before convergence.")
        return 0
#     if is_reversed:
#         return ((yr + B)[::-1])
    else:
        return (yr + B)

def subtract_shirley_bg(xp : XPS_experiment, region : str, maxit : int = 10,
                        lb : str = '__nolabel__', offset: float = 0,
                        ax = None, store: bool = True,
                        verbose: bool = False, inplace: bool = False) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values
    col = plot_region(xp = xp, region = region, lb = lb, ax = ax, offset=offset).get_color()

    find_integration_limits(x, y, flag_plot=verbose, region = region, ax = ax)
    ybg = shirley_loop(x, y, maxit = maxit)

    if ax == None: ax = plt.gca()
    ax.plot(x, ybg + offset, '--', color=col, label=lb);
    #cosmetics_plot(ax = ax)

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y - ybg})

    if inplace:
        xp.dfx[region] = dfnew
        if store:
            xp.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xp.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        if store:
            xpNew.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xpNew.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xpNew

def subtract_double_shirley(xp : XPS_experiment, region : str, xlim : float, maxit : int = 10,
                        lb : str = None, ax = None, flag_plot : bool = False,
                         store: bool = True, inplace: bool = False) -> XPS_experiment:
    """Shirley bg subtraction for double peak"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values
    col = plot_region(xp, region, ax = ax, lb=lb).get_color()
    if xlim == None: xlim = find_separation_point(x, y)

    y1 = y[ x >= xlim ]
    x1 = x[ x >= xlim ]
    y2 = y[ x <= xlim ]
    x2 = x[ x <= xlim ]

    ybg1 = shirley_loop(x1, y1, maxit = maxit, flag_plot=flag_plot, ax = ax)
    ybg2 = shirley_loop(x2, y2, maxit = maxit, flag_plot=flag_plot, ax = ax)

    if ax == None: ax = plt.gca()
    ybg = np.append(np.append(ybg1[:-1], np.average([ybg1[-1], ybg2[0]])), ybg2[1:] )
    if ybg.shape != x.shape:         # This might happen if xlim is not in x (when the spectra have been shifted f.ex.)
        ybg = np.append(ybg1, ybg2 )

    ax.plot(x, ybg, '--', color=col, label='__nolabel__')
    y12 = y - ybg

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y12})

    if inplace:
        xp.dfx[region] = dfnew
        if store:
            xp.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xp.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        if store:
            xpNew.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xpNew.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xpNew

def subtract_ALShirley_bg(xp : XPS_experiment, region : str, maxit : int = 10,
                          lb : str = '__nolabel__', ax = None,
                          store: bool = True, inplace: bool = False) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values
    col = plot_region(xp = xp, region = region, lb = lb, ax = ax).get_color()

    lmidx, rmidx = find_integration_limits(x, y, flag_plot=True, region = region, ax = ax)
    ybg = shirley_loop(x, y, maxit = maxit)

    yAlsDw = baseline_als(y[:lmidx])
    ybg[:lmidx] = yAlsDw
#     try:
#         yAlsUp = baseline_als(y[rmidx:])
#         ybg[rmidx:] = yAlsUp
#     except ValueError:
#         pass

    if ax == None: ax = plt.gca()
    ax.plot(x, ybg, '--', color=col, label=lb);
    #cosmetics_plot(ax = ax)

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y - ybg})

    if inplace:
        xp.dfx[region] = dfnew
        if store:
            xp.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xp.dfx[region+'_bg', 'counts']  = pd.Series(yAlsDw)
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        if store:
            xpNew.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xpNew.dfx[region+'_bg', 'counts']  = pd.Series(yAlsDw)
        return xpNew


"""        Tougaard        """


def tougaard_loop(x, y, tb=2866, tc=1643, tcd = 1, td=1, maxit=100):
    # https://warwick.ac.uk/fac/sci/physics/research/condensedmatt/surface/people/james_mudd/igor/

    Btou = np.zeros_like(y)

    it = 0
    while it < maxit:
        for i in range(len(y)-1, -1, -1):
            Bint = np.trapz( (y - y[-1]) * (x[i] - x) / ((tc + tcd * (x[i] - x)**2)**2 + td * (x[i] - x)**2),  dx = (x[0]-x[1]))
            Btou[i] = Bint * tb

        Boffset = Btou[0] - (y[0] - y[len(y)-1])
        #print(Boffset, Btou[0], y[0], tb)
        if abs(Boffset) < (1e-7 * Btou[0]) or maxit == 1:
            break
        else:
            tb = tb - (Boffset/Btou[0]) * tb * 0.5
        it += 1

    print("Tougaard B:", tb, ", C:", tc, ", C':", tcd, ", D:", td)

    return y.min() + Btou

def subtract_tougaard_bg(xp : XPS_experiment, region : str, maxit : int = 10,
                        lb : str = '__nolabel__', ax = None,
                        store: bool = True, inplace: bool = False) -> XPS_experiment:
    """Plot region and shirley background. Decorator for shirley_loop function"""
    x, y = xp.dfx[region].dropna().energy.values, xp.dfx[region].dropna().counts.values
    col = plot_region(xp = xp, region = region, lb = lb, ax = ax).get_color()

    ybg = tougaard_loop(x, y, maxit = maxit)

    if ax == None: ax = plt.gca()
    ax.plot(x, ybg, '--', color=col, label=lb);
    ax.legend()

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y - ybg})

    if inplace:
        xp.dfx[region] = dfnew
        if store:
            xp.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xp.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        if store:
            xpNew.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xpNew.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xpNew



"""        Linear and ALS        """


def subtract_linear_bg (xp : XPS_experiment, region, offset:bool = False,
                        lb : str = None, ax = None,
                        store: bool = True, inplace: bool = False) -> XPS_experiment:
    """Fit background to line and subtract from data"""

    from scipy import stats, polyval
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    col = plot_region(xp, region, lb=region, ax=ax).get_color()

    bl = peakutils.baseline(y, deg=1)
    if offset == True:
        yno  = np.where(y < bl)[0]
        offset = np.max(bl[yno] - y[yno])
        bl -= offset
    if ax == None: ax = plt.gca()
    ax.plot(x, bl, '--', color=col, label='Linear Background')

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y - bl})

    if inplace:
        xp.dfx[region] = dfnew
        if store:
            xp.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xp.dfx[region+'_bg', 'counts']  = pd.Series(bl)
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        if store:
            xpNew.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xpNew.dfx[region+'_bg', 'counts']  = pd.Series(bl)
        return xpNew

def baseline_als(y: np.array, lam: float =1e4, p: float = 0.01, niter=30) -> np.array:
    """Asymmetric Least Squares Smoothing algorithm
    Parameters:
        y: in data for baseline search
        lamb: smoothness parameter, 100 ≤ lam ≤ 1e9
        p: asymmetry parameter, 0.001 ≤ p ≤ 0.1

    Returns:
        z: baseline    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def subtract_als_bg (xp : XPS_experiment, region,
                    lb : str = None, ax = None,
                    store: bool = True, inplace: bool = False) -> XPS_experiment:
    """Fit background to asymmetric Least Square Smoothed bg and subtract from data"""

    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    if ax == None: ax = plt.gca()
    col = plot_region(xp, region, lb=region, ax=ax).get_color()

    ybg = baseline_als(y)
    ax.plot(x, ybg, '--', color=col, label='ALS Baseline')

    dfnew = pd.DataFrame({'energy' : x, 'counts' : y - ybg})

    if inplace:
        xp.dfx[region] = dfnew
        if store:
            xp.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xp.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        if store:
            xpNew.dfx[region+'_bg', 'energy'] = pd.Series(x)
            xpNew.dfx[region+'_bg', 'counts']  = pd.Series(ybg)
        return xpNew



def fix_tail_bg(xp: XPS_experiment, region: str, eup: float = None, edw: float = None,
                ax = None, store:bool = True, inplace: bool = False):
    """Subtract ALS of the upper or lower tailing of a peak.
       Parameters:
           eup: upper energy limit
           edw: lower energy limit. Specify either one or the other"""

    xpf = deepcopy(xp)
    if ax == None: ax = plt.gca()

    xpcrop = crop_spectrum(xpf, region, eup=eup, edw=edw)  # Crop the tail to fix bg subtraction
    xpcropBg = subtract_als_bg(xpcrop, region, ax=ax)           # Subtract ALS bg

    dfshort = xpcropBg.dfx[region]                         # Set energy column as index for the whole and the cropped data
    dfshort.set_index('energy', drop=False, inplace=True)
    dfWhole = xpf.dfx[region]
    dfWhole.set_index('energy', drop=False, inplace=True)

    if eup != None:
        dfWhole['counts'].loc[eup:] = dfshort.dropna()['counts'].loc[eup:]   # Update the values of the tail
    elif edw != None:
        dfWhole['counts'].loc[:edw] = dfshort.dropna()['counts'].loc[:edw]   # Update the values of the tail
    else: print('Error: specify upper or lower limit')

    dfWhole.reset_index(drop=True, inplace=True)             # Reset the index and update the dfx in XPS_experiment

    xpf.dfx[region] = dfWhole
    plot_region(xpf, region, ax = ax)
    ax.invert_xaxis()

    if inplace:
        xp.dfx[region] = dfWhole

        if store:
            region += '_bg'
            dfWhole = xp.dfx[region]                # Set energy column as index for the whole and the cropped data
            dfshort = xpcropBg.dfx[region]

            dfWhole.set_index('energy', drop=False, inplace=True)
            dfshort.set_index('energy', drop=False, inplace=True)

            dfWhole.counts.loc[dfshort.index.dropna()] += dfshort.counts.dropna()
            dfWhole.reset_index(drop=True, inplace=True)             # Reset the index and update the dfx in XPS_experiment
            xp.dfx[region] = dfWhole
        return xp

    else: return xpf

def manual_limit_noise(xp: XPS_experiment, region: str, eup: float, edw: float,
                       ydw: float, yup: float, inplace: bool = True):
    """Limit noise within an energy range [eup, edw] and a count range [ydw, yup]"""
    assert edw < eup, "Enter edw < eup"
    assert ydw < yup, "Enter ydw < yup"

    df = xp.dfx[region]
    df.set_index('energy', drop=False, inplace=True)

    rang = df.index[(df.index <= eup) & (df.index >= edw)]

    # from random import uniform
#     eup, edw = 196.2, 192.5
    sim = []
    for i in rang:
        df.counts.loc[i] = uniform(ydw, yup)
    df.index = xp.dfx[region].index

    if inplace: xp.dfx[region] = df
    else:
        xpnew = deepcopy(xp)
        xpnew.dfx[region] = df
    return xpnew

######## Factoring Background functions #########
class XPBackground(object):
    @staticmethod
    def bg_handler(self, xp, region,  *args, **kwargs):
        x = xp.dfx[region].dropna().energy.values
        y = xp.dfx[region].dropna().counts.values
        return x, y

    @staticmethod
    def edit_xp(self, xp, region, x, y, ybg, ax = None):
        if ax == None: ax = plt.gca()
        col = plot_region(xp, region, lb=region, ax=ax).get_color()
        ax.plot(x, ybg, '--', color=col, label='__nolabel__')
        cosmetics_plot(ax=ax)

        dfnew = pd.DataFrame({'energy' : x, 'counts' : y - ybg})
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        return xpNew

    def linear(self, xp, region, *args, **kwargs):
        x, y = self.bg_handler(self, xp, region, *args, **kwargs)

        ybg = peakutils.baseline(y, deg=1)
        return self.edit_xp(xp, region, x, y, ybg)

    def shirley(self, xp, region, maxit=40, **kwargs):
        kwargs['maxit'] = maxit
        x,y = self.bg_handler(self, xp, region, **kwargs)
        ybg = shirley_loop(x,y, **kwargs)
        return self.edit_xp(xp, region, x, y, ybg)

    def doubleShirley(self, xp, region, xlim, maxit=40, **kwargs):
        x, y = self.bg_handler(self, xp, region, **kwargs)
        y1 = y[ x >= xlim ]
        x1 = x[ x >= xlim ]
        y2 = y[ x <= xlim ]
        x2 = x[ x <= xlim ]

        ybg1 = shirley_loop(x1, y1, maxit = maxit)#, flag_plot=flag_plot, ax = ax)
        ybg2 = shirley_loop(x2, y2, maxit = maxit)#, flag_plot=flag_plot, ax = ax)
        ybg = np.append(np.append(ybg1[:-1], np.average([ybg1[-1], ybg2[0]])), ybg2[1:] )
        return self.edit_xp(xp, region, x, y, ybg, **kwargs)

## Usage example: #
# fig, ax = plt.subplots(2)
# bg2 = XPBackground().doubleShirley(xp=trim_exps[0], region='overview_', xlim = 346, maxit=50, ax=ax[0])
# bglin = XPBackground().linear(xp=trim_exps[0], region='overview_', ax=ax[1])

###########################   To use in nb with list of experiments   ###########################

def batch_bg_subtract(experiments : list, regions : list, flag_plot:bool = True, flag_debug:bool = False) -> list:
    """Perform shirley bg subtraction on specified regions from several experiments
    Plot results and store them new list of experiments"""
    bg_exps = deepcopy(experiments)

    fig, ax = plt.subplots(len(regions),2, figsize=(18, 8 * len(regions)))
    for i,r in enumerate(regions):
        for j,xp in enumerate(bg_exps):
            if flag_debug:
                print(xp.name, r)
            try:
                xp = subtract_shirley_bg(xp, r, maxit=80, lb=xp.label, ax=ax[i][0],
                                         offset=500*j, store=True, inplace=True);    # Perform bg subtraction
                plot_region(xp, r, ax=ax[i][1], offset=0*j)

            except AssertionError:
                print('Max iterations exceeded, subtract ALS baseline')
                xp = subtract_als_bg(xp, r, lb=xp.label, ax = ax[i][0],
                                     store=True, inplace=True)
                plot_region(xp, r, ax=ax[i][1], offset=0*j)

            except KeyError as e:
                print('KeyError on ', e)
                continue

        ax[i][0].set_title(r)
        ax[i][1].set_title('Subtraction result')

        ax[i][0].invert_xaxis()
        ax[i][1].invert_xaxis()
        ax[i][1].legend(bbox_to_anchor=(1.05, 0), loc=3)

    fig.tight_layout()
    if not flag_plot: plt.clf(); plt.close()
    return bg_exps

def region_bg_subtract(experiments : list, region = str, flag_plot:bool = True) -> list:
    """Inspect individual shirley bg subtraction for specified region from several experiments
    Plot results and store them new list of experiments"""
    bg_exps = []
    fig, ax = plt.subplots(len(experiments), 2, figsize=(12, 6 * len(experiments)))
    fig.set_size_inches(12, 6*len(experiments))

    for j, xp in enumerate(experiments):
        try:
            xp_bg = subtract_shirley_bg(xp, region, maxit=100, lb='__nolabel__', ax = ax[j][0], offset=500*j)
            plot_region(xp_bg, region, ax=ax[j][1], offset=500*j)
            ax[j][0].set_title(xp.name)
            ax[j][1].set_title('Subtraction result')
            for i in range(2): cosmetics_plot(ax=ax[j][i])
        except AssertionError:
            print('Max iterations exceeded, subtract linear baseline')
            xp_bg = subtract_als_bg(xp, region, lb='__nolabel__', ax = ax[j][0])
        except KeyError as e:
            xp_bg = xp
            print('KeyError in', e)

        bg_exps.append(xp_bg)

    if not flag_plot: plt.clf(); plt.close()
    else: fig.tight_layout()
    return bg_exps

def region_2bg_subtract(experiments : list, region : str, xlim : float, flag_plot : bool = True) -> list:
    """Inspect double shirley bg subtraction for specified region from several experiments
    Plot results and store them new list of experiments"""
    bg_exps = []

    fig, ax = plt.subplots(len(experiments), 2, figsize=(12, 6 * len(experiments)))
    fig.set_size_inches(12, 6*len(experiments))

    for j, xp in enumerate(experiments):
        try:
            xp_bg = subtract_double_shirley(xp, region, xlim=xlim, maxit=100, lb=xp.name, flag_plot=flag_plot, ax = ax[j][0])
            plot_region(xp_bg, region, ax=ax[j][1])
            ax[j][0].set_title(xp.name)
            ax[j][1].set_title('Subtraction result')
            for i in range(2): cosmetics_plot(ax=ax[j][i])
        except AssertionError:
            print('Max iterations exceeded, subtract linear baseline')
            xp_bg = subtract_als_bg(xp, region, lb=xp.name, ax = ax[j][0])
        except KeyError as e:
            xp_bg = xp
            print('KeyError in', e)

        bg_exps.append(xp_bg)

    fig.tight_layout()
    if not flag_plot: plt.clf(); plt.close()
    return bg_exps
