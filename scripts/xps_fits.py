import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
import re

from copy import deepcopy
from scipy.optimize import curve_fit

from dataclasses import dataclass
from xps.xps_import import XPS_experiment
from xps.xps_analysis import plot_region, cosmetics_plot

from lmfit.model import ModelResult
from lmfit.models import PseudoVoigtModel, GaussianModel, VoigtModel

def gauss(x, *p):
    A, mu, sigma = p
    return A *  np.exp(-( x-mu )**2 / (2.*sigma**2))

def double_gauss(x, *p):
    return gauss(x, *p[:3]) + gauss(x, *p[3:])

def compute_p0_peaks(x : np.array, y : np.array, thres0 : float, Npeaks : int) -> list:
    """Rough first estimation of fit parameters p0 from peak search"""

    peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
    while len(peaks) > Npeaks:
        thres0 += 0.05
        peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
    while len(peaks) < Npeaks:
        thres0 -= 0.05
        peaks = peakutils.indexes(y, thres = thres0, min_dist=10)

    p0 = []
    for i in range(Npeaks):
        xmax, ymax = x[peaks[i]], y[peaks[i]]
        p0.append(ymax)
        p0.append(xmax)
        p0.append(x[y > ymax/2][0] - xmax)
    return p0

def fit_double_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.5):
    """Fit to double gauss, estimate loc and scale from peak finding"""

    col = plot_region(xp, region, lb=region).get_color()

    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, Npeaks=2)
    fit, cov = curve_fit(double_gauss, xdata = x , ydata= y, p0=p0)

    plt.plot(x, double_gauss(x, *fit), '--', color=col, label='Double gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    plt.text(s='%.1f'%fit[4], x=fit[4], y=fit[3]*1.05)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)

    return fit[1], fit[4]

def fit_gauss(xp : XPS_experiment, region : str, thres0 : float = 0.7):
    """Fit to gauss, estimate loc and scale from peak finding"""

    col = plot_region(xp, region, lb=region).get_color()
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values

    p0 = compute_p0_peaks(x, y, thres0, 1)
    fit, cov = curve_fit(self.gauss, x, y, p0=p0)

    plt.plot(x, self.gauss(x, *fit), '--', color=col, label='Gauss fit')

    plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
    yl = plt.ylim()
    plt.ylim(yl[0], yl[1]*1.5)
    self.cosmetics_plot()

    return fit

def compute_gauss_area(fit, prefix):
    sigma = fit.best_values[prefix+'sigma']
    amp = fit.best_values[prefix+'amplitude']
    return amp * np.sqrt(np.pi/sigma)

def fit_voigt(xp : XPS_experiment, region : str,  prefix : str = 'v_',
              pars : list = None, bounds : list = None, ax = None, flag_plot: bool = True):
    """General method for fitting voigt model
        Input
        ----------
        xp : class XPS_experiment
            XPS data
        region : str
            core level name
        pars, bounds : list
            initial guess of the fit parameters and bounds. If unspecified, guessed automatically
        Returns
        -----------
        fitv : lmfit.model
            fit result to Voigt model
    """

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts

    mod = PseudoVoigtModel(prefix=prefix)
    if pars == None:
        pars = mod.guess(y, x=x)
        pars[prefix+'sigma'].set(value=1) # Usually guessed wrong anyway
        pars[prefix+'fraction'].set(value=0.2, min=0.15, max=0.20)

    fitv = mod.fit(y, pars, x=x)
    xp.fit.update({region : fitv})

    if flag_plot:
        hatchplot_fit(xp, region, fitv, ax=ax, plot_comps=True)
    return fitv

def add_gauss_shoulder(xp : XPS_experiment, region : str, par_g : list, bounds_g: list,
                       fitv = None, Ng : int = 1, lb : str = None, ax = None, flag_plot : bool = True):
    """
        Add gaussian shoulder to fit
        Input
        ----------
        xp : class XPS_experiment
            XPS data
        region : str
            core level name
        par_g, bounds_g : list
            initial guess of the gauss fit parameters and bounds.
        Returns
        -----------
        fitvg : lmfit.model
            fit result to Voigt + Gaussian model
    """
    from lmfit.models import PseudoVoigtModel, GaussianModel

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts

    gauss2 = GaussianModel(prefix='g'+str(Ng)+'_')
    pars = fitv.params
    pars.update(gauss2.make_params())

    for k,p,b in zip(gauss2.param_names, par_g, bounds_g):
        pars[k].set(value=p, min=b[0], max=b[1])
    mod2 = fitv.model + gauss2

    fitvg = mod2.fit(y, pars, x=x)
    xp.fit.update({region : fitvg})


    if flag_plot:
        hatchplot_fit(xp, region, fitvg, ax=ax, plot_comps=True)
    return fitvg

def fit_double_voigt(xp : XPS_experiment, region : str, pars : list = None, bounds : list = None, sepPt : float = None, frac: float = None,
                     lb : str = None, ax = None, flag_plot : bool = True, plot_comps: bool = False, DEBUG : bool = False):
    """Fitting double voigt model
        Input
        ----------
        xp : class XPS_experiment
            XPS data
        region : str
            core level name
        pars, bounds : list
            initial guess of the fit parameters and bounds. If unspecified, guessed automatically
        sepPt : float
            separation point in energy between the two peaks. If unspecified guessed automatically
        flag_plot, DEBUG : bool
            flags to plot intermediate and final fit results
        Returns
        -----------
        fitv : lmfit.model
            fit result to Voigt model
    """
    from lmfit.models import PseudoVoigtModel

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts
    if sepPt == None: sepPt = find_separation_point(x, y)

    x1 = x[x<sepPt].values
    x2 = x[x>sepPt].values
    y1 = y[x<sepPt].values
    y2 = y[x>sepPt].values

    mod1 = PseudoVoigtModel(prefix='v1_')
    mod2 = PseudoVoigtModel(prefix='v2_')
    if pars == None:
        pars1 = mod1.guess(y1, x=x1)
        pars1['v1_sigma'].set(value=1) # Usually guessed wrong anyway
        pars2 = mod2.guess(y2, x=x2)
        pars2['v2_sigma'].set(value=1) # Usually guessed wrong anyway

    if frac != None:
        pars1['v1_fraction'].set(value=frac, vary=False)
        pars2['v2_fraction'].set(value=frac, vary=False)

    mod = mod1 + mod2
    pars = mod.make_params()
    pars.update(pars1)
    pars.update(pars2)
    if DEBUG:   # Fit & plot individual components separately
        fit1 = mod1.fit(y1, x=x1, params=pars1)
        fit2 = mod2.fit(y2, x=x2, params=pars2)
        plot_fit_result(xp, region, fit1)
        plot_fit_result(xp, region, fit2)

    fitv = mod.fit(y, pars, x=x)
    xp.fit.update({region : fitv})

    if flag_plot:
        hatchplot_fit(xp, region, fitv, ax=ax, plot_comps=True)
    return fitv

def plot_fit_result(xp: XPS_experiment, region: str, fitRes: ModelResult = None,
                    lb : str = None, ax = None, col:str = None, offset: float = 0.3,
                    plot_comps: bool = True, plot_bg:bool = False, flag_fill: bool = False):
    if ax == None : ax = plt.gca()
    if col == None: col = xp.color
    if lb == None: lb = xp.name
    if fitRes == None: fitRes = xp.fit[region]

    x = xp.dfx[region].dropna().energy.values
    ybg = np.zeros_like(x)
    if plot_bg == True:
        try:
            ybg = xp.dfx[region+'_bg'].dropna().counts
            ax.plot(x, ybg , ls='dotted', color=col, lw=2)
        except KeyError as e:
            print('Background not found for ', e)

    offset *= np.average(xp.dfx[region].dropna().counts.values)

    p1 = ax.scatter(x, xp.dfx[region].dropna().counts + ybg + offset,
                    color=col, label=lb, zorder = 1)

    ax.plot(x, fitRes.best_fit + ybg + offset, '-', color=col)#, label='best fit, $\chi^2_N$ = %i' %fitRes.redchi)
#     ax.legend(loc='upper left')

    if plot_comps:
        comps = fitRes.eval_components(x=x)
        for compo in comps:
            posx = fitRes.best_values[compo+'center']
            colc = ax.plot(x, comps[compo] + ybg + offset, ls='dashed', lw=3, color=col, label='__nolabel__')[0].get_color()

            ax.vlines(x=posx, ymin=ybg.min() + offset, ymax=(comps[compo] + ybg + offset).max(), lw=2, linestyle='-', colors=col)
            ax.text(x=posx, y=(comps[compo] + offset + ybg).max()*0.99, s='%.1f' %posx)
            if flag_fill:
                ax.fill_between(x, y1 = ybg, y2 = comps[compo] + ybg, alpha=0.3, color=col)
    ax.invert_xaxis()
    return ax, offset

def hatchplot_fit(xp: XPS_experiment, region: str, fitRes: ModelResult,
                  lb : str = None, marker = 'o', ls: str = 'solid', colc: str = None,
                  ax = None, plot_comps: bool = True, flag_fill: bool = False):

    """"Plot fit result with predefined hatch patterns for each component (up to three components)"""
    if ax == None : ax = plt.gca()
    if lb == None: lb = xp.name
    if colc == None: colc = xp.color

    p1 = ax.scatter(xp.dfx[region].energy, xp.dfx[region].counts, marker=marker, label=lb, zorder = 1)
    p1.set_color(colc)

    x = xp.dfx[region].dropna().energy

    ax.plot(x, fitRes.best_fit, linestyle=ls, color=colc, lw=1.5, label='Fit, $\chi^2_N$ = %i' %fitRes.redchi)

    hatch = ['//', 'ox', '||', '+']


    if plot_comps:
        comps = fitRes.eval_components(x=x)
        for i,compo in enumerate(comps):
            posx = fitRes.best_values[compo+'center']
            ax.text(x=posx, y=comps[compo].max()*1.02, s='%.1f' %posx, fontsize=12)
            ax.fill_between(x, y1 = 0, y2 = comps[compo], alpha=1, label = 'Component @ %.1f eV' %posx ,
                            facecolor='w', hatch = hatch[i], edgecolor=colc, zorder = -1)
    #ax.legend(loc='best')#, bbox_to_anchor=(1.12, 0.5), fontsize=16)
    cosmetics_plot()
    return ax

def find_separation_point(x : np.array, y : np.array, min_dist : int = 20,
                          thres0 :float = 0.5, ax = None, DEBUG : bool = False) -> float:
    """Autolocate separation point between two peaks for double fitting"""
    peaks = [0, 0, 0]
    thres = thres0
    while len(peaks) > 2:
        peaks = peakutils.indexes(y, thres=thres, min_dist=min_dist)
        thres += 0.01
    if DEBUG:
        if ax == None : ax = plt.gca()
        ax.plot(x[peaks], y[peaks], '*', ms=10)
        ax.axvline(x[peaks].sum()/2)
    return x[peaks].sum()/2

def check_pars_amplitud(pars, prefix : str, x : np.array, y : np.array):
    if pars[prefix + 'amplitude'] < 0 :
        amp = y[np.where(x == pars[prefix + 'center'].value)[0][0]]
        pars[prefix + 'amplitude'].set(value=amp)
    return pars

def fit_double_shouldered_voigt(xp : XPS_experiment, region : str, par_g1 : list, bound_g1 : list,
                               par_g2 : list, bound_g2 : list, lb : str = None,
                                ax = None, flag_plot : bool = True) -> tuple:

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts
    ### Separate two peaks ###
    sepPt = find_separation_point(x, y)
    x1 = x[x<sepPt].values
    x2 = x[x>sepPt].values
    y1 = y[x<sepPt].values
    y2 = y[x>sepPt].values

    ### Set and guess main (voigt) components  ####
    vo1 = PseudoVoigtModel(prefix='v1_')
    pars1 = vo1.guess(y1, x=x1)
    vo2 = PseudoVoigtModel(prefix='v2_')
    pars2 = vo2.guess(y2, x=x2)
    pars1['v1_sigma'].set(value=1) # Usually guessed wrong anyway
    pars1['v1_fraction'].set(value=0.15, min=0.15, max=0.2)
    pars2['v2_sigma'].set(value=1) # Usually guessed wrong anyway
    pars2['v2_fraction'].set(value=0.15, min=0.15, max=0.2)

    ### Set gaussian shoulders ###
    gauss1 = GaussianModel(prefix='g1_')
    pars1.update(gauss1.make_params())

    gauss2 = GaussianModel(prefix='g2_')
    pars2.update(gauss2.make_params())
    mod1 = vo1 + gauss1
    mod2 = vo2 + gauss2

    for k,p,b in zip(gauss1.param_names, par_g1, bounds_g1):
        pars1[k].set(value=p, min=b[0], max=b[1])

    for k,p,b in zip(gauss2.param_names, par_g2, bounds_g2):
        pars2[k].set(value=p, min=b[0], max=b[1])

    fitvg1 = mod1.fit(y1, pars1, x=x1)
    fitvg2 = mod2.fit(y2, pars2, x=x2)

    if ax == None: ax = plt.gca()
    col = plot_region(xp, region, lb, ax).get_color()
    comps1 = fitvg1.eval_components(x=x1)
    ax.plot(x1, fitvg1.best_fit, '-r', label = 'best fit, $\chi^2_N$ = %i' %fitvg1.redchi)
    for compo in comps1:
        colc = ax.plot(x1, comps1[compo], ls='dashdot', label = '%scenter: %.2f' %(compo, fitvg1.best_values[compo+'center']) )[0].get_color()
        ax.fill_between(x1, y1 = 0, y2 = comps1[compo], alpha=0.3, color=colc)

    comps2 = fitvg2.eval_components(x=x2)
    ax.plot(x2, fitvg2.best_fit, '-', label = 'best fit, $\chi^2_N$ = %i' %fitvg2.redchi)
    for compo in comps2:
        colc = ax.plot(x2, comps2[compo], ls='dashdot', label = '%scenter: %.2f' %(compo, fitvg2.best_values[compo+'center']) )[0].get_color()
        ax.fill_between(x2, y1 = 0, y2 = comps2[compo], alpha=0.3, color=colc)

    ax.legend()
    return fitvg1, fitvg2

################################################################
########           N-voigt fitting functions            ########
################################################################
def guess_pars(mod, x, y, prefix):
    """Guesses position of component and sets sensible values for amplitude & fraction"""
    pars = mod.make_params()
    guess = mod.guess(y, x)

    pars[prefix+'center'].set(value=guess[prefix+'center'].value, min=x[-1], max=x[0])
    pars[prefix+'amplitude'].set(value=abs(guess[prefix+'amplitude'].value), min=0)
    pars[prefix+'sigma'].set(value=guess[prefix+'sigma'].value, min=0.2, max=2.)

    #pars[prefix+'fraction'].set(value=0.7, vary=False)
    return pars

def split_component(xs:list, ys:list, largest_comp:int):
    """Splits the component in position xs[largest_comp] into 2 energy arrays xl, xr
    and two count arrays yl, ys and append them to array of xs and ys """
    sepPt = np.average(xs[largest_comp])
    xsplit = xs[largest_comp]
    ysplit = ys[largest_comp]
    xl = xsplit[xsplit<sepPt]
    xr = xsplit[xsplit>sepPt]
    yl = ysplit[xsplit<sepPt]
    yr = ysplit[xsplit>sepPt]

    xs.pop(largest_comp)
    xs += [xl, xr]
    ys.pop(largest_comp)
    ys += [yl, yr]
    return xs, ys

def set_mod_pars(ncomps: int, xs: list, ys: list):
    """Sets composite model for a number of components ncomps and calls guess_pars for each comp"""
    mods = [PseudoVoigtModel(prefix='v'+str(i)+'_') for i in range(ncomps)]
    parsi = []
    mod = mods[0]

    for i,(m, xx, yy) in enumerate(zip(mods, xs, ys)):
        parsi.append(guess_pars(m, xx, yy, prefix='v'+str(i)+'_'))
        if i > 0: mod += m

    pars = mod.make_params()
    for i,p in enumerate(parsi):
        pars.update(p)
    return mod, pars

def fit_n_voigt(xp: XPS_experiment, region: str, max_comps: int=3, flag_save: bool = False):
    """Fits up to max_comps voigt components and returns reduced chi2 for each fit
    Args:
        max_comps: int, maximum number of components to look for
        flag_save: bool, if True, stores each fit (as many as max_comps) in xp.fit[region_i_comps]
    """
    chis = []
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    xs = [x]
    ys = [y]
    fig, ax = plt.subplots(1, max_comps, figsize=(6*max_comps, 6))

    for ncomps in range( max_comps):
        if ncomps > 0:
            sigmas = np.array([fitv.best_values['v'+str(i)+'_sigma'] for i in range(ncomps)])

            largest_comp = np.argmax(sigmas)
            xs, ys = split_component(xs, ys, largest_comp)

        mod, pars = set_mod_pars(ncomps+1, xs, ys)

        fitv = mod.fit(y, pars, x=x)
        plot_fit_result(xp, region, fitv, ax=ax[ncomps])
#         ax[ncomps].invert_xaxis()
        chis.append(fitv.redchi)
        if flag_save: xp.fit[region+'_'+str(ncomps+1)+'comps'] = fitv
    print('Best chi2 value for %i components' %(np.argmin(chis)+1))
    return chis

def subtract_fit_component(xp: XPS_experiment, region:str, prefix: str, fitRes = None,
                           flag_plot:bool = True, store:bool = True, inplace:bool = False):
    """Subtract a fit component from the data, prefix specifies which component
        Returns XPS_experiment: with the subtraction result
        if inplace, the store flag is assumed True too"""
    x = xp.dfx[region].dropna().energy.values
    y = xp.dfx[region].dropna().counts.values
    if fitRes == None:  fitRes = xp.fit[region]

    comps = fitRes.eval_components(x=x)
    yv1 = comps[prefix]
    yb1 = y - yv1
    if flag_plot:
        fig, ax = plt.subplots(1, 2, sharex=True, figsize=(12, 6))
        plot_fit_result(xp, region, ax=ax[0], fitRes=fitRes)
        ax[0].set(title=xp.name+ ' original fit')
        lim = ax[0].get_ylim()
        ax[1].plot(x, yb1, color=xp.color)
        ax[1].set(title='v1 subtraction', yticks=[], ylim=lim)

    xpNew = deepcopy(xp)
    xpNew.dfx[region, 'energy'] = pd.Series(x)
    xpNew.dfx[region, 'counts']  = pd.Series(yb1)

    if store:
        xpNew.dfx[region+'_'+prefix+'bg', 'energy'] = pd.Series(x)
        xpNew.dfx[region+'_'+prefix+'bg', 'counts']  = pd.Series(yv1)

    if inplace:
        insert_dfx_region(xp, xpNew, region=region, inplace=True)
        insert_dfx_region(xp, xpNew, region=region+'_'+prefix+'bg', inplace=True)

    else:
        return xpNew

################################################################
########           Refactor into fit class              ########
################################################################


""" Usage example:
fig, ax = plt.subplots(1, 3, figsize=(30, 12))
for i,xp in enumerate(fbiBa_exps):
    Fn = XPFit(xp, region='N_1s')
    Fn.voigt()
    Fn.plot( ax = ax[i])
"""

class XPFit(object):
    def __init__(self, xp: XPS_experiment, region: str):
        self.xp = xp
        self.region = region
        self.x = self.xp.dfx[self.region].dropna().energy.values
        self.y = self.xp.dfx[self.region].dropna().counts.values
        self.userPars = {}

    def preset_pars(self, key: str, val: str):
        self.userPars.update({key: val})

    def set_userPars(self, pars):
        for key, val in zip(self.userPars.keys(), self.userPars.values()):
            pars[key].set(value=val, vary=False)
        return pars

    @staticmethod
    def guess_pars(self, mod, x, y, prefix):
        pars = mod.make_params()
        guess = mod.guess(y, x)

        pars[prefix+'center'].set(value=guess[prefix+'center'].value)
        pars[prefix+'amplitude'].set(value=abs(guess[prefix+'amplitude'].value), min=0)
        #pars[prefix+'fraction'].set(value=0.7, vary=False)
        return pars

    @staticmethod
    def finish_fit(self, mod, pars):
        if self.userPars != {}:
            print('Modify user pars')
            pars = self.set_userPars(pars)

        fitv = mod.fit(self.y, pars, x=self.x)
        self.xp.fit.update({self.region: fitv})
        return fitv

    def plot(self, ax = None, plot_bg: bool = False):
        if ax == None: ax = plt.gca()
        plot_fit_result(self.xp, self.region, fitRes = self.xp.fit[self.region],
                      ax=ax, plot_comps=True, plot_bg = plot_bg)

    def set_areas(self):
        fit = self.xp.fit[self.region]
        dx = self.x[0] - self.x[1]
        areas, rel_areas = {}, {}
        for key, val in zip(fit.eval_components().keys(), fit.eval_components().values()):
            areas.update({key : np.trapz(val, dx = dx)})
        for key, val in zip(areas.keys(), areas.values()):
            self.xp.area.update({self.region: sum(areas.values() )})
            self.xp.area.update({self.region+'_'+key : val/sum(areas.values())})

    """Model options"""

    def pvoigt(self, pars: list = None):
        mod = PseudoVoigtModel(prefix='p1_')
        if pars == None:
            pars = self.guess_pars(self, mod, self.x, self.y, prefix='p1_')

        return self.finish_fit(self, mod, pars)

    def voigt(self, pars: list = None):
        mod = VoigtModel(prefix='v1_')
        if pars == None:
            pars = self.guess_pars(self, mod, self.x, self.y, prefix='v1_')

        return self.finish_fit(self, mod, pars)

    def double_voigt(self, sepPt = None, pars: list = None, bounds: list = None):
        if sepPt == None: sepPt = find_separation_point(self.x, self.y)

        x1 = self.x[self.x<sepPt]
        x2 = self.x[self.x>sepPt]
        y1 = self.y[self.x<sepPt]
        y2 = self.y[self.x>sepPt]

        mod1 = VoigtModel(prefix='v1_')
        mod2 = VoigtModel(prefix='v2_')

        if pars == None:
            pars1 = self.guess_pars(self, mod1, x1, y1, prefix='v1_')
            pars2 = self.guess_pars(self, mod2, x2, y2, prefix='v2_')

        mod = mod1 + mod2
        pars = mod.make_params()
        pars.update(pars1)
        pars.update(pars2)

        return self.finish_fit(self, mod, pars)

    def double_pvoigt(self, sepPt = None, pars: list = None, bounds: list = None):
        if sepPt == None: sepPt = find_separation_point(self.x, self.y)

        x1 = self.x[self.x<sepPt]
        x2 = self.x[self.x>sepPt]
        y1 = self.y[self.x<sepPt]
        y2 = self.y[self.x>sepPt]

        mod1 = PseudoVoigtModel(prefix='v1_')
        mod2 = PseudoVoigtModel(prefix='v2_')

        if pars == None:
            pars1 = self.guess_pars(self, mod1, x1, y1, prefix='v1_')
            pars2 = self.guess_pars(self, mod2, x2, y2, prefix='v2_')

        mod = mod1 + mod2
        pars = mod.make_params()
        pars.update(pars1)
        pars.update(pars2)

        return self.finish_fit(self, mod, pars)

    def gauss_shoulder(self, fitv, par_g: list, bounds_g: list, Ng: int = 1):
        fitv = self.xp.fit[self.region]
        last_prefix = list(fitv.eval_components().keys())[-1]
        current_prefix = 'g' + str(int(last_prefix[1]) + 1) + '_'

        gauss2 = GaussianModel(prefix=current_prefix)

        pars = fitv.params
        pars.update(gauss2.make_params())

        if bounds_g == None: # Set (amplitude, center, sigma, gamma) bounds automatically
            bounds_g = [(0.8*par_g[0], 1.2*par_g[0]), (par_g[1]-0.2, par_g[1]+0.2),
                        (0.8*par_g[2], 1.2*par_g[2])]

        for k,p,b in zip(gauss2.param_names, par_g, bounds_g):
            pars[k].set(value=p, min=b[0], max=b[1])
        mod = fitv.model + gauss2

        return self.finish_fit(self, mod, pars)

    def add_voigt(self, par_v: list, bounds_v: list = None):
        fitv = self.xp.fit[self.region]
        last_prefix = list(fitv.eval_components().keys())[-1]
        current_prefix = 'v' + str(int(last_prefix[1]) + 1) + '_'

        voigtN = VoigtModel(prefix=current_prefix)

        pars = fitv.params
        pars.update(voigtN.make_params())

        if bounds_v == None: # Set (amplitude, center, sigma, gamma) bounds automatically
            bounds_v = [(0.8*par_v[0], 1.2*par_v[0]), (par_v[1]-0.5, par_v[1]+0.5),
                        (0.8*par_v[2], 1.2*par_v[2])]
            try:
                bounds_v.append( (par_v[3]-0.05, par_v[3]+0.05) )
            except IndexError:
                pass

        for k,p,b in zip(voigtN.param_names, par_v, bounds_v):
            pars[k].set(value=p, min=b[0], max=b[1])
        mod = fitv.model + voigtN

        return self.finish_fit(self, mod, pars)
