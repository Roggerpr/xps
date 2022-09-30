import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
import re

from copy import deepcopy
from scipy.optimize import curve_fit
from IPython.display import HTML, display

from dataclasses import dataclass
from xps.xps_import import XPS_experiment
#from xps.xps_fits import XPFit
#from invisible_cities.core.system_of_units import nm

nm = 1e-6

def plot_region(xp : XPS_experiment, region : str, col: str = None, lb : str = None, ax = None, offset: float = 0.):
    """Quick region plotter"""
    if lb == None: lb = xp.name
    if ax == None: ax = plt.gca()
#     offset *= np.average(xp.dfx[region].dropna().counts.values)
    p1 = ax.plot(xp.dfx[region].energy, xp.dfx[region].counts + offset, label=lb)

    if col != None:
        p1[0].set(color=col)
    elif xp.color :
        p1[0].set(color=xp.color)

    if xp.ls:
        p1[0].set(linestyle=xp.ls)

    cosmetics_plot()
    return p1[0]

def cosmetics_plot(ax = None, leg : bool = True):
    if ax == None: ax = plt.gca()
    ax.invert_xaxis()
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('CPS [A.U.]')
    if leg: ax.legend()

def scatter_region(xp : XPS_experiment, region : str, col: str = None, lb : str = None, ax = None, offset: float = 0.):
    """Quick region plotter"""
    if lb == None: lb = xp.name
    if ax == None: ax = plt.gca()
#     offset *= np.average(xp.dfx[region].dropna().counts.values)
    p1 = ax.plot(xp.dfx[region].energy, xp.dfx[region].counts + offset, 'o', label=lb)

    if col != None:
        p1[0].set(color=col)
    elif xp.color :
        p1[0].set(color=xp.color)

    if xp.ls:
        p1[0].set(linestyle=xp.ls)

    cosmetics_plot()
    return p1[0]

def trim_spectra(xp : XPS_experiment, xpRef : XPS_experiment, region, inplace : bool = False) -> XPS_experiment:
    """Crop spectra with different bounds so that they coincide with xpRef
    xpRef should have the shortest spectrum on both ends"""
    eup = xpRef.dfx[region].energy.head(1)
    edw = xpRef.dfx[region].dropna().energy.tail(1)

    cutup = np.where(xp.dfx[region].energy.values > eup.values)[0]
    cutdown = np.where(xp.dfx[region].dropna().energy.values < edw.values)[0]

    dfnew = xp.dfx[region].dropna().drop(cutup).drop(cutdown).reset_index(drop=True)

    if inplace:
        xp.dfx[region] = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
    return xpNew

def crop_spectrum(xp : XPS_experiment, region : str,
                  eup : float = None, edw : float = None, inplace : bool = False):
    """Manually bound region spectrum to upper (eup) and lower (edw) limits """
    if eup == None: eup = xp.dfx[region].energy.head(1).values[0]
    if edw == None: edw = xp.dfx[region].dropna().energy.tail(1).values[0]

    dropup = np.where(xp.dfx[region].energy.values > eup)[0]
    dfup = xp.dfx[region].dropna().drop(dropup).reset_index(drop=True)

    dropdw = np.where(dfup.energy.values < edw)[0]
    dfnew = dfup.drop(dropdw).reset_index(drop=True)

    if inplace:
        xp.dfx[region] = dfnew
        return xp
    else:
        xpNew = deepcopy(xp)
        xpNew.dfx[region] = dfnew
        return xpNew

def batch_trimming(experiments: list, regions: list,
                flag_plot: bool = True)->list:
    """Loop over specified regions, locate the experiment with shortest ends
        and trim all spectra to meet those bounds"""

    ### First find shortest spectra on both ends:
    fig, ax = plt.subplots(len(regions), figsize=(8, 8*len(regions)))
    trimmed_exps = deepcopy(experiments)

    for i,r in enumerate(regions):
        rng = []
        for j,xp in enumerate(trimmed_exps):
            try:
                rng.append(len(xp.dfx[r].dropna().energy.values))
            except KeyError: rng.append(10000)

        shorty = np.argmin(rng)
        for j,xp in enumerate(trimmed_exps):
            try:
                trim_spectra(xp, xpRef=trimmed_exps[shorty], region=r, inplace=True)
            except KeyError: pass

        if flag_plot:
            plot_exps(trimmed_exps, r, ax=ax[i])

    return trimmed_exps

def gaussian_smooth(xp : XPS_experiment, region, sigma : int = 2) -> XPS_experiment:
    from scipy.ndimage.filters import gaussian_filter1d

    y = gaussian_filter1d(xp.dfx[region].dropna().counts.values, sigma = sigma)
    dfnew = pd.DataFrame({'energy' : xp.dfx[region].energy.dropna(), 'counts' : y})

    xpNew = deepcopy(xp)
    xpNew.dfx[region] = dfnew
    return xpNew

def flexible_integration_limits(xp : XPS_experiment, region : str, doublePeak : float = 0, ax=None, flag_plot : bool = True) -> list:
    """Autolocate limits for area integration.
    doublePeak > 0 : second peak on the rhs of the main one
    doublePeak < 0 : second peak on the lhs of the main one
    doublePeak == 0 : no second peak
    Returns:
    --------
    [maxidx, (maxidx2), lmidx(2), rmidx(2)]
    Position index of [maxima, (second max), left minima, right minima]
    (2) denotes from secondary peak"""

    x = xp.dfx[region].dropna().energy
    y = xp.dfx[region].dropna().counts

    maxidx = abs(y - np.max(y)).idxmin()
    lmidx = abs(y[0:maxidx] - np.min(y[0:maxidx])).idxmin()
    rmidx = abs(y[maxidx:] - np.min(y[maxidx:])).idxmin() #+ maxidx

    if doublePeak < 0:
        maxidx2 = abs(y[:lmidx] - np.max(y[:lmidx])).idxmin()
        lmidx2 = abs(y[:maxidx2] - np.min(y[:maxidx2])).idxmin()
        ind = [maxidx, maxidx2, lmidx2, rmidx]
    elif doublePeak > 0:
        maxidx2 = abs(y[rmidx:] - np.max(y[rmidx:])).idxmin()
        rmidx2 = abs(y[maxidx2:] - np.min(y[maxidx2:])).idxmin()
        ind = [maxidx, maxidx2, lmidx, rmidx2]
    else:
        ind = [maxidx, lmidx, rmidx]

    if  flag_plot:
        if ax == None: ax = plt.gca()
        ax.plot(x, y, label=xp.name)
        ybase = ax.get_ylim()[0]
        for i in ind:
            ax.vlines(x[i], ymin=ybase, ymax=y[i], linestyles='--')
            ax.text(s='%.2f'%x[i], x = x[i], y = y[i])
        cosmetics_plot()
    return ind

def compare_areas(xp_ref : XPS_experiment, xp_sg : XPS_experiment, region : str,
                  lmidx : int, rmidx : int, lb : str = None,  ax = None, flag_fill : bool = False):
    """Returns absolute and relative area in a region xp_sg and w.r.t. xp_ref
    between indices lmidx and rmidx"""
    y_ref = xp_ref.dfx[region].dropna().counts
    y_sg = xp_sg.dfx[region].dropna().counts

    if ax == None: ax = plt.gca()
    x = xp_sg.dfx[region].dropna().energy
    step = x[0] - x[1]

    area = np.trapz(y_sg [ lmidx : rmidx ], dx = step)
    area_rel = area / np.trapz(y_ref [ lmidx : rmidx ], dx = step)

    if lb == None: lb = xp_sg.name
    ax.plot(x, y_sg, '-', label=lb)
    if flag_fill:
        ax.fill_between(x [lmidx : rmidx], y1 = y_sg[lmidx], y2 = y_sg [lmidx : rmidx], alpha=0.3)

    cosmetics_plot()
    return area_rel, area

def inset_rel_areas(area_rel : list, names : list) -> None:
    ax = plt.gca()
    axins = plt.axes([0.65, 0.5, 0.25, 0.3])
    axins.bar(names, area_rel)
    axins.set_ylabel('$A_{exp}/A_{ref}$', fontsize=12)
    axins.tick_params(labelrotation=45)
    ax.legend(loc='upper left')

def guess_xpRef(exps:list , region: str):
    """Guess the best resolved spectrum  in the list as the highest peak in the region
    (use after pre-processing)"""
    ymx = []
    for xp in exps:
        try:
            ymx.append(xp.dfx[region].counts.max())
        except KeyError:
            ymx.append(0)

    yarr = np.array(ymx)
    yarr.sort()
    yarr = yarr[yarr > 0]
    # Check if the maximum is a noisy peak ( 2σ over the aveage )

    if yarr[-1] > np.average(yarr) + 2*np.std(yarr):
        print('Noisy peak')
        return(np.where(ymx == yarr[-2])[0][0])
    else:
        return np.argmax(np.array(ymx))


def search_dnames(name: str,
                  dnames: dict = None, experiments: list = None):
    if dnames == None:
        try:
            dnames = {i: xp.name for i, xp in enumerate(experiments)}
            return list(dnames.values()).index(name)
        except TypeError: print("Error: Specify list of experiments")
    else:
        return list(dnames.values()).index(name)

########################################################
############ Integration area functions ################
########################################################

def asfdb_read(source: str = ['Al', 'Mg']):
    path = '/Users/pabloherrero/sabat/sabatsw/xps/Scofield.csv'
    asfdb = pd.read_csv(path, sep=';')

    asfdb.index = asfdb['AtomicLevel.symbol'] + asfdb['AtomicLevel.level']
    asfdb.SensitivityFactor = asfdb.SensitivityFactor.apply(lambda x: float(x.replace(' Mb', '')))

    asfMg = asfdb.where(asfdb['ExcitationEnergy'] == '1254 eV').dropna(axis=0, how='all')
    asfAl = asfdb.where(asfdb['ExcitationEnergy'] == '1487 eV').dropna(axis=0, how='all')
    return asfAl if source == 'Mg' else asfAl

def search_asf(region):
    asdf = asfdb_read('Al')

    # Filter symbols in region
    regionf = region.replace('_', '')
    m = re.search(r"\([0-9]\)", regionf)
    try:
        regionf = regionf.replace(m.group(), '')
    except AttributeError: pass

    try:
        val_Scofield = asdf.loc[regionf].SensitivityFactor
        val_powTanum = asdf.loc[regionf].SensitivityFactor / asdf.loc['F1s'].SensitivityFactor
        return {region : val_powTanum}
    except KeyError:
        print(regionf, " region not found in database")

def integrateRegions(exps: list, region : str,  asf: dict, indRef: int= None,
                     eup: float = None, edw: float = None,
                     lb : str = None, flag_fill : bool = True):
    """Integrate peaks for a list of experiments between two minima
       The minima are automatically located for exps[indRef] unless they are specified by eup and edw
    The boundary are fixed for the whole list."""

    if indRef == None: indRef = guess_xpRef(exps, region)

    xRef = exps[indRef].dfx[region].dropna().energy     # Use the energy array of reference xp to crop the other xp's

    if eup == None or edw == None:
        ind = flexible_integration_limits(exps[indRef], region=region, doublePeak=0, flag_plot=False)
        lmidx, rmidx = ind[-2:] # The index of the minima are always the last two
        eup, edw = xRef[lmidx], xRef[rmidx]

    # Distribute the experiments in the plots in sets of 10
    nrows = len(exps) // 10 + int((len(exps) % 10) != 0)
    fig, ax = plt.subplots(nrows, 10, figsize=(10*5, nrows*5) )
    area = []
    for i, xp in enumerate(exps):
        l, m = i%10, i//10
        try:
            y = xp.dfx[region].dropna().counts
        except KeyError as e:          #Check the region exists in this xp
            print(e, 'region does not exist in ' + xp.name)
            xp.area.update({region: 1e-10})
            continue

        x = xp.dfx[region].dropna().energy
        ax[m,l].plot(x, y, label=xp.name)

        xpCrop = crop_spectrum(xp, region, eup = eup, edw = edw)
        yc = xpCrop.dfx[region].dropna().counts.values
        xc = xpCrop.dfx[region].dropna().energy.values    # Integrate only in the cropped range

        step = x[0] - x[1]
        area.append(np.trapz(yc, dx=step))

        if asf == None:
            asf = search_asf(region)
        try:
            xp.area.update({region : area[-1]/asf[region]})
        except (KeyError, NameError) as e:
            print(e, ', asf missing, returning raw area')
            pass

        #### Plotting the operation

        if flag_fill:
            if yc[0] > yc[-1]:
                ax[m,l].fill_between(xc , y1 = yc[-1], y2 = yc, alpha=0.3)
            else:
                ax[m,l].fill_between(xc, y1 = yc[0], y2 = yc, alpha=0.3)
            ybase = ax[m,l].get_ylim()[0]

            for j in [0, -1]:
                ax[m,l].vlines(xc[j], ymin=ybase, ymax=yc[j], linestyles='--')
                ax[m,l].text(s='%.2f'%xc[j], x = xc[j], y = yc[j])
        cosmetics_plot(ax=ax[m,l])
    plt.tight_layout()
    fig.suptitle(region)
    return area

def integratePeak(xp: XPS_experiment, region: str, asf: dict, nsigma: int = 4,
                  fitm: str = ['v', 'dv'], sepPt: float = None, flag_fill: bool = True):
    Fn = XPFit(xp, region)
    if fitm == 'dv':
        fitv = Fn.double_voigt(sepPt)
        edw = fitv.best_values['v1_center'] - nsigma*fitv.best_values['v1_sigma']
        eup = fitv.best_values['v2_center'] + nsigma*fitv.best_values['v2_sigma']

    else:
        fitv = Fn.voigt()
        edw = fitv.best_values['v1_center'] - nsigma*fitv.best_values['v1_sigma']
        eup = fitv.best_values['v1_center'] + nsigma*fitv.best_values['v1_sigma']

    Fn.plot()
    ax = plt.gca()
    xpCrop = crop_spectrum(xp, region, eup = eup, edw = edw)
    yc = xpCrop.dfx[region].dropna().counts.values
    xc = xpCrop.dfx[region].dropna().energy.values    # Integrate only in the cropped range

    area = np.trapz(yc, dx= xc[0] - xc[1])

    try:
        xp.area.update({region : area/asf[region]})
    except (KeyError, NameError) as e:
        print(e, ', asf missing, storing raw area')
        xp.area.update({region : area})

    #### Plotting the operation

    if flag_fill:
        if yc[0] > yc[-1]:
            ax.fill_between(xc , y1 = yc[-1], y2 = yc, color='b', alpha=0.3)
        else:
            ax.fill_between(xc, y1 = yc[0], y2 = yc, color='b', alpha=0.3)
        ybase = ax.get_ylim()[0]
        for j in [0, -1]:
            ax.vlines(xc[j], ymin=ybase, ymax=yc[j], linestyles='--')
            ax.text(s='%.2f'%xc[j], x = xc[j], y = yc[j])

################################################################
########           Stoichiometry functions              ########
################################################################

def display_table(data):
    html = "<table>"
    for row in data:
        html += "<tr>"
        for field in row:
            html += "<td><h4>%s</h4><td>"%(field)
        html += "</tr>"
    html += "</table>"
    display(HTML(html))

def display_stoichiometry(exps: list, num: list, denom: list):
    """Utility for proper HTML display of table
        Use make_stoichometry_table to export data"""
    head = ['Experiment']   # Make header
    for n, d in zip(num, denom):
        try:
            cln = re.search(r'\d+', n).span()[0]
            cld = re.search(r'\d+', d).span()[0]
        except AttributeError:
            (cln, cld) = (-1,-1)
        head.append(n[:cln] + '/' + d[:cld])
    data = [head]

    for k, xp in enumerate(exps):
        row = [xp.name]
        for i, j in zip (num, denom):
            try:
                row.append('%.2f' %(xp.area[i]/xp.area[j]) )
            except KeyError: print(i, ' or ', j, ' was not measured')
        data.append(row)
    display_table(data)

def make_header(num : list, denom : list):
    head = 'Experiment\t'
    for n, d in zip(num, denom):
        try:
            cln = re.search(r'\d+', n).span()[0]
            cld = re.search(r'\d+', d).span()[0]
        except AttributeError:
            cln = -1
            cld = -1
        head += n[:cln] + '/' + d[:cld] + '\t'
    print(head)

def regionPairs2numDenom(pairs : tuple):
    """Reshape a tuple of region pairs (ex. N/C, Br/O)
    into (num, denom) tuple (ex. ('N1s', 'C1s'), ('Br3p', 'O1s'))"""
    transpose = np.array(pairs).T
    assert transpose.shape[0] == 2, "Passed tuple has incorrect shape"
    num, denom = transpose
    return num, denom

def make_stoichometry_table(exps : list, num : list, denom : list, sep='\t'):
    """Print stoichiometry table of the experiments exps at the regions in num/denom
    Example: make_stoichometry_table(oxid_exps, 'N1s', 'C1s'], ['Br3p', 'O1s'])
    will print the stoichiometry N/C, Br/O for the passed experiments"""

    make_header(num = num, denom = denom)
#     print('Experiment, ' + )
    for k, xp in enumerate(exps):
        row = xp.name + sep
        for i, j in zip (num, denom):
            try:
                row += ('%.2f %s ' %(xp.area[i]/xp.area[j], sep))
            except KeyError: print(i, ' or ', j, ' was not measured')

        print(row )

def make_stodev_table(exps : list, num : list, denom : list, nominal: list):
    """Print table of stoichiometry deviation from nominal values.
    Compute for the experiments exps at the regions in num/denom
    Example: make_stodev_table(oxid_exps, ['N1s', 'C1s'], ['Br3p', 'O1s'], [2, 6.2])
    will print the stoichiometry deviations N/C, Br/O for the passed experiments
    and the reference nominal values [2, 6.2]"""

    make_header(num = num, denom = denom)
    for k, xp in enumerate(exps):
        tot_dev = 0
        row = xp.name + '\t'
        for i, j, m in zip (num, denom, nominal):
            area = xp.area[i]/xp.area[j]
            dev = (m - area)/m
            tot_dev += dev**2
            row += ('%.2f\t ' %dev)
        row += ('%.2f' %np.sqrt(tot_dev))
        print(row )

def component_areas(fit, x : np.array = None) -> tuple:
    """Numerical integration of area components of lmfit.result
    Arguments:
    fit : lmfit.Model.result
    x : np.array
        Independendt variable, if unspecified, dx = 1 for numerical integration
    Returns:
    rel_areas : dict
        Component areas normalized to total sum
    areas : dict
        keys are component prefixes and values the integrated areas"""
    if x == None: dx = 1
    else: dx = x[0] - x[1]

    areas, rel_areas = {}, {}
    for key, val in zip(fit.eval_components().keys(), fit.eval_components().values()):
        areas.update({key+'area' : np.trapz(val, dx = dx)})
    for key, val in zip(areas.keys(), areas.values()):
        rel_areas.update({key : val/sum(areas.values())})

    return rel_areas, areas


def table_fit_area(exps: list, region: str):
    """Print a table with fit results and relative areas dict"""
    par_table = ['center', 'fwhm', 'amplitude', 'area']
    head = 'comp\t'

    for par in par_table:
        head += '%s\t'%par

    print(head)

    for xp in exps:
        fit = xp.fit[region]
        for i, comp in enumerate(fit.components):
            pref = comp.prefix
            line = pref[:-1] + '\t'

            for par in par_table[:-1]:
                line += '%.2f\t '%fit.values[pref + par]
            line += '%.2f'%xp.area[region+'_'+pref]

            print(line)

def barplot_fit_fwhm(experiments : list, fit : np.array):
    names = [xp.name for xp in experiments]

    colv = plt.errorbar(x = fit[:,0], y = names, xerr=fit[:,1]/2, fmt='o', mew=2, label='Main component')[0].get_color()

    dif = fit [:-1,0] - fit[1:,0]
    for i, d in enumerate(dif) :
        plt.annotate(s = '$\Delta E = $%.2f'%d, xy=(fit[i+1,0], 0.8 * (i+1)), color=colv)
        plt.fill_betweenx(y=(i, i+1), x1=fit[i,0], x2=fit[i+1,0], alpha=0.3, color=colv)

    if fit.shape[1] > 2:
        colg = plt.errorbar(x = fit[:,2], y = names, xerr=fit[:,3]/2, fmt='o', mew=2,label='Shoulder')[0].get_color()
        difg = fit [:-1,2] - fit[1:,2]
        for i, d in enumerate(difg) :
            plt.annotate(s = '$\Delta E = $%.2f'%d, xy=(fit[i+1,2], 0.8 * (i+1)), color=colg)
            plt.fill_betweenx(y=(i, i+1), x1=fit[i,2], x2=fit[i+1,2], alpha=0.3, color=colg)
    cosmetics_plot()
    plt.ylabel('')


################################################################
########           Batch plotting functions             ########
################################################################
def plot_exps(exps: list, region: str, ax = None, off = 0):
    if ax == None: ax = plt.gca()
    lines = []
    for xp in exps:
        try:
            li = plot_region(xp, region, offset=off, ax=ax)
            lines.append(li)
        except KeyError:
            pass
    ax.legend(bbox_to_anchor=(1.05, 0), loc=3)
    if len(lines)%2==0: ax.invert_xaxis()

def plot_xp_regions(experiments : list, regions : list, colors : list = [], ncols: int = 3, flag_shift: bool = False):
    """Subplots all regions of a list of experiments (unnormalised)"""
    rows = int(np.ceil(len(regions) / ncols))

    fig, ax = plt.subplots(rows, ncols, figsize=(16, 8))
    fig.add_subplot(111, frameon=False, xticks=[], yticks=[])  # Used for common xlabel and ylabel

    for i,r in enumerate(regions):
        enmx, comx = [], [] # Peak point lists
        for c,xp in enumerate(experiments):
            j, k = i//ncols, i%ncols

            if i == len(regions) - 1:   # Set labels from last region
                lb = xp.name

            else:
                lb='__nolabel__'

            try:
                li = plot_region(xp, r, ax=ax[j][k], lb=lb)
                ax[j][k].invert_xaxis()
            except KeyError:    # Auto-exclude regions not recorded for a particular experiment
                pass
            if flag_shift:
                argmx = np.argmax(xp.dfx[r].counts)
                enmx.append(xp.dfx[r].energy.loc[argmx])
                comx.append(xp.dfx[r].counts.loc[argmx])

        ax[j][k].text(s=r.replace('_', ' '), y=0.9, x=0.1, transform=ax[j][k].transAxes)
        ax[j][k].set_yticks([])
        if flag_shift:  ax[j][k].plot(enmx, comx, '--k', lw=2.5)

        if len(experiments)%2 == 0:
            ax[j][k].invert_xaxis()
    ax[j][k].legend().remove()

    plt.xlabel('\n\nEnergy [eV]', ha='center')
    plt.figlegend(  ncol=1, bbox_to_anchor=(1.1, 0.45), framealpha=1., labelspacing=0.8 )
    plt.tight_layout(w_pad=0.5, h_pad=0.5, pad=0.1)

    return ax

def plot_normal_regions(experiments : list, regions : list, colors : list = [], ncols: int = 3):
    """Subplots all regions of a list of experiments (unnormalised)"""
    from xps.xps_bg import normalise_dfx
    rows = int(np.ceil(len(regions) / ncols))

    fig, ax = plt.subplots(rows, ncols, figsize=(16, 8))
    fig.add_subplot(111, frameon=False, xticks=[], yticks=[])  # Used for common xlabel and ylabel

    for i,r in enumerate(regions):
        for c,xp in enumerate(experiments):
            xp_norm = normalise_dfx(xp, inplace=False)
            j, k = i//ncols, i%ncols

            if i == len(regions) - 1:   # Set labels from last region
                lb = xp.name

            else:
                lb='__nolabel__'

            try:
                li = plot_region(xp_norm, r, ax=ax[j][k], lb=lb)
                ax[j][k].legend()
                ax[j][k].invert_xaxis()
            except KeyError:    # Auto-exclude regions not recorded for a particular experiment
                pass

            ax[j][k].set_title(r)
            ax[j][k].set_yticks([])
#
        if len(experiments)%2 == 0:
            ax[j][k].invert_xaxis()
    plt.xlabel('\nEnergy [eV]', ha='center')
    plt.tight_layout(w_pad=0.5, h_pad=0.5, pad=0.1)
    return ax


################################################################
########           Coverage functions                   ########
################################################################


def layer_thickness(xpf: XPS_experiment, xp0: XPS_experiment, region: str, mfp: float, takeoff: float):
    """Estimate layer thickness from the attenuation in substrate (region) between an experiment
    with the layer xpf and a reference (clean) experiment xp0.
    Parameters:
    ------------
    - xpf: Experiment with substrate CL attenuated by layer.
    - xp0: Clean experiment for reference.
    - region: substrate region.
    - mfp: mean free path of the molecule conforming the layer (use QUASES to compute it).
        NOTE: Use EAL to account for elastic scattering, if negligible use IMFP.
    - takeoff: angle in degrees between the surface normal and the analyser entry.

    Returns:
    thick, dthick: layer thickness and associated error in the same units as mfp
    -----------

    """
    #from invisible_cities.core.system_of_units import nm
    If = np.trapz(xpf.dfx[region].dropna().counts, dx=0.1)
    I0 = np.trapz(xp0.dfx[region].dropna().counts, dx=0.1)

    dIf = np.sqrt(If)
    dI0 = np.sqrt(I0)

    costh = np.cos(takeoff*np.pi/180)

    thick = mfp*costh*np.log(I0/If)
    dthick = mfp*costh*np.sqrt((dI0/I0)**2 + (dIf/If)**2)
    return thick , dthick

def fbi_n_density(thick_ml: tuple):
    """Compute the atomic density for an FBI (molecule radius hardcoded) cylindric layer
    of height thick_ml (must include central value and error)"""

    r_fbi = 0.683 * nm   # from Fernando's calculation, globularity 1
    V_fbi = r_fbi**3 * 4/3* np.pi

    a_fov = np.pi*1**2          # Assume an homogeneous layer of radius 1 mm2
    V_fov = a_fov * thick_ml[0] * nm
    dV_fov = a_fov * thick_ml[1] * nm

    N_molecs = V_fov / V_fbi # Molecs per mm3
    dN_molecs = dV_fov / V_fbi
    return N_molecs, dN_molecs

def n_layers(xpf: XPS_experiment, xp0: XPS_experiment, r_ml: float, region: str, mfp: float, takeoff: float):
    """Estimate number of layers from the attenuation in substrate (region) between an experiment
    with the layer xpf and a reference (clean) experiment xp0.
    For uncorrected thickness estimation (in nm), use function layer_thickness.
    Parameters:
    ------------
    - xpf: Experiment with substrate CL attenuated by layer.
    - xp0: Clean experiment for reference.
    - r_ml: Size of a Monolayer, in the same units as mfp
        (if the attenuation corresponds to sub-ML regime, correct for bare substrate contribution)
    - region: substrate region.
    - mfp: mean free path of the molecule conforming the layer (use QUASES to compute it).
        NOTE: Use EAL to account for elastic scattering, if negligible use IMFP.
    - takeoff: angle in degrees between the surface normal and the analyser entry.

    Returns:
    layers, dlayers: number layer and associated error
    -----------
    """
    #from invisible_cities.core.system_of_units import nm

    try:
        If = xpf.area[region]
        I0 = xp0.area[region]

    except KeyError:
        If = np.trapz(xpf.dfx[region].dropna().counts, dx=0.1)
        I0 = np.trapz(xp0.dfx[region].dropna().counts, dx=0.1)

    dIf = np.sqrt(If)
    dI0 = np.sqrt(I0)

    costh = np.cos(takeoff*np.pi/180)

    thick = mfp*costh*np.log(I0/If)
    dthick = mfp*costh*np.sqrt((dI0/I0)**2 + (dIf/If)**2)

    if thick < r_ml:
        layers = (If/I0 - 1) / (np.exp(- r_ml * costh / mfp) - 1)
        dlayers = layers * np.sqrt( (dIf/If)**2 + (dI0/I0)**2 )
    else:
        layers = thick / r_ml
        dlayers = dthick / r_ml

    xpf.area.update({'layers' : layers})
    xpf.area.update({'dlayers' : dlayers})

    return layers, dlayers

def arrange_coverages(experiments: list, inds: list,
                      r_ml: float, region: str, mfp: float, takeoff: float)->np.matrix:
    """Estimate n_layers for a list of experiments following the indices inds
    Parameters:
     - experiments: list of XPS_experiment to look into
     - inds: list of indices. It must be arranged such that each set of measurements
             has its reference (clean substrate) index at the end of each list.
             Example:  inds = [[0,1,2,3,5,7,8, 4], [9, 10]]
             will take experiments[4] as clean_substrate, and compute the thickness of experiments[0],
             experiments[1], experiments[2]... Then it will take experiments[10] as clean_substrate and
             compute the thickness for experiments[9]
     - all other params must be passed as for n_layers
    Returns:
    layers_res: matrix (M x 2), with column 0 is the mean value of n_layers and column 1 its error, and
    where M is the number of experiments passed for computation.
    """
    layers_res = []
    for lref in inds:
        for li in lref[:-1]:
            lay, dlay = n_layers(xpf=experiments[li], xp0=experiments[lref[-1]],
                                           r_ml = r_ml, region=region, mfp=mfp, takeoff=takeoff)
            experiments[li].area['layers'] = lay
            experiments[li].area['dlayers'] = dlay

            layers_res.append((lay, dlay))
    layers_res = np.matrix(layers_res)
    return layers_res

def plot_coverages(experiments, label='__nolabel__'):
    layers, dlayers = [], []
    names = []
    for xp in experiments:
        try:
            layers.append(xp.area['layers'])
            dlayers.append(xp.area['dlayers'])
            names.append(xp.name.replace('_', ' ') )
        except KeyError:
            pass

    ax = plt.gca()
    ax.errorbar(x=layers, xerr=dlayers, y=names, fmt='o', markersize=10, label=label)
    ax.set_xlabel('Layers')
    ax.legend()
    return ax

def guess_clean_xp(exp_set):
    lref = []
    li = []
    for i, xp in enumerate(exp_set):
        if 'clean' in xp.name: lref.append(i)
        else: li.append(i)
    if len(lref) == 1:
        inds = li + lref
        return inds
    elif len(lref) < 1:
        print('Did not find clean experiments in this set!')
        raise TypeError
    else:
        print('Too many clean experiments!')
        raise ValueError


############################################################
""" Scofield ASF database functions """
############################################################


def asfdb_read(source: str = ['Al', 'Mg']):
    path = '/Users/pabloherrero/sabat/sabatsw/xps/Scofield.csv'
    asfdb = pd.read_csv(path, sep=';')

    asfdb.index = asfdb['AtomicLevel.symbol'] + asfdb['AtomicLevel.level']
    asfdb.SensitivityFactor = asfdb.SensitivityFactor.apply(lambda x: float(x.replace(' Mb', '')))

    asfMg = asfdb.where(asfdb['ExcitationEnergy'] == '1254 eV').dropna(axis=0, how='all')
    asfAl = asfdb.where(asfdb['ExcitationEnergy'] == '1487 eV').dropna(axis=0, how='all')
    return asfAl if source == 'Mg' else asfAl

def search_asf(region):
    asdf = asfdb_read('Al')

    # Filter symbols in region
    regionf = region.replace('_', '')
    m = re.search(r"\([0-9]\)", regionf)
    try:
        regionf = regionf.replace(m.group(), '')
    except AttributeError: pass

    try:
        val_Scofield = asdf.loc[regionf].SensitivityFactor
        val_powTanum = asdf.loc[regionf].SensitivityFactor / asdf.loc['F1s'].SensitivityFactor
        return {region : val_powTanum}
    except KeyError:
        print(regionf, " region not found in database")
