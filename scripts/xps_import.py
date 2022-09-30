import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils
import os
import re
from copy import deepcopy
from scipy.optimize import curve_fit
from lmfit.model import ModelResult, Model, Parameters
from dataclasses import dataclass

def find_groups(path : str):
    """Utility to find number of groups contained in file"""
    groups = []
    with open(path) as infile:
        for i,line in enumerate(infile):
            if '# Group:' in line :
                groups.append(line[21:-1])

    return groups

def xy_region_delimiters(path: str) -> tuple:
    """Retrieve position, name and number of lines of each spectrum in a .xy file"""

    skipRows0 = []
    nrows0 = []
    names = []

    with open(path) as infile:
        for i,line in enumerate(infile):
            if '# ColumnLabels: energy' in line:
                skipRows0.append(i)
            if '# Region:' in line:
                names.append(line[21:-1].replace(' ', '_'))
            if '# Values/Curve:' in line:
                if len(nrows0) == len(names)-1:
                    nrows0.append(int(line[21:-1]))
    return (skipRows0, nrows0, names)

def xy_group_delimiters(path: str) -> tuple:
    """Retrieve position, name and number of lines of each spectrum in a .xy file"""

    skipRows0 = []
    nrows0 = []
    regions = []
    group_meta = []
    group_names = []
    with open(path) as infile:
        for i,line in enumerate(infile):
            if '# ColumnLabels: energy' in line:
                skipRows0.append(i)
            if '# Region:' in line:
                regions.append(line[21:-1].replace(' ', '_'))
            if '# Values/Curve:' in line:
                if len(nrows0) == len(regions)-1:
                    nrows0.append(int(line[21:-1]))
            if '# Group:' in line :
                name = line[21:-1]
                group_meta.append((skipRows0, nrows0, regions))
                skipRows0, nrows0, regions = [], [], []
                group_names.append(name)

    group_meta.append((skipRows0, nrows0, regions))      # The Group name appears at the beginning
                                                   # so the last bunch of skipr, nrows, regions needs to be appended here

    group_meta = group_meta[1:]        # Also, the first entry contains empty lists, drop it
    return group_meta, group_names

def import_single_df(path, skipRows0, nrows0, regions):
    """Concatenate regions of an experiment group in a dfx"""
    frames = []

    for j, re in enumerate(skipRows0):
        if j < len(skipRows0):
            frames.append(pd.read_table(path, sep='\s+', skiprows=re+2, nrows = nrows0[j], header=None, names=[regions[j], 'counts'],
                                            decimal='.', encoding='ascii', engine='python'))

    dfx = pd.concat(frames, axis=1)

    index2 = np.array(['energy', 'counts'])
    mi = pd.MultiIndex.from_product([regions, index2], names=['range', 'properties'])
    mi.to_frame()
    dfx.columns = mi

    return dfx

def import_xps_df(path: str) -> pd.DataFrame:
    """Join all spectra in an xps .xy file, each region contains a column with energy [in eV] and count values"""

    skipRows0, nrows0, regions = xy_region_delimiters(path) # there are len(skipRows0) - 1 regions in the file
    dfx = import_single_df(path, skipRows0, nrows0, regions)
    return dfx

def import_group_xp(path):
    """Separate groups in a xy file into several XPS_experiments"""
    group_meta, gr_names = xy_group_delimiters(path)

    experiments = []

    for i, (delimiters, n) in enumerate(zip(group_meta, gr_names)):
        skipRows0, nrows0, regions = delimiters[:]

        dfx = import_single_df(path, skipRows0, nrows0, regions)

        xp = XPS_experiment(name = n, path = path, dfx = dfx,
                            delimiters = delimiters)
        experiments.append(xp)
    return experiments

def sort_group_xp(group: list):
    """Search for overview_HD group and merge in main group"""
    if 'overview' in group[1].name:
        print('Merging groups, overview_HD found')
        indx = group[1].dfx.columns.levels[0].values
        ov = indx[np.where('overview' in indx)[0][0]]
        insert_dfx_region(group[0], group[1], region=ov)

        return group[0].dfx
    else:
        print('Found zy scanning group or other pattern, import separately ', os.path.split(group[0].path)[1] )
        print([xp.name for xp in group])
        raise TypeError

def excitation_energy_metadata(path : str , name : str):
        """Find the excitation energy for a region in XPS '.xy' file
        Parameters:
        path : str
            Absolute path to file to search into
        name : str
            Name of the region with underscores"""

        with open(path) as infile:
            for i, line in enumerate(infile):
                if name.replace('_', ' ') in line:
                    chunk = infile.readlines(800)
    #                 print(chunk)
                    for li in chunk:
                        if '# Excitation Energy: ' in li:
                            hv = li[21:-1]
                            return float(hv)
            print('Region %s not found' %name)

def ke_to_be(dfx : pd.DataFrame, hv : float) -> pd.DataFrame:
    """Transform energy scale from kinetic to binding"""
    names = list(dfx.columns.levels[0])
    dfnew = pd.DataFrame()

    frames = []
    for n in names:    # Loop over regions
        x = dfx[n].energy.dropna().apply(lambda E : hv - E)  # Subtract hv from KE to yield binding energy
        frames.append( pd.DataFrame([x, dfx[n].counts]).T )
    dfnew = pd.concat(frames, axis=1)

    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
    mi.to_frame()
    dfnew.columns = mi
    return dfnew

def check_arrays(dfr) -> bool:
    """Check whether file is in BE or KE scale"""
    x, y = dfr.dropna().energy.values, dfr.dropna().counts.values
    # Next ensure the energy values are *decreasing* in the array,
    if x[0] < x[-1]:
        is_reversed = True
        return is_reversed
    else:
        is_reversed = False
        return is_reversed

    dfx = import_multiscan_df(path, skipRows0, nrows0, regions)

    relpath, filename = os.path.split(path)
    dir_name = os.path.split(relpath)[1]
    da = re.search('\d+_', filename).group(0).replace('/', '').replace('_', '')
    if da[:4] != '2020':
        date = re.sub('(\d{2})(\d{2})(\d{4})', r"\1.\2.\3", da, flags=re.DOTALL)
    else:
        date = re.sub('(\d{4})(\d{2})(\d{2})', r"\1.\2.\3", da, flags=re.DOTALL)

    other_meta = filename.replace(da, '')[1:].strip('.xy')
    if name == None: name = other_meta
    if label == None: label = da+'_'+other_meta

    return XPS_experiment(path = path, dfx = dfx, delimiters = delimiters, color = color,
                          name = name, label = label, date = date, other_meta = other_meta, fit={})

@dataclass
class XPS_experiment:
    """XPS dataclass with regions dfx and metadata
    Attrs:
    -------
    dfx : pd.DataFrame
        table containing all regions found in .xy file
    delimiters : tuple
        position, extension and name of each region to correctly import dfx
    name : str = None
        short name to reference the experiment
    label : str = None
        longer description of the experiment (cleaning, preparation conditions...)
    date : str = None
        experiment date as read in the filename
    other_meta : str = None
        other info contained in the filename
    area : dict
        dictionary with name of regions and integrated areas
    color: str
        color for plotting (property not stored)
    ls: str
        linestyle (solid, dashed...)
    """
    path : str = None
    delimiters : tuple = None
    name : str = None
    label : str = None
    date : str = None
    other_meta : str = None
    dfx : pd.DataFrame = None
    area : dict = None
    fit : dict = None
    color : str = None
    ls : str = None

def xps_data_import(path : str, name : str = None, label : str = None, color: str = None) -> XPS_experiment:
    """Method to arrange a XPS_experiment data"""
    import re
    delimiters = xy_region_delimiters(path)
    skipRows0, nrows0, regions = delimiters[:]
    dfx = import_single_df(path, skipRows0, nrows0, regions)

    try:
        if check_arrays(dfx[delimiters[2][0]]):
            hv = excitation_energy_metadata(path, delimiters[2][0])
            dfx = ke_to_be(dfx, hv)
    except ValueError:
        group = import_group_xp(path)
        dfx = sort_group_xp(group)  # Search for overview_HD

    relpath, filename = os.path.split(path)
    dir_name = os.path.split(relpath)[1]
    da = re.search('\d+_', filename).group(0).replace('/', '').replace('_', '')
    if da[:4] != '2020':
        date = re.sub('(\d{4})(\d{2})(\d{2})', r"\1.\2.\3", da, flags=re.DOTALL)
    else:
        date = re.sub('(\d{2})(\d{2})(\d{4})', r"\1.\2.\3", da, flags=re.DOTALL)

    other_meta = filename.replace(da, '')[1:].strip('.xy')
    if name == None: name = other_meta
    if label == None: label = da+'_'+other_meta

    return XPS_experiment(path = path, dfx = dfx, delimiters = delimiters, color = color,
                          name = name, label = label, date = date, other_meta = other_meta, fit={})

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
    """Import a file with scans stored separately"""
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

def read_scan_acquisition_time(path):
    with open(path) as infile:
        read_time = []
        for i,line in enumerate(infile):
            if ('# Acquisition Date:' in line) and (i > 19):
                rt = line[20:-1]
                read_time.append(datetime.datetime.strptime(rt, '%m/%d/%y %H:%M:%S UTC') )
    return read_time

def get_all_regions(experiments: list)->list:
    allr = []
    for xp in experiments:
        for r in xp.dfx.columns.levels[0].values:
            if ('overview' not in r) and (r not in allr):
                allr.append(r)
    return allr

##############################   Insert region from other experiment  ###########################


def insert_dfx_region(xp: XPS_experiment, xpFrom:XPS_experiment, region: str, inplace:bool = False):
    """Insert a region from one dfx (xpFrom) into another (xp) for which it is missing"""
    sourcedf = xp.dfx
    newreg = xpFrom.dfx[region]
    names = list(sourcedf.columns.levels[0].values)
    dfnew = pd.DataFrame()
    frames = []

    for n in names:    # Loop over regions
        x = sourcedf[n].energy.dropna()
        frames.append( pd.DataFrame([x, sourcedf[n].counts]).T )

    frames.append(pd.DataFrame([newreg.energy, newreg.counts]).T)
    names.append(region)
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

##############################   Processed files  ###########################

def write_processed_xp(filepath : str, xp : XPS_experiment):
    """Save processed XPS experiment to file"""
    import csv;
    with open(filepath, 'w') as fout:
        writer = csv.writer(fout, delimiter='=')
        for att in xp.__dict__.keys():   # Loop over class attributes except dfx (last)
            if (att != 'dfx') and (att != 'ls'):
                writer.writerow([att, getattr(xp, att)])
        writer.writerow(['dfx', ''])
        xp.dfx.to_csv(fout, sep=',')

    if xp.fit != {}: store_fits(xp, filepath)

def read_dict_area(line : str):
    """Read area/fit dictionary from processed XPS file"""
    line_area = line.split('=')[1].split(', ')
    try:
        line_area[0] = line_area[0][1:]   # Remove '{}'
        line_area[-1] = line_area[-1][:-2] # Remove '}\n'
        area = dict((key.replace("'", ''), float(val)) for key, val in [word.split(':') for word in line_area])
    except ValueError: area = {}  # ValueError raises if the dict is empty
    return area

def read_processed_dfx(path : str, names : list, skiprows0 : int = 9) -> pd.DataFrame:
    """Read and format dfx from processed file path"""
    dfx = pd.read_csv(path, header=None, index_col=0, skiprows=skiprows0, engine='python')
    index2 = np.array(['energy', 'counts'])
    mi = pd.MultiIndex.from_product([names, index2], names=['range', 'properties'])
    mi.to_frame()
    dfx.columns = mi
    dfx.index.name=None
    return dfx

def read_processed_xp(path: str, color: str = None, ls: str = None) -> XPS_experiment:
    """Read XPS_experiment class from file
       If old head format is detected, correct it automatically"""
    from itertools import islice

    with open(path) as fin:
        head = list(islice(fin, 11))

        delimiters = head[1].split('=')[1][:-1]
        name = head[2].split('=')[1][:-1]
        label = head[3].split('=')[1][:-1]
        date = head[4].split('=')[1][:-1]
        other_meta = head[5].split('=')[1][:-1]

        try:
            assert ('area' in head[6]) and ('fit' in head[7]), 'Old head format'
            area = read_dict_area(head[6])
            fit = load_fits(path)
            color = head[8].split('=')[1][:-1]
            if color == '': color = None

            names = head[10].split(',')[1:-1:2]
            dfx = read_processed_dfx(path, names, skiprows0=12)

        except AssertionError as e:      # Process old head format and raise flag to write corrected file
            print(e)
            correct_processed_head(path)

    return XPS_experiment(path = path, dfx = dfx, delimiters = delimiters, name = name, color = color, ls = ls,
                        label = label, date = date, other_meta = other_meta, area = area, fit = fit)


def correct_processed_head(path: str):
    """If old head format is detected, process file accordingly and correct the file"""
    from itertools import islice

    with open(path) as fin:
        head = list(islice(fin, 11))

        delimiters = head[1].split('=')[1][:-1]
        name = head[2].split('=')[1][:-1]
        label = head[3].split('=')[1][:-1]
        date = head[4].split('=')[1][:-1]
        other_meta = head[5].split('=')[1][:-1]
        ### Up to here proceed as with new format (same as in read_processed_xp)

        lindex = np.where(np.array(head) == 'dfx=\n')[0][0] + 1
        area, fit = {}, {}
        color, ls = None, None
        names = head[lindex].split(',')[1:-1:2]
        dfx = read_processed_dfx(path, names, skiprows0=lindex+2)

        xp = XPS_experiment(path = path, dfx = dfx, delimiters = delimiters, name = name, color = color, ls = ls,
                            label = label, date = date, other_meta = other_meta, area = area, fit = fit)
    write_processed_xp(path, xp)


def itx_import(path : str, name: str, label: str) -> XPS_experiment:
    with open(path) as fin:
        lines = fin.readlines()
        EE, SS = [], []
        KE = []
        names = []
        counts = []    # skiplines = []
        pattern = r"'([A-Za-z0-9_\./\\-\\ ]*)'"
        npoints = []
        for i,l in enumerate(lines):
            if 'SetScale/I x' in l:
                KE.append(l[16:-1])  # This all could be regex...
            if 'Scan Step' in l:
                SS.append(float(l[24:-1]))
            if 'WAVES/S/N' in l:
                npoints.append(int(re.findall(r'\((\d+)\)', l)[0]))
                names.append(re.findall(pattern, l)[0])
            if 'Excitation Energy' in l:
                EE.append(float(l[24:-1]))
            if 'BEGIN' in l:
                counts.append(pd.Series(list(map(float, lines[i+1].split()))))
#                 skiplines.append(i)

    frames = []
    for i, c in enumerate(counts):
#         counts = pd.Series(pd.read_table(pathfile, sep='\s+', skiprows=skiplines[i]+1, nrows=1).T.index)
        ke0, kef = re.findall('\d+\.\d+', KE[i])
        kenergy = np.linspace(float(ke0), float(kef)+SS[i], num=npoints[i])
        bindenergy = EE[i] - kenergy
        assert len(bindenergy) == len(c), "Import error: xy lengths do not coincide"
        region = pd.DataFrame({'energy': bindenergy, 'counts':c})
        frames.append(region)

    dfx = pd.concat(frames, axis=1)
    mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])], names=['range', 'properties'])

    mi.to_frame()
    dfx.columns = mi
    dfx.index.name=None

    return XPS_experiment(path = path, dfx = dfx, name= name, label=label)

def export_csv_region(file: str, xp: XPS_experiment, regions: list = None):
    """Export to Igor compatible csv format"""
    if regions == None: regions = xp.dfx.columns.levels[0].values
    for r in regions:
        filename, extension = os.path.splitext(file)
        filename += '_'+r
        file = filename+extension
        with open(file, 'w') as fout:
            header = r+'_BE\t' +r+'_cps\t'
            fout.write(header+'\n')
            xp.dfx[r].to_csv(fout, sep='\t', na_rep='NaN', index=None, header=False)

def pickle_xp(file : str, xp : XPS_experiment):
    import pickle
    with open(file, 'wb') as out:
        pickle.dump(xp, out, -1)

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

def store_proc_results(exps: list):
    for xpu in exps:
        print('Stored ', xpu.path)
        write_processed_xp(xpu.path, xpu)
        
###################     Fit file store and load       ###################

def store_fits(xp: XPS_experiment, path: str = None):
    """Store fits of a XPS_experiment in separate file"""
    if path == None:
        path = xp.path + '.ft'
    else:
        path += '.ft'
    with open(path, 'w') as fout:
        for k in xp.fit.keys():
            fout.write('XPFit %s:\n\n' %k)
            dump_region = xp.fit[k].dumps()
            fout.write(dump_region + '\n\n')
    print('Storing of file %s successful' %path)

def load_fits(path: str):
    """Read stored fits in .ft files"""

    file = path + '.ft'
    fits = {}
    try:
        with open(file) as infile:
            lines = infile.readlines()
            for i, line in enumerate(lines):
                if 'XPFit ' in line:
                    region = line.replace('XPFit ', '').replace(':\n', '')
                    rs = ModelResult(Model(lambda x: x, None), Parameters())
                    fits[region] = rs.loads(lines[i+2])
    except FileNotFoundError as e:
        print('%s, returning empty fit dict' %e)
    return fits

###################     SESSA simulation files       ###################

def import_simulation_file(path: str, ke: bool = False) -> XPS_experiment:
    """Import SESSA .dat simulation data. If ke flag is True: convert KE to BE
        Return XPS_experiment with dfx containing only overview region"""
    filename = os.path.split(path)[1]
    name = os.path.splitext(filename)[0]

    df = pd.read_csv(path, skiprows=10, sep='\t', names=['energy', 'counts', 'no'])
    df = df.drop('no', axis=1)
    if ke:
        df.energy = 1486.6 - df.energy
    df.counts /= df.counts.max() # Normalization to 1
    # df.set_index('energy', drop=True, inplace=True)
    mi = pd.MultiIndex.from_product([['overview'], np.array(['energy', 'counts'])])
    mi.to_frame()
    df.columns = mi

    xpsim = XPS_experiment(path = path, name=name, dfx=df, area={}, fit={})
    return xpsim


def splice_overview(xp: XPS_experiment, region: str, eup: float, edw: float, ov: str = 'overview') -> XPS_experiment:
    """Select a region of the spectrum overview and insert it in the dfx"""
    xpc = deepcopy(xp)
    xpc = crop_spectrum(xp, ov, eup=eup, edw=edw)
    xpc.dfx.rename(columns={ov:region}, inplace=True)

    insert_dfx_region(xp, xpc, region, inplace=True)
    return xpc
