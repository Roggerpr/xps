import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy.optimize import curve_fit

class XPSImport():
    """Class to import XPS '.xy' data and metadata.
    If more than one group needs to be retrieved, use import_xps_groups() function.
    Otherwise use the more efficient import_xps_df"""
    def __init__(self, path : str):
        self.path = path

    def find_groups(self):
        """Utility to find number of groups contained in file"""
        groupCount = 0
        with open(self.path) as infile:
            for i,line in enumerate(infile):
                if '# Group:' in line :
                    groupCount += 1
        return groupCount

    def xps_file_metadata(self) -> tuple:
        """Retrieve position, name and number of lines of each spectrum in a .xy file"""

        skipRows0 = []
        nrows0 = []
        names = []

        with open(self.path) as infile:
            for i,line in enumerate(infile):
                if '# ColumnLabels: energy' in line:
                    skipRows0.append(i)
                if '# Region:' in line:
                    names.append(line[21:-1].replace(' ', '_'))
                if '# Values/Curve:' in line:
                    nrows0.append(int(line[21:-1]))
        return (skipRows0, nrows0, names)

    def import_xps_df(self) -> pd.DataFrame:
        """Join all spectra in an xps .xy file, each region contains a column with energy [in eV] and count values"""

        skipRows0, nrows0, names = self.xps_file_metadata()

        frames = []
        for j, re in enumerate(skipRows0):
            if j < len(skipRows0):
                frames.append(pd.read_table(self.path, sep='\s+', skiprows=re+2, nrows = nrows0[j], header=None, names=[names[j], 'counts'],
                                            decimal='.', encoding='ascii', engine='python'))

        df = pd.concat(frames, axis=1)

        index2 = np.array(['energy', 'counts'])
        mi = pd.MultiIndex.from_product([names, index2], names=['range', 'properties'])
        mi.to_frame()
        df.columns = mi

        return df

    def xps_groups_metadata(self) -> list:
        """Retrieve position, name and number of lines of each spectrum in a .xy file and arrange them by groups"""

        groupRows = []
        groupNames = []
        groupCount = -1
        skipRows0 = []
        nrows0 = []
        names = []

        with open(self.path) as infile:
            for i,line in enumerate(infile):
                if '# Group:' in line :
                    groupRows.append(i)
                    groupNames.append(line[21:-1].replace(' ', '_'))
                    groupCount += 1
                    skipRows0.append([])
                    nrows0.append([])
                    names.append([])

                if '# ColumnLabels: energy counts/s' in line:
                    skipRows0[ groupCount ].append( i )

                if '# Region:' in line:
                    names[groupCount ].append(line[21:-1].replace(' ', '_'))
                if '# Values/Curve:' in line:
                    nrows0[ groupCount ].append(int(line[21:-1]))
        return (groupRows, groupNames, skipRows0, nrows0, names)

    def import_xps_groups(self) -> list:
        """Arrange spectra df's by groups from file"""
        dfs = []
        skipRows0, nrows0, names, groupRows = self.xps_groups_metadata()

        for i, gr in enumerate(groupRows):
            frames = []
            for j, sk in enumerate(skipRows0[i]):
                if  (i == len(groupRows)-1 ) and (j == len(skipRows0[i]) -1 ) :
                    frames.append(pd.read_table(self.path, sep='\s+', skiprows=sk+2,  header=None, names=[names[i][j], 'counts'],
                                                    decimal='.', encoding='ascii', engine='python'))

            df = pd.concat(frames, axis=1)
            dfs.append(df)
            return dfs

class XPSana(XPSImport):
    """Analysis methods for XPS spectra"""
    def __init__(self, path, name):
        super().__init__(path)
        self.df = XPSImport.import_xps_df(self)
        self.name = name

    def cosmetics_plot(self, ax = None):
        if ax == None: ax = plt.gca()
        ax.invert_xaxis()
        ax.legend()
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('CPS [A.U.]')

    def plot_region(self, region : str, lb : str = None):

        if lb == None: lb = self.name

        p1 = plt.plot(self.df[region].energy, self.df[region].counts, label=lb)
        self.cosmetics_plot()
        return p1[0]

    def scale_df(self, scale_factor):

        names = list(self.df.columns.levels[0])
        dfnew = pd.DataFrame()

        frames = []
        for n in names:
            x = self.df[n].counts.apply(lambda c : c * scale_factor)
            frames.append( pd.DataFrame([self.df[n].energy, x]).T )
        dfnew = pd.concat(frames, axis=1)

        mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
        mi.to_frame()
        dfnew.columns = mi
        self.df = dfnew

    def reverse_energy_scale(self):
        """Transform energy scale from kinetic to binding"""
        def excitation_energy_metadata(name : str):
            """Find the excitation energy for a region in XPS '.xy' file
            Parameters:
            path : str
                Absolute path to file to search into
            name : str
                Name of the region with underscores"""

            with open(self.path) as infile:
                for i, line in enumerate(infile):
                    if name.replace('_', ' ') in line:
                        chunk = infile.readlines(800)
        #                 print(chunk)
                        for li in chunk:
                            if '# Excitation Energy: ' in li:
                                hv = li[21:-1]
                                return float(hv)
                print('Region %s not found' %name)

        names = list(self.df.columns.levels[0])
        dfnew = pd.DataFrame()

        frames = []

        for n in names:    # Loop over regions
            hv = excitation_energy_metadata(n)    # Find excitation energy from file data
            x = self.df[n].energy.dropna().apply(lambda E : hv - E)  # Subtract hv from KE to yield binding energy
            frames.append( pd.DataFrame([x, self.df[n].counts]).T )
        dfnew = pd.concat(frames, axis=1)

        mi = pd.MultiIndex.from_product([names, np.array(['energy', 'counts'])])
        mi.to_frame()
        dfnew.columns = mi
        self.df = dfnew

    def check_arrays(self, x, y):
        # Make sure we've been passed arrays and not lists.
        x = np.array(x)
        y = np.array(y)
        # Sanity check: Do we actually have data to process here?
        if not (x.any() and y.any()):
            print ("specs.shirley_calculate: One of the arrays x or y is empty. Returning zero background.")
            #return zeros(x.shape)

        # Next ensure the energy values are *decreasing* in the array,
        # if not, reverse them.
        if x[0] < x[-1]:
            is_reversed = True
            x = x[::-1]
            y = y[::-1]
            return x, y, is_reversed
        else:
            is_reversed = False
            return x, y, is_reversed

    def find_integration_limits(self, x, y, flag_plot = False, region : str = None):
        # Locate the biggest peak.
        maxidx = abs(y - np.max(y)).argmin()

        # It's possible that maxidx will be 0 or -1. If that is the case,
        # we can't use this algorithm, we return a zero background.
        if maxidx == 0 or maxidx >= len(y) - 1:
            print ("specs.shirley_calculate: Boundaries too high for algorithm: returning a zero background.")

        # Locate the minima either side of maxidx.
        lmidx = abs(y[0:maxidx] - np.min(y[0:maxidx])).argmin()
        rmidx = abs(y[maxidx:] - np.min(y[maxidx:])).argmin() + maxidx

        if flag_plot:
            #plt.plot(x, y, label='__nolegend__')
            ybase = plt.ylim()[0]
            ind = [maxidx, lmidx, rmidx]
            for i in ind:
                plt.vlines(x = x[i], ymin=ybase, ymax=y[i], color='k')
                plt.text(s= '%.2f'%x[i], x = x[i], y = y[i])

        return lmidx, rmidx

    def shirley_loop(self, x, y,
                     lmidx : int = None,
                     rmidx : int = None,
                     maxit : int = 10, tol : float = 1e-5,
                     DEBUG : bool = False):
        # Initial value of the background shape B. The total background S = yr + B,
        # and B is equal to (yl - yr) below lmidx and initially zero above.

        x, y, is_reversed = self.check_arrays(x, y)

        if (lmidx == None) or (rmidx == None):
            lmidx, rmidx = self.find_integration_limits(x, y, flag_plot=False)
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

        if it >= maxit:
            print("specs.shirley_calculate: Max iterations exceeded before convergence.")
        if is_reversed:
            return ((yr + B)[::-1])
        else:
            return (yr + B)

    def subtract_shirley_bg(self, region, maxit : int = 10, lb : str = None):
        """Plot region and shirley background. Decorator for shirley_loop function"""

        x, y = self.df[region].dropna().energy.values, self.df[region].dropna().counts.values

        if lb == None : lb = self.name
        p1 = plt.plot(x, y, label=lb)
        col = p1[0].get_color()

        self.find_integration_limits(x, y, flag_plot=True, region = region)
        ybg = self.shirley_loop(x, y, maxit = maxit)

        plt.plot(x, ybg, '--', color = col, label='Shirley Background')
        self.cosmetics_plot()

        dfnew = pd.DataFrame({'energy' : self.df[region].energy.dropna(), 'counts' : y - ybg})
        self.df[region] = dfnew

    def subtract_double_shirley(self, region : str, xlim : float, maxit : int = 10):
        """Shirley bg subtraction for double peak"""
        x = self.df[region].dropna().energy
        y = self.df[region].dropna().counts
        p1 = plt.plot(x, y, label=self.name)

        y1 = y[ x > xlim ]
        x1 = x[ x > xlim ]
        y2 = y[ x <= xlim ]
        x2 = x[ x <= xlim ]

        ybg1 = self.shirley_loop(x1, y1, maxit = maxit)
        ybg2 = self.shirley_loop(x2, y2, maxit = maxit)

        col = p1[0].get_color()
        plt.plot(x, np.append(ybg1, ybg2), '--', color=col, label='Double shirley bg')
        y = np.append( y1 - ybg1, y2 - ybg2)
        self.cosmetics_plot()

        dfnew = pd.DataFrame({'energy' : x, 'counts' : y})
        self.df[region] = dfnew

    def subtract_linear_bg (self, region, lb : str = None) -> np.array:
        """Fit background to line and subtract from data"""

        from scipy import stats, polyval
        x = self.df[region].dropna().energy.values
        y = self.df[region].dropna().counts.values
        if lb == None : lb = self.name
        p1 = plt.plot(x, y, label=lb)
        col = p1[0].get_color()

        slope, intercept, r, p_val, std_err = stats.linregress(x, y)
        ybg = polyval([slope, intercept], x);
        plt.plot(x, ybg, '--', color=col, label='Linear Background')
        self.cosmetics_plot()

        dfnew = pd.DataFrame({'energy' : self.df[region].energy.dropna(), 'counts' : y - ybg})
        self.df[region] = dfnew

    def gaussian_smooth(self, region, sigma : int = 2) -> pd.DataFrame:
        from scipy.ndimage.filters import gaussian_filter1d

        y = gaussian_filter1d(self.df[region].dropna().counts.values, sigma = 2)
        dfnew = pd.DataFrame({'energy' : self.df[region].energy.dropna(), 'counts' : y})
        self.df[region] = dfnew

    def gauss(self, x, *p):
        A, mu, sigma = p
        return A *  np.exp(-( x-mu )**2 / (2.*sigma**2))

    def double_gauss(self, x, *p):
        return self.gauss(x, *p[:3]) + self.gauss(x, *p[3:])

    def fit_double_gauss(self, region : str, thres0 : float = 0.5, lb : str = None):
        """Fit to double gauss, estimate loc and scale from peak finding"""

        if lb == None: lb = self.name
        p1 = self.plot_region(region, lb=lb)
        col = p1.get_color()

        x = self.df[region].dropna().energy.values
        y = self.df[region].dropna().counts.values
        def compute_p0_peaks(x, y, thres0) -> list:
            peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
            while len(peaks) > 2:
                thres0 += 0.05
                peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
            while len(peaks) < 2:
                thres0 -= 0.05
                peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
            p0 = [y[peaks[0]], x[peaks[0]], 2, y[peaks[1]], x[peaks[1]], 2]
            return p0

        p0 = compute_p0_peaks(x, y, thres0)
        fit, cov = curve_fit(self.double_gauss, xdata = x , ydata= y, p0=p0)

        plt.plot(x, self.double_gauss(x, *fit), '--', color=col, label='Double gauss fit')

        plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)
        plt.text(s='%.1f'%fit[4], x=fit[4], y=fit[3]*1.05)
        yl = plt.ylim()
        plt.ylim(yl[0], yl[1]*1.5)
        self.cosmetics_plot()

        return fit

    def fit_gauss(self, region : str, thres0 : float = 0.7, lb : str = None):
        """Fit to gauss, estimate loc and scale from peak finding"""

        if lb == None: lb = self.name
        p1 = self.plot_region(region, lb=lb)
        col = p1.get_color()
        x = self.df[region].dropna().energy.values
        y = self.df[region].dropna().counts.values

        def compute_p0_peaks(x, y, thres0) -> list:
            peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
            while len(peaks) > 1:
                thres0 += 0.05
                peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
            while len(peaks) < 1:
                thres0 -= 0.05
                peaks = peakutils.indexes(y, thres = thres0, min_dist=10)
            p0 = [y[peaks[0]], x[peaks[0]], 2]
            return p0

        p0 = compute_p0_peaks(x, y, thres0)
        fit, cov = curve_fit(self.gauss, x, y, p0=p0)

        plt.plot(x, self.gauss(x, *fit), '--', color=col, label='Gauss fit')

        plt.text(s='%.1f'%fit[1], x=fit[1], y=fit[0]*1.1)

        self.cosmetics_plot()

        return fit

def find_and_plot_peaks(df : pd.DataFrame, thres : float = 0.5, col : str = 'r'):
    leny = len(df.index)
    peaks =  peakutils.indexes(df.counts.values, thres=thres)
    x_peaks = leny - df.index[peaks]
    y_peaks = df.counts.values[peaks]
    plt.plot(x_peaks, y_peaks, col+'o', label='Peaks at thres = %.1f' %thres)

    return peaks

def scale_and_plot_spectra(df : pd.DataFrame, dfRef : pd.DataFrame, lb : tuple, thres: float = 0.5) -> float:
    """Plot two spectra and compute average count ratio between main peaks for scaling
    Input:
    -----------------
    df: pd.DataFrame
        DataFrame containing the spectrum region to scale UP
    dfRef: pd.DataFrame
        Reference DataFrame to compare to
    lb : tuple
        Labels for legend
    thres : float
        Peak-finding count threshold, shouldn't be too low
    Output:
    ------------------
    normAv : float
        Scale factor computed as the average ratio between peak heights. Should be > 1,
        otherwise the reference spectrum has weaker intensity than the one intended to scale up
        """
    plt.figure(figsize=(10,8))
    plt.plot(df.energy, df.counts, '-b', label=lb[0])
    plt.plot(dfRef.energy, dfRef.counts, '-r', label=lb[1])

    plt.xlabel('Energy [eV]', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.legend()
    plt.gca().invert_xaxis()

    pe = find_and_plot_peaks(df, thres=0.5, col='b')
    pRef = find_and_plot_peaks(dfRef, thres=0.5, col='r')

    norm = dfRef.counts[pRef] / df.counts[pe]
    normAv = np.average(norm)
    return normAv

def plot_xps_element_spectra(df_ref : pd.DataFrame, df_sg : pd.DataFrame,
                            lb : str,  ax = None):

    y_ref = df_ref.counts
    y_sg = df_sg.counts

    if ax == None: ax = plt.gca()

    step = df_ref.energy[0] - df_ref.energy[1]
    area = np.trapz(df_sg.counts, dx = step) #- np.trapz(df_ref.counts, dx = step)

#    ax.plot(df_ref.energy, y_ref, '-', label='Substrate')
    ax.plot(df_sg.energy, y_sg, '-', label=lb + '\nArea: %.0f' %area)

    ax.set_xlabel('Energy [eV]', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.legend(fontsize = 12)
    ax.invert_xaxis()
    return area
