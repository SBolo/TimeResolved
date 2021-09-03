import glob
import pickle
import os
import sys
import subprocess
import matplotlib as mpl
from matplotlib import gridspec, rcParams
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import scipy as sp
import scipy.signal as signal
from scipy.stats import gaussian_kde
try:
    import fplanck as fp
    fplib = True
except ImportError:
    fplib = False
try:
    from kneed import KneeLocator
    kneelib = True
except ImportError:
    kneelib = False
import BME as bme

def filter_array(x, a):
    x = x[~np.isinf(a)]
    a = a[~np.isinf(a)]
    x = x[~np.isnan(a)]
    a = a[~np.isnan(a)]
    return x, a

class FESNotSetUP(Exception):
    """Custom exception. Used to raise a FESNotSetUP error when the user wants to run the FP simulation without
    setting up the FES first."""
    pass

class TimeResolvedBME():
    def __init__(self, esaxs_dir, meta_saxs, bfile, intname = 'intensity_'):
        """BME wrapper for time-resolved SAXS experiments

        Input
        -----
        esaxs_dir (str): path to the directory with experimental saxs intensities (in .dat format)
        meta_trj (str): path to the metadynamics simulation
        top (str): path to the topology file
        meta_saxs (str): path to the BME file with the metadynamics saxs intensities (in .dat format)
        bfile (str): path to the metadynamics bias file
        intname (str): name of the intensity files
        """
        ### PRIVATE MEMBERS
        # path to the directory where experimental intensities are stored
        self.__esaxs_dir = esaxs_dir
        # list containing experimental intensities
        self.__esaxs = []
        # path to the BME file containing the SAXS intensities from the MetaD simulation
        self.__meta_saxs = meta_saxs
        # list where to store experimental intensities
        self.__exp_ints = []
        # list of optimized thetas for reweighting
        self.__opt_ths = []
        # list of chi^2 of dynamical priors for FP time optimisation
        self.__FP_x2 = []
        # optimal FP simulation time
        self.__FP_opt_time = None
        # optimal FP simulation temperature
        self.__FP_opt_temp = None
        # FP drag (fixed by the user)
        self.__drag = None
        # FP box size
        self.__box_size = None
        # FP fes interpolation function
        self.__fes = None
        # FP evolution
        self.__pt = []
        # dynamical priors
        self.__priors = []
        # flags for sanity checks
        self.__own_priors = False
        self.__computed_priors = False
        self.__fes_is_setup = False
        # constants
        self.__NM = 1e-9

        ### PUBLIC MEMBERS
        # lists of phi, x2i and x2f created after optimising theta
        self.opt_phi = []
        self.opt_x2i = []
        self.opt_x2f = []
        # time difference from the equilibrium distribution of the FP evolution
        self.diff_from_eq = []
        # lists for optimal thetas
        self.opt_ths = []
        self.theta_idxs = []
        self.smooth_ths = []
        # dictionary with reweighting results
        self.res = {'x2i': [], 'x2f': [], 'phi': [], 'w': []}


        # check if the directory with experimental saxs intensities exists
        if not os.path.isdir(self.__esaxs_dir):
            raise ValueError(f'Experimental intensities directory "{self.__esaxs_dir}" not found.')

        # check if the saxs intensities are there. If so, save their names for later, count them and load them
        tmp = glob.glob(self.__esaxs_dir + '/*.dat')
        if not tmp:
            raise ValueError(f'No intensity file found in "{self.__esaxs_dir}".')
        else:
            self.__nframes = len(tmp)
            for i in range(self.__nframes):
                self.__esaxs.append(f'{self.__esaxs_dir}/{intname}{i}.dat') # needed for BME
                self.__exp_ints.append(self.quick_load(self.__esaxs[i])) # needed for FP and plotting

        if not os.path.isfile(self.__meta_saxs):
            raise ValueError(f'BME file "{self.__meta_saxs}" not found.')
        else:
            self.__data = self.quick_load(self.__meta_saxs) #needed for BME and plotting

        # check if the weigh file exists and if it's in the right format
        if not os.path.isfile(bfile):
            raise ValueError(f'Weight file "{bfile}" not found.')
        else:
            """convert MetaD bias to usable weights for reweighting
            The file is expected in Plumed format: 1. column time, 1. column CV and 3. column bias"""
            w = self.quick_load(bfile) # load log-weights
            self.__cv = w[:,1] # save metad cv
            w = np.exp((w[:,-1] - np.max(w[:,-1]))) # apply exponential
            self.__wmetad = w[:] # weights to reconstruct free energy
            self.w0 = w[:]
            self.w0 /= np.sum(self.w0) # static prior weights for reweighting (only for proof of concept!!)

        # plotting style
        self.__set_style()

    def quick_load(self,file):
        """Uses pandas read_csv to read a table file. Much faster than numpy.genfromtxt() and numpy.loadtxt().

        Input
        -----
        file (str): path to a multi-column file

        Returns
        -------
        arr (np.ndarray): numpy array with file content
        """
        dataset = read_csv(file, delimiter=r"\s+", header=None, comment='#')
        arr = dataset.values
        return arr

    def load_priors(self, priors):
        """Load custom ndarray to be used as priors in the dynamical reweighting"""
        if self.__computed_priors:
            print("# WARNING: overwriting FP computed priors.")
        self.__priors = priors
        self.__own_priors = True

    def load_results(self, results):
        """Load results dictionary from a previous calculation

        Input
        -----
        results (str): path to a pkl results file
        """
        pin = open(results, "rb")
        r = pickle.load(pin)
        self.res = r

    def save_results(self, outfile):
        """Save results dictionary in a pickle

        Input
        -----
        outfile (str): path to the pkl to be saved
        """
        with open(outfile + ".pkl", "wb") as fp:
            pickle.dump(self.res, fp)

    def setup_fes(self, T, kde=False, scale=1, s=0, bins=50, mesh_points=501, box_size=300):
        """Set up the free energy for the FP simulation

        Input
        -----
        T (float): simulation temperature
        kde (bool): use KDE to estimate the free energy from data
        bins (int): number of bins used to build the free energy histogram
        mesh_points (int): number of points used in the simulation mesh
        box_size (float): size (in nm) of the simulation box (default 300 means the box will go from -150 to 150 nm).
        """
        AVOGADRO = 6.022*1e-20
        KB = 0.008314463 # Boltzmann constant in kJ/mol

        x = np.linspace(self.__cv.min(), self.__cv.max(), int(bins))
        if kde:
            kdep = gaussian_kde(self.__cv, weights=self.__wmetad)
            kdep.set_bandwidth(kdep.scotts_factor()/scale)
            h = kdep(x)
        else:
            h, _ = np.histogram(self.__cv, weights=self.__wmetad, bins=int(bins))
        fes = -KB * T * np.log(h)  # free energy in kJ/mol
        fes -= np.min(fes)
        x, fes = filter_array(x, fes)
        # us a box_size sized grid to interpolate the free energy
        # also rescale free energy to J for FPlanck
        tck_fes = sp.interpolate.splrep(x, fes, s=s)
        cv_mesh = np.linspace(x.min(), x.max(), int(mesh_points))
        fes_int = sp.interpolate.splev(cv_mesh, tck_fes, der=0)
        xnm = np.linspace(-box_size / 2. * self.__NM, box_size / 2 * self.__NM, int(mesh_points))
        func = sp.interpolate.splev(cv_mesh, tck_fes, der=0) / np.max(fes) * AVOGADRO  # free energy in J
        self.__fes = sp.interpolate.splrep(xnm, func, s=0)
        self.__box_size = box_size
        self.__fes_is_setup = True

    def fpsolve(self, p0_loc, temp, drag=1e-6, t=10, npriors=None):
        """Solves FP equation using MetaD free energy as potential. Used to generate dynamical priors.
        Employs FPlanck library as a depencency. FPlanck employs SI units, so length has to be expressed
        in meters and energy in J.

        Input
        -----
        p0_loc (list): mean (0) and variance (1) of the starting Gaussian (in nm)
        temp (float): temperature of the FP simulation
        drag (float): value of the drag to define the diffusion constant D=kT/drag
        t (float): simulation length
        npriors (int): number of priors to save at the end of the simulation (if None the number of experimental
        frames will be used)
        """
        if not fplib:
            raise ImportError("The fplanck library is missing. Please, install it via: 'pip install fplanck'.")
        if not self.__fes_is_setup:
            raise FESNotSetUP('Free energy surface has not been set up yet. Run setup_fes() first.')

        # define the function to be used as potential by FPlanck
        def U(x):
            return sp.interpolate.splev(x, self.__fes, der=0)

        # set up simulation
        sim = fp.fokker_planck(temperature=temp, drag=drag, extent=self.__box_size*self.__NM,
                            resolution=self.__NM, boundary=fp.boundary.reflecting, potential=U)
        self.eq = sim.steady_state() # equilibrium solution

        # fall back to default number of priors in case no number of steps is provided
        if npriors == None:
            npriors = self.__nframes

        # time-evolve the solution
        pdf = fp.gaussian_pdf(p0_loc[0]*self.__NM, p0_loc[1]*self.__NM)
        time, self.__pt = sim.propagate_interval(pdf, t, Nsteps=npriors)

        # computes the difference between the probability density at time t and the equilibrium one
        # can be used to judge if the simulation is long enough to cover all necessary priors
        self.diff_from_eq = []
        for p in self.__pt:
            self.diff_from_eq.append(np.sum(np.fabs(self.eq - p)))

    def optimise_FP_params(self, times, temps, p0_loc, drag=1e-6, npriors=None, start=0, stride=1, end=-1):
        """Optimises the FP simulation length in such a way that the priors optimally match the experimental data

        Input
        -----
        times (list): list of times to be looped over
        temps (list): list of temperatures to be looped over
        p0_loc (list): mean (0) and variance (1) of the starting Gaussian (in nm)
        drag (float): value of the drag to define the diffusion constant D=kT/drag
        npriors (int): number of priors to save at the end of the simulation (if None the number of experimental
        frames will be used)
        start (int): index of starting solution
        stride (int): stride in selecting solutions
        end (int): index of final solution
        """
        if npriors == None:
            npriors = self.__nframes
        self.__FP_x2 = np.empty((len(temps), len(times)))
        # pre-load metad intensities
        data = self.quick_load(self.__meta_saxs)
        # pre-load experimental intensities
        expis = []
        for i in range(self.__nframes):
            expis.append(self.quick_load(self.__esaxs[i]))

        for j,temp in enumerate(temps):
            for n,time in enumerate(times):
                print("# SOLVING FP: TEMP={i}K, TIME={n}\r".format(i=temp, n=time), end="")
                # solve FP and convert pdfs to weights
                self.fpsolve(p0_loc, drag=drag, t=time, npriors=npriors, temp=temp)
                self.FP2weights(start=start, stride=stride, end=end)
                # compute the prior chi2s and save them in a matrix
                x2 = 0
                for i in range(self.__nframes):
                    expi = expis[i]
                    prior = np.sum(data * self.priors[:,i][:,np.newaxis], axis=0)
                    x2 += np.average(((prior[1:] - expi[:,1])/expi[:,2]) ** 2)
                self.__FP_x2[j,n] = x2/self.__nframes
        # save the optimised indices
        opt_idx = np.unravel_index(self.__FP_x2.argmin(), self.__FP_x2.shape)
        self.__drag = drag
        self.__FP_opt_time = times[opt_idx[1]]
        self.__FP_opt_temp = temps[opt_idx[0]]

    def FP2weights(self, start=0, stride=1, end=-1):
        """Converts evolved FP solutions into prior weights for each frame. The number of selected solutions
        is expected to be strictly equal to the number of experimental frames.

        Input
        -----
        start (int): index of starting solution
        stride (int): stride in selecting solutions
        end (int): index of final solution
        """
        if self.__own_priors:
            print("# WARNING: Overwriting loaded priors.")

        priors = []
        if end < 0:
            end = len(self.__pt)+end+1
        # rng = np.arange(start, end, stride)
        # assert len(rng) == self.__nframes, f'The number of selected priors ({len(rng)}) is different from the number ' \
        #                                    f'of experimental frames ({self.__nframes})'
        mesh = np.linspace(self.__cv.min(), self.__cv.max(), 300)
        for i in np.arange(start, end, stride):
            interp_prob = sp.interpolate.splrep(mesh, self.__pt[i], s=0)
            prior = np.abs(sp.interpolate.splev(self.__cv, interp_prob, der=0))  # associates weights to q values
            prior = prior / np.sum(prior) # normalize the weights to unity
            prior = prior.reshape(len(prior), 1)
            priors.append(prior)
        self.priors = np.concatenate(priors, axis=1) # concatenate all priors into an array
        self.__computed_priors = True

    # def optimise_priors_to_exp(self, verbose=False):
    #     """Optimise the assignement of FP generated priors to each experimental frame. To use this function, make
    #     sure the number of priors generated by FP is much larger (e.g. 10x larger) than the number of experimental
    #     frames and that the simulation time is long enough to reach the equilibrium state.
    #
    #     Returns
    #     -------
    #     chis (list): list of the chi-squared of the optimized chi-squared between priors and experimental frames
    #     """
    #     if self.__computed_priors or self.__loaded_priors:
    #         start = 0
    #         opts, chis = [], []
    #         print("# PRE-COMPUTING PRIOR AVERAGES ...")
    #         priors = [np.average(self.__data, weights=i, axis=0) for i in self.priors.T]
    #         for n,j in enumerate(self.__exp_ints):
    #             x2s = []
    #             for p in priors[start:]:
    #                 x2 = np.average(((j[:, 1] - p[1:]) / j[:, 2]) ** 2)
    #                 x2s.append(x2)
    #             if verbose:
    #                 print(f"# FRAME {n}, X2S: {np.sort(x2s)[:5]}")
    #             opt = np.argmin(x2s) + start
    #             opts.append(opt)
    #             start = opt
    #             chis.append(x2s[opt - start])
    #         self.diff_from_eq = [self.diff_from_eq[i] for i in opts]
    #         self.__pt = self.__pt[opts]
    #         self.priors = (self.priors.T[opts]).T
    #         print(f"# AVERAGE CHI2: {np.average(chis):.2f}")
    #         return chis
    #     else:
    #         raise ValueError("# No priors available. Generate priors with fpsolve() and FP2weights() or load "
    #                          "custom priors with load_priors().")

    def optimise_priors_to_exp(self, s=0.1, mode='max'):
        """Optimise the assignement of FP generated priors to each experimental frame. To use this function, make
        sure the number of priors generated by FP is much larger (e.g. 10x larger) than the number of experimental
        frames and that the simulation time is long enough to reach the equilibrium state.

        Input
        -----
        s (float): threshold of similarity with the minimal x2
        mode (str): 'max', 'min' or 'average'

        Returns
        -------
        chis (list): list of the chi-squared of the optimized chi-squared between priors and experimental frames
        """
        if self.__computed_priors or self.__loaded_priors:
            start = 0
            opts, chis = [], []
            print("# PRE-COMPUTING PRIOR AVERAGES ...")
            priors = [np.average(self.__data, weights=i, axis=0) for i in self.priors.T]
            for n,j in enumerate(self.__exp_ints):
                x2s = []
                for p in priors[start:]:
                    x2 = np.average(((j[:, 1] - p[1:]) / j[:, 2]) ** 2)
                    x2s.append(x2)
                # among all the possible x2s closer to the minimum than 10%, select the latest one
                if mode == 'min':
                    opt = np.argmin(x2s) + start
                else:
                    x2min = np.min(x2s)
                    threshold = x2min + s*x2min
                    relevant_x2 = [idx for idx in np.argsort(x2s) if x2s[idx] <= threshold]
                    if mode == 'max':
                        opt = np.max(relevant_x2) + start
                    elif mode == 'average':
                        index = np.argmin(np.abs(np.array(relevant_x2) - np.average(relevant_x2)))
                        opt = index + start
                    else:
                        return ValueError(f'Not valid mode {mode}. Select among "min", "max" and "average".')

                opts.append(opt)
                start = opt
                chis.append(x2s[opt - start])
            self.diff_from_eq = [self.diff_from_eq[i] for i in opts]
            self.__pt = self.__pt[opts]
            self.priors = (self.priors.T[opts]).T
            print(f"# AVERAGE CHI2: {np.average(chis):.2f}")
            return chis
        else:
            raise ValueError("# No priors available. Generate priors with fpsolve() and FP2weights() or load "
                             "custom priors with load_priors().")

    def populations(self, cv, icutmx, icutmn, weights, theta_idxs):
        '''Computes populations of three states from a MetaD trajectory given two cutoffs and an array of a 1D CV'''
        U, I, F = [], [], []
        for i in range(self.__nframes):
            uu, ii, ff = 0, 0, 0

            for n, qq in enumerate(cv):
                if qq > icutmx:
                    ff += weights[i][theta_idxs[i]][n]
                elif qq < icutmx and qq > icutmn:
                    ii += weights[i][theta_idxs[i]][n]
                else:
                    uu += weights[i][theta_idxs[i]][n]
            U.append(uu)
            F.append(ff)
            I.append(ii)
        return np.array(U), np.array(I), np.array(F)

    def crossval_thetas(self, w0, theta_range, nfold=2):
        """Optimize the values of theta for reweighting by using the nfold cross validation implemented in BME. 
        To be used only if the SAXS points are statistically independent. 

        Input
        -----
        w0 (np.ndarray): one- or two-dimensional array of priors. Expected dimension is nweights or nweights x nframes
        theta_range (np.array): range of thetas to be scanned
        nfold (int): number of folds used in cross validation
        """
        self.opt_ths = []
        for n, f in enumerate(self.__esaxs):
            if len(np.shape(w0)) == 2:
                w = w0[:,n]
            elif len(np.shape(w0)) == 1:
                w = w0[:]
            else:
                raise ValueError(f'Non compatible shape of initial weights, {np.shape(w0)}.')

            rew = bme.Reweight("theta_optimisation", w0=w)
            rew.load(f, self.__meta_saxs)
            print(f"\n# FRAME {n+1}/{self.__nframes}")
            opt_theta = rew.theta_scan(thetas=theta_range, nfold=nfold)
            self.opt_ths.append(opt_theta)

    def elbow_thetas(self, default_idx=None):
        """Use KneeLocator to find the location of the elbow in the phi-X^2 curve.

        Input
        -----
        default_idx (int): theta index to use in case no elbow is found. Default one is the theta in the middle
        of the provided range
        """
        if not kneelib:
            raise ImportError("The kneed library is missing. Please, install it via: 'conda install -c conda-forge "
                              "kneed'.")
        if default_idx == None:
            default_idx = int(len(self.res['phi'][0])/2)

        self.theta_idxs = []
        self.opt_phi, self.opt_x2i, self.opt_x2f = [], [], []
        for i in range(self.__nframes):
            try:
                kneedle = KneeLocator(self.res['phi'][i], self.res['x2f'][i], curve="convex", direction="increasing")
                idx = self.res['phi'][i].index(kneedle.knee)
                self.theta_idxs.append(idx)
            except ValueError:
                idx = default_idx
                self.theta_idxs.append(idx)
            self.opt_phi.append(self.res['phi'][i][idx])
            self.opt_x2i.append(self.res['x2i'][i][idx])
            self.opt_x2f.append(self.res['x2f'][i][idx])

    def get_weights(self, theta_idx, theta_value=None, optimal=None):
        # if theta_value != None and optimal != None:
        #     print("# Warning: theta_value will override optimal.")
        #     theta_idx = self.thetas.index(theta_value)
        # if theta_value:
        #     theta_idx = self.thetas.index(theta_value)
        # elif optimal:
        #     theta_idx = self.theta_idxs[optimal]
        # return self.res['w'][]
        pass

    def smooth_thetas(self, window_size=11, polyorder=1):
        """Smoothens the behaviour of the optimal thetas to provide a continuous function for reweighting

        Input:
        window_size (int): size of the window used for averaging
        polyorder (int): polynomial order for fit
        """
        if not list(self.opt_ths):
            raise ValueError('The list of optimal thetas is empty.')
        self.smooth_ths = signal.savgol_filter(self.opt_ths, window_size, polyorder)
        for i,t in enumerate(self.smooth_ths):
            if t<0:
                self.smooth_ths[i] = min(self.opt_ths)

    def reweight_optimised(self, w0, thetas):
        """Run reweighting for a list of optimised thetas

        Input
        -----
        w0 (np.ndarray): prior weights used in reweighting. Can be either of size n or size n x nframes
        thetas (np.array): array of thetas to be used for each reweighting frame (one per frame)
        """
        self.res = {'x2i': [], 'x2f': [], 'phi': [], 'w': []}
        for n,f in enumerate(self.__esaxs):
            print("# REWEIGHTING {i}/{n}\r".format(i=n+1, n=self.__nframes), end="")
            if len(np.shape(w0)) == 2:
                w = w0[:,n]
            elif len(np.shape(w0)) == 1:
                w = w0[:]
            else:
                raise ValueError('Weights array cannot be more than 2 dimensional')
            rew = bme.Reweight("reweighting", w0=w)
            rew.load(f, self.__meta_saxs)
            chi2i, chi2f, phi = rew.fit(theta=thetas[n])
            weights_opt = rew.get_weights()

            # save relevant quantities
            self.res['x2i'].append(chi2i)
            self.res['x2f'].append(chi2f)
            self.res['phi'].append(phi)
            self.res['w'].append(weights_opt)

    def reweight(self, w0, thetas):
        """Run reweighting with multiple thetas per frame

        Input
        -----
        w0 (np.ndarray): prior weights used in reweighting. Can be either of size n or size n x nframes
        thetas (np.array): array of multiple thetas to be used for each reweighting frame
        """
        self.res = {'x2i': [], 'x2f': [], 'phi': [], 'w': []}
        for n,f in enumerate(self.__esaxs):
            print("# REWEIGHTING {i}/{n}\r".format(i=n+1, n=self.__nframes), end="")
            tmp_chi2i, tmp_chi2f, tmp_phi, tmp_weights_opt = [], [], [], []
            if len(np.shape(w0)) == 2:
                w = w0[:, n]
            elif len(np.shape(w0)) == 1:
                w = w0[:]
            else:
                raise ValueError('Weights array cannot be more than 2 dimensional')
            for t in thetas:
                rew = bme.Reweight("reweighting", w0=w)
                rew.load(f, self.__meta_saxs)
                chi2i, chi2f, phi = rew.fit(theta=t)
                weights_opt = rew.get_weights()
                tmp_chi2i.append(chi2i)
                tmp_chi2f.append(chi2f)
                tmp_phi.append(phi)
                tmp_weights_opt.append(weights_opt)

            # save relevant quantities
            self.res['x2i'].append(tmp_chi2i)
            self.res['x2f'].append(tmp_chi2f)
            self.res['phi'].append(tmp_phi)
            self.res['w'].append(tmp_weights_opt)

    def pdens_per_frame(self, obs, idx_ths, prior_weights, bins=50):
        """Builds probability density of a given CV per each experimental frame. It uses the class optimised weights
        and requires some prior weights to compute the prior probability distribution

        Input
        -----
        obs (np.array): numpy array of the CV
        prior_weights (np.ndarray): priors weights
        bins (int): number of bins to be used in the histogram
        """
        histograms = {'hrw':[], 'hpr':[], 'avrw':[], 'avpr':[]}
        if len(np.shape(prior_weights)) == 1:
            hpr, _ = np.histogram(obs, bins=bins, density=True, weights=prior_weights, range = (obs.min(), obs.max()))
            avpr = np.average(obs, weights=prior_weights)
            histograms['hpr'].append(hpr)
            histograms['avpr'].append(avpr)

        for i in range(self.__nframes):
            if len(np.shape(prior_weights)) == 2:
                hpr, _ = np.histogram(obs, bins=bins, density=True, weights=prior_weights[:,i],
                                      range = (obs.min(), obs.max()))
                avpr = np.average(obs, weights=prior_weights[:,i])
                histograms['hpr'].append(hpr)
                histograms['avpr'].append(avpr)

            hrw, _ = np.histogram(obs, bins=bins, density=True, weights=self.res['w'][i][idx_ths[i]],
                                  range = (obs.min(), obs.max()))
            avrw = np.average(obs, weights=self.res['w'][i][idx_ths[i]])
            histograms['hrw'].append(hrw)
            histograms['avrw'].append(avrw)

        return histograms

    def get_esaxs(self):
        """Returns the private esaxs"""
        return self.__esaxs

    def get_esaxs_dir(self):
        """Returns the private esaxs_dir"""
        return self.__esaxs_dir

    def get_meta_saxs(self):
        """Returns the private meta_saxs"""
        return self.__meta_saxs

    def get_nframes(self):
        """Returns the private nframes"""
        return self.__nframes

    def get_cv(self):
        """Returns the private cv"""
        return np.copy(self.__cv)

    def get_wmetad(self):
        """Returns the private wmetad"""
        return np.copy(self.__wmetad)

    def get_FP_x2(self):
        """Return the private FP_x2"""
        return np.copy(self.__FP_x2)

    def get_FP_opt_time(self):
        """Return private FP_opt_time"""
        return self.__FP_opt_time

    def get_FP_opt_temp(self):
        """Return private FP_opt_temp"""
        return self.__FP_opt_temp

    def get_drag(self):
        """Return private drag"""
        return self.__drag

    def get_pt(self):
        """Return private pt"""
        return np.copy(self.__pt)

    def get_priors(self):
        """Return private priors"""
        return np.copy(self.__priors)

    def get_exp_ints(self):
        return np.copy(self.__exp_ints)

    def prior_status(self):
        """Prints the status of priors: priors can be provided by the user or computed solving FP equation"""
        if self.__own_priors:
            print("# User priors have been loaded")
        if self.__computed_priors:
            print("# Priors have been computd solving FP equation")
        if not self.__own_priors and not self.__computed_priors:
            print("# No priors available")

    def fes_status(self):
        """Returns the status of the free energy"""
        if self.__fes_is_setup:
            print("# Free energy is set up.")
        else:
            print("# Free energy is not setup.")

    def get_fes(self, mesh_points=501):
        """Returns the normalized free energy for FP calculation in a plottable format

        Input
        -----
        mesh_points (int): number of points used to build the mesh for the fes

        Returns
        -------
        x (np.array): x fes points
        fes (np.array): y fes points
        """
        if self.__fes_is_setup:
            x = np.linspace(-self.__box_size / 2. * self.__NM, self.__box_size / 2 * self.__NM, int(mesh_points))
            fes = sp.interpolate.splev(x, self.__fes, der=0)
            return x, fes
        else:
            return [], []

    def __set_style(self):
        rcParams['xtick.labelsize'] = 16
        rcParams['ytick.labelsize'] = 16
        rcParams['axes.labelsize'] = 18
        rcParams['legend.fontsize'] = 16
        rcParams['axes.titlesize'] = 16

    def plot_dynamical_priors(self, ini=0, fin=0, distance=False, outfig=None):
        """
        Plots the progression of the FP solution in time and the deviation of from the equilibrium distribution
        as a function of time.

        Input
        -----
        ini (int): number of points to discard from plot at time 0
        fin (int): number of points to discard from plot at time -1
        outfig (str): file name to save the picture
        """
        if list(self.__pt):

            if distance:
                plt.subplot(1, 2, 1)
                plt.figure(figsize=(15, 5))
            else:
                plt.figure(figsize=(7,5))
            c = np.arange(0, len(self.__pt), 1)
            norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
            cmap.set_array([])

            x = np.linspace(-self.__box_size/2, self.__box_size/2, 300)
            for i in np.arange(int(ini), len(self.__pt) - int(fin), 1):
                plt.plot(x, self.__pt[i], color=cmap.to_rgba(i))
            plt.xlabel('x [nm]')
            plt.ylabel('Probability density')
            cbar = plt.colorbar(cmap)
            cbar.set_label(r'Steps')

            if distance:
                plt.subplot(1, 2, 2)
                plt.plot(self.diff_from_eq)
                plt.xlabel('Steps')
                plt.ylabel(r'$||p(t)-p_0||$')
            plt.tight_layout()

            if outfig:
                plt.savefig(outfig)
            else:
                plt.show()
        else:
            print("# Nothing to plot.")

    def plot_priors_with_fes(self, cv_label='', dt=1, time_unit='ns', outfig=None):

        if list(self.__pt):
            fig, ax = plt.subplots(figsize=(6, 6))
            c = np.arange(0, len(self.__pt)*dt, dt)
            norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
            cmap.set_array([])

            x = np.linspace(self.__cv.min(), self.__cv.max(), 300)
            for i in range(len(self.__pt)):
                ax.plot(x, self.__pt[i], color=cmap.to_rgba(i*dt))
            ax.set_xlabel(cv_label)
            ax.set_ylabel('Probability density')

            ax1 = ax.twinx()
            x = np.linspace(self.__cv.min(), self.__cv.max(), 100)
            h, _ = np.histogram(self.__cv, weights=self.__wmetad, bins=100)
            fes = -np.log(h)
            fes -= np.min(fes)
            ax1.plot(x, fes, c='k', lw=2, ls=':')
            ax1.set_ylabel('Free energy [kT]')
            cbar = plt.colorbar(cmap, orientation='horizontal')
            cbar.set_label(f'Time [{time_unit}]')
            plt.tight_layout()

            if outfig:
                plt.savefig(outfig)
            else:
                plt.show()
        else:
            print("# Nothing to plot.")


    def plot_saxs_fits(self, frames, thidx, w0=[], dt=0.5, time_unit='ns', xlog=True, legend=False, outfig=None):
        """
        Plots the SAXS intensities and residuals (experimental, prior and reweighted) of three selected frames.

        Input
        -----
        frames (list): list with 3 frame indices to be plotted
        thidx (list): list of optimal theta indices corresponding to each plotted frame
        w0 (np.array or np.ndarray): prior weights
        dt (float): time (in time_unit) separating each experimental frame
        xlog (bool): plot x-axis in log-scale
        outfig (str): path to output figure to be saved
        """
        if len(np.shape(w0)) == 1 and w0 != []:
            prior = np.average(self.__data, weights=w0, axis=0)  # constant prior
        if len(frames) > 3:
            frames = frames[:3] # keep only the firtst 3 frames for plotting reasons

        fig = plt.figure(figsize=(9, 5))
        gs = fig.add_gridspec(nrows=4, ncols=3)

        for n, i in enumerate(frames):
            ax1 = fig.add_subplot(gs[0:3, n])
            ax2 = fig.add_subplot(gs[3, n])

            ax1.set_title(f't = {round(i*dt)} {time_unit}')
            posterior = np.average(self.__data, weights=self.res['w'][i][thidx[n]], axis=0)
            if len(np.shape(w0)) == 2:
                prior = np.average(self.__data, weights=w0[:,i], axis=0)
            exp = self.__exp_ints[i]
            ax1.errorbar(exp[:,0] * 10, exp[:,1], yerr=exp[:,2], fmt='o', color='w', ecolor='k',
                         markeredgecolor='k', ms=5, label='Experiment', zorder=0)
            # skip the first one because it's the BME label index
            if w0 != []:
                ax1.plot(exp[:,0] * 10, prior[1:], lw=2, label='Prior', c='tab:blue', zorder=1)
            ax1.plot(exp[:,0] * 10, posterior[1:], lw=2, label='Posterior', c='tab:red', zorder=2)
            ax1.set_yscale('log')
            if xlog:
                ax1.set_xscale('log')
            ax1.set_xticks([])

            if n == 0:
                ax1.set_ylabel(r'Intensity [cm$^{-1}$]')
            if legend:
                ax1.legend(fontsize=12, loc=3)

            if w0 != []:
                ax2.plot(exp[:,0] * 10, (exp[:, 1] - prior[1:]) / exp[:, 2], lw=2, label='Prior', c='tab:blue',
                         zorder=1)
            ax2.plot(exp[:,0] * 10, (exp[:, 1] - posterior[1:]) / exp[:, 2], lw=2, label='Posterior', c='tab:red',
                     zorder=2)

            ax2.set_xlabel(r'q [nm$^{-1}$]')
            if xlog:
                ax2.set_xscale('log')
            if n == 0:
                ax2.set_ylabel(r'$\Delta$I/$\sigma$')

        plt.tight_layout()
        if outfig:
            plt.savefig(outfig, dpi=300)
        else:
            plt.show()

