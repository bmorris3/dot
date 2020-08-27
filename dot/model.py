import logging

import numpy as np
import pymc3 as pm
from pymc3.smc import sample_smc


__all__ = ['Model']


class MeanModel(pm.gp.mean.Mean):
    """
    Mean model for Gaussian process regression on photometry with starspots
    """
    def __init__(self, light_curve, rotation_period, n_spots, contrast, t0,
                 latitude_cutoff=10, partition_lon=True):
        pm.gp.mean.Mean.__init__(self)

        if contrast is None:
            contrast = pm.TruncatedNormal("contrast", lower=0.01, upper=0.99,
                                          testval=0.4, mu=0.5, sigma=0.5)

        self.f0 = pm.TruncatedNormal("f0", mu=0, sigma=1,
                                     testval=light_curve.flux.max(),
                                     lower=-1, upper=2)

        self.eq_period = pm.TruncatedNormal("P_eq",
                                            lower=0.8 * rotation_period,
                                            upper=1.2 * rotation_period,
                                            mu=rotation_period,
                                            sigma=0.2 * rotation_period,
                                            testval=rotation_period)

        BoundedHalfNormal = pm.Bound(pm.HalfNormal, lower=1e-6, upper=0.99)
        self.shear = BoundedHalfNormal("shear", sigma=0.2, testval=0.01)

        self.comp_inclination = pm.Uniform("comp_inc",
                                           lower=np.radians(0),
                                           upper=np.radians(90),
                                           testval=np.radians(1))

        if partition_lon:
            lon_lims = 2 * np.pi * np.arange(n_spots + 1) / n_spots
            lower = lon_lims[:-1]
            upper = lon_lims[1:]
        else:
            lower = 0
            upper = 2 * np.pi

        self.lon = pm.Uniform("lon",
                              lower=lower,
                              upper=upper,
                              shape=(1, n_spots))
        self.lat = pm.TruncatedNormal("lat",
                                      lower=np.radians(latitude_cutoff),
                                      upper=np.radians(180 - latitude_cutoff),
                                      mu=np.pi / 2,
                                      sigma=np.pi / 2,
                                      shape=(1, n_spots))
        self.rspot = BoundedHalfNormal("R_spot",
                                       sigma=0.2,
                                       shape=(1, n_spots),
                                       testval=0.3)
        self.contrast = contrast
        self.spot_period = self.eq_period / (1 - self.shear *
                                             pm.math.sin(self.lat - np.pi / 2) ** 2)
        self.sin_lat = pm.math.sin(self.lat)
        self.cos_lat = pm.math.cos(self.lat)
        self.sin_c_inc = pm.math.sin(self.comp_inclination)
        self.cos_c_inc = pm.math.cos(self.comp_inclination)
        self.t0 = t0

    def __call__(self, X):
        phi = 2 * np.pi / self.spot_period * (X - self.t0) - self.lon
        spot_position_x = (pm.math.cos(phi - np.pi / 2) *
                           self.sin_c_inc *
                           self.sin_lat +
                           self.cos_c_inc *
                           self.cos_lat)
        spot_position_y = -(pm.math.sin(phi - np.pi/2) *
                            self.sin_lat)
        spot_position_z = (self.cos_lat *
                           self.sin_c_inc -
                           pm.math.sin(phi) *
                           self.cos_c_inc *
                           self.sin_lat)
        rsq = spot_position_x ** 2 + spot_position_y ** 2
        spot_model = self.f0 - pm.math.sum(self.rspot ** 2 * (1 - self.contrast) *
                                           pm.math.where(spot_position_z > 0,
                                                         pm.math.sqrt(1 - rsq),
                                                         0),
                                           axis=1)

        return spot_model


class DisableLogger():
    """
    Simple logger disabler to minimize info-level messages during PyMC3
    integration
    """
    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            logging.disable(logging.CRITICAL)

    def __exit__(self, a, b, c):
        if not self.verbose:
            logging.disable(logging.NOTSET)


class Model(object):
    def __init__(self, light_curve, rotation_period, n_spots, scale_errors=1,
                 skip_n_points=1, latitude_cutoff=10, rho_factor=250,
                 verbose=False, min_time=None, max_time=None, contrast=0.7,
                 partition_lon=False):
        """
        Construct a new instance of `~dot.Model`.

        Parameters
        ----------
        light_curve : `~lightkurve.lightcurve.LightCurve`
        rotation_period : float
            Stellar rotation period
        n_spots : int
            Number of spots
        latitude_cutoff : float
            Don't place spots above/below this number of degrees from the pole
        verbose : bool
            Allow PyMC3 dialogs to print to stdout
        partition_lon : bool
            Enforce strict partitions on star in longitude for sampling
        skip_n_points : int (optional)
            Skip every n points for faster runs
        min_time : float or None (optional)
            Minimum time to consider in the model
        max_time : float or None (optional)
            Maximum time to consider in the model
        contrast : float or None (optional)
            Starspot contrast
        rho_factor : float (optional)
            Scale up the GP length scale by a factor `rho_factor`
            larger than the estimated `rotation_period`
        """
        self.lc = light_curve
        self.pymc_model = None
        self.skip_n_points = skip_n_points
        self.rotation_period = rotation_period
        self.n_spots = n_spots
        self.verbose = verbose

        if min_time is None:
            min_time = self.lc.time.min()
        if max_time is None:
            max_time = self.lc.time.max()

        self.mask = (self.lc.time >= min_time) & (self.lc.time <= max_time)
        self.contrast = contrast
        self.scale_errors = scale_errors

        self.pymc_model = None
        self.pymc_gp = None
        self.pymc_gp_white = None
        self.pymc_gp_matern = None

        self._initialize_model(latitude_cutoff=latitude_cutoff,
                               rho_factor=rho_factor,
                               partition_lon=partition_lon)

    def _initialize_model(self, latitude_cutoff, partition_lon, rho_factor):
        """
        Construct a PyMC3 model instance for use with samplers.

        Parameters
        ----------
        latitude_cutoff : float
            Don't place spots above/below this number of degrees from the pole
        partition_lon : bool
            Enforce strict partitions on star in longitude for sampling
        rho_factor : float
            Scale up the GP length scale by a factor `rho_factor`
            larger than the estimated `rotation_period`
        """
        with pm.Model(name='dot') as model:
            mean_func = MeanModel(
                  self.lc,
                  self.rotation_period,
                  n_spots=self.n_spots,
                  latitude_cutoff=latitude_cutoff,
                  contrast=self.contrast,
                  t0=self.lc.time[self.mask][::self.skip_n_points].mean(),
                  partition_lon=partition_lon
            )

            x = self.lc.time[self.mask][::self.skip_n_points]
            y = self.lc.flux[self.mask][::self.skip_n_points]
            yerr = self.scale_errors * self.lc.flux_err[self.mask][::self.skip_n_points]

            ls = rho_factor * self.rotation_period
            mean_err = yerr.mean()
            gp_white = pm.gp.Marginal(mean_func=mean_func,
                                      cov_func=pm.gp.cov.WhiteNoise(mean_err))
            gp_matern = pm.gp.Marginal(cov_func=mean_err ** 2 *
                                       pm.gp.cov.Matern32(1, ls=ls))

            gp = gp_white + gp_matern

            gp.marginal_likelihood("y", X=x[:, None], y=y, noise=yerr)

        self.pymc_model = model
        self.pymc_gp = gp
        self.pymc_gp_white = gp_white
        self.pymc_gp_matern = gp_matern
        self.mean_model = mean_func(x[:, None])

        return self.pymc_model

    def __enter__(self):
        """
        Mocking the pymc3 context manager for models
        """
        self._check_model()
        return self.pymc_model.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Mocking the pymc3 context manager for models
        """
        self._check_model()
        return self.pymc_model.__exit__(exc_type, exc_val, exc_tb)

    def _check_model(self):
        """
        Check that a model instance exists on this object
        """
        if self.pymc_model is None:
            raise ValueError('Must first call `Model._initialize_model` first.')

    def sample_smc(self, draws, random_seed=42, **kwargs):
        """
        Sample the posterior distribution of the model given the data using
        Sequential Monte Carlo.

        Parameters
        ----------
        draws : int
            Draws for the SMC sampler
        random_seed : int
            Random seed
        parallel : bool
            If True, run in parallel
        cores : int
            If `parallel`, run on this many cores

        Returns
        -------
        trace : `~pymc3.backends.base.MultiTrace`
        """
        self._check_model()
        with DisableLogger(self.verbose):
            with self.pymc_model:
                trace = sample_smc(draws, random_seed=random_seed, **kwargs)
        return trace

    def sample_nuts(self, trace_smc, draws, cores=96,
                    target_accept=0.99, **kwargs):
        """
        Sample the posterior distribution of the model given the data using
        the No U-Turn Sampler.

        Parameters
        ----------
        trace_smc : `~pymc3.backends.base.MultiTrace`
            Results from the SMC sampler
        draws : int
            Draws for the SMC sampler
        cores : int
            Run on this many cores
        target_accept : float
            Increase this number up to unity to decrease divergences

        Returns
        -------
        trace : `~pymc3.backends.base.MultiTrace`
            Results of the NUTS sampler
        """
        self._check_model()
        with DisableLogger(self.verbose):
            with self.pymc_model:
                trace = pm.sample(draws,
                                  start=trace_smc.point(-1), cores=cores,
                                  target_accept=target_accept, **kwargs)
                summary = pm.summary(trace)

        return trace, summary

    def optimize(self, start=None, plot=False, **kwargs):
        """
        Optimize the free parameters in `Model` using
        `~scipy.optimize.minimize` via `~exoplanet.optimize`

        Thanks x1000 to Daniel Foreman-Mackey for making this possible.
        """
        from exoplanet import optimize

        with self.pymc_model:
            map_soln = optimize(start=start, **kwargs)

        if plot:
            best_fit = self(map_soln)

            import matplotlib.pyplot as plt
            ax = plt.gca()
            ax.errorbar(self.lc.time[self.mask][::self.skip_n_points],
                        self.lc.flux[self.mask][::self.skip_n_points],
                        self.lc.flux_err[self.mask][::self.skip_n_points],
                        fmt='.', color='k', ecolor='silver', label='obs')
            ax.plot(self.lc.time[self.mask][::self.skip_n_points],
                    best_fit, label='dot')
            ax.set(xlabel='Time', ylabel='Flux')
            ax.legend(loc='lower left')
        return map_soln

    def __call__(self, point=None, **kwargs):
        """
        Evaluate the model with input parameters at ``point``

        Thanks x1000 to Daniel Foreman-Mackey for making this possible.
        """
        from exoplanet import eval_in_model

        with self.pymc_model:
            result = eval_in_model(
                self.mean_model,
                point=point,
                **kwargs
            )
        return result
