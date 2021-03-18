import logging

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from celerite2.theano import terms, GaussianProcess

__all__ = ['Model']


class MeanModel(object):
    """
    Mean model for Gaussian process regression on photometry with starspots
    """
    def __init__(self, light_curve, rotation_period, n_spots, contrast, t0,
                 latitude_cutoff=10, partition_lon=True):

        if contrast is None:
            contrast = pm.TruncatedNormal("contrast", lower=0.01, upper=0.99,
                                          testval=0.4, mu=0.5, sigma=0.5)
        self.contrast = contrast

        self.f0 = pm.TruncatedNormal("f0", mu=light_curve.flux.max(), sigma=1,
                                     testval=light_curve.flux.max(),
                                     lower=-2, upper=2)
        self.eq_period = pm.TruncatedNormal("P_eq",
                                            lower=0.8 * rotation_period,
                                            upper=1.2 * rotation_period,
                                            mu=rotation_period,
                                            sigma=0.2 * rotation_period,
                                            testval=rotation_period)

        eps = 1e-5  # Small but non-zero number
        BoundedHalfNormal = pm.Bound(pm.HalfNormal, lower=eps, upper=0.8)
        self.shear  = BoundedHalfNormal("shear", testval=0.2)

        self.comp_inclination = 0  #pm.Uniform("comp_inc",
                                   #        lower=0,
                                   #        upper=np.pi/2,
                                   #        testval=np.radians(1))

        if partition_lon:
            lon_lims = 2 * np.pi * np.arange(n_spots + 1) / n_spots
            lower = lon_lims[:-1]
            upper = lon_lims[1:]
            testval = np.mean([lower, upper], axis=0)
        else:
            lower = 0
            upper = 2 * np.pi
            testval = 2 * np.pi * np.arange(n_spots) / n_spots + 0.01

        self.lon = pm.Uniform("lon",
                              lower=lower,
                              upper=upper,
                              shape=(1, n_spots),
                              testval=testval)
        self.lat = pm.Uniform("lat",
                              lower=np.radians(latitude_cutoff),
                              upper=np.radians(180 - latitude_cutoff),
                              shape=(1, n_spots),
                              testval=np.pi/2)

        self.rspot = BoundedHalfNormal("R_spot",
                                       sigma=0.4,
                                       shape=(1, n_spots),
                                       testval=0.3)

        self.spot_period = self.eq_period / (1 - self.shear *
                                             tt.sin(self.lat - np.pi / 2) ** 2)
        self.sin_lat = tt.sin(self.lat)
        self.cos_lat = tt.cos(self.lat)
        self.sin_c_inc = tt.sin(self.comp_inclination)
        self.cos_c_inc = tt.cos(self.comp_inclination)
        self.t0 = t0

    def __call__(self, X):
        phi = 2 * np.pi / self.spot_period * (X[:, None] - self.t0) - self.lon

        spot_position_x = (tt.cos(phi - np.pi / 2) *
                           self.sin_c_inc *
                           self.sin_lat +
                           self.cos_c_inc *
                           self.cos_lat)
        spot_position_y = -(tt.sin(phi - np.pi/2) *
                            self.sin_lat)
        spot_position_z = (self.cos_lat *
                           self.sin_c_inc -
                           tt.sin(phi) *
                           self.cos_c_inc *
                           self.sin_lat)
        rsq = spot_position_x ** 2 + spot_position_y ** 2
        spot_model = self.f0 - tt.sum(self.rspot ** 2 *
                                      (1 - self.contrast) *
                                      tt.where(spot_position_z > 0,
                                               tt.sqrt(1 - rsq), 0),
                                      axis=1)

        return spot_model


class DisableLogger(object):
    """
    Simple logger disabler to minimize info-level messages during PyMC3
    integration
    """
    def __init__(self, verbose=False):
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
                 partition_lon=True, use_gp=False, name=None):
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
        self.gp = None

        self._initialize_model(latitude_cutoff=latitude_cutoff,
                               rho_factor=rho_factor,
                               partition_lon=partition_lon,
                               use_gp=use_gp, name=name)

    def _initialize_model(self, latitude_cutoff, partition_lon, rho_factor,
                          use_gp=False, name=None):
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
        with pm.Model(name=name) as model:
            mean_func = MeanModel(
                  self.lc,
                  self.rotation_period,
                  n_spots=self.n_spots,
                  latitude_cutoff=latitude_cutoff,
                  contrast=self.contrast,
                  t0=self.lc.time[self.mask][::self.skip_n_points].mean(),
                  partition_lon=partition_lon
            )

            x = np.ascontiguousarray(self.lc.time[self.mask][::self.skip_n_points], dtype='float64')
            y = np.ascontiguousarray(self.lc.flux[self.mask][::self.skip_n_points], dtype='float64')
            yerr = np.ascontiguousarray(self.scale_errors * self.lc.flux_err[self.mask][::self.skip_n_points], dtype='float64')

            ls = rho_factor * self.rotation_period
            mean_err = yerr.mean()
            sigma = pm.HalfNormal("sigma", sigma=mean_err)

            if use_gp:
                # Set up the kernel an GP
                kernel = terms.Matern32Term(sigma=sigma, rho=ls)
                gp = GaussianProcess(kernel, t=x, yerr=yerr, mean=mean_func)
                gp.marginal("gp", observed=y)
            else:
                pm.Normal("obs", mu=mean_func(x), sigma=yerr, observed=y)

        if use_gp:
            self.gp = gp

        self.pymc_model = model
        self.mean_model = mean_func(x)

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

    def __call__(self, point=None, use_gp=False, **kwargs):
        """
        Evaluate the model with input parameters at ``point``

        Thanks x1000 to Daniel Foreman-Mackey for making this possible.
        """
        from exoplanet import eval_in_model

        with self.pymc_model:
            if use_gp:
                mu, var = eval_in_model(
                    self.gp.predict(self.lc.time[self.mask][::self.skip_n_points],
                                    return_var=True), point
                )

            mean_eval = eval_in_model(
                self.mean_model, point
            )
        if use_gp:
            return mu + mean_eval, var
        else:
            return mean_eval