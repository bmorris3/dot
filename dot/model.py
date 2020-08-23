import matplotlib.pyplot as plt
import numpy as np
from exoplanet.gp import terms, GP
import pymc3 as pm
from pymc3.smc import sample_smc
import logging

__all__ = ['Model']


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
    def __init__(self, light_curve, rotation_period, n_spots,
                 skip_n_points=1, latitude_cutoff=10, rho_factor=250,
                 verbose=False, min_time=None, max_time=None, contrast=0.7):
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
            min_time = self.lc.time.min() - 1
        if max_time is None:
            max_time = self.lc.time.max() + 1

        self.mask = (self.lc.time > min_time) & (self.lc.time < max_time)
        self.contrast = contrast
        self._initialize_model(rotation_period=rotation_period,
                               n_spots=n_spots,
                               latitude_cutoff=latitude_cutoff,
                               contrast=contrast,
                               rho_factor=rho_factor)

    def _initialize_model(self, rotation_period, n_spots, latitude_cutoff=10,
                          partition_lon=True, contrast=0.7, rho_factor=250):
        """
        Construct a PyMC3 model instance for use with samplers.

        Parameters
        ----------
        rotation_period : float
            Stellar rotation period
        n_spots : int
            Number of spots
        latitude_cutoff : float
            Don't place spots above/below this number of degrees from the pole
        scale_error : float
            Scale up the errorbars by a factor of `scale_error`
        partition_lon : bool
            Enforce strict partitions on star in longitude for sampling
        contrast : float or None
            Starspot contrast
        rho_factor : float (default: 250)
            Scale up the GP length scale by a factor `rho_factor`
            larger than the estimated `rotation_period`
        """
        with pm.Model(name=f'{n_spots}') as model:
            eq_period = pm.TruncatedNormal("P_eq",
                                           lower=0.4 * rotation_period,
                                           upper=1.5 * rotation_period,
                                           mu=rotation_period,
                                           sigma=0.2 * rotation_period)
            shear = pm.HalfNormal("shear",
                                  sigma=0.2)
            comp_inclination = pm.Uniform("comp_inc",
                                          lower=np.radians(0),
                                          upper=np.radians(90))

            if partition_lon:
                lon_lims = 2 * np.pi * np.arange(n_spots + 1) / n_spots
                lower = lon_lims[:-1]
                upper = lon_lims[1:]
            else:
                lower = 0
                upper = 2 * np.pi

            lon = pm.Uniform("lon",
                             lower=lower,
                             upper=upper,
                             shape=(1, n_spots))
            lat = pm.TruncatedNormal("lat",
                                     lower=np.radians(latitude_cutoff),
                                     upper=np.radians(180 - latitude_cutoff),
                                     mu=np.pi / 2,
                                     sigma=np.pi / 2,
                                     shape=(1, n_spots))
            rspot = pm.HalfNormal("R_spot",
                                  sigma=0.1,
                                  shape=(1, n_spots))

            spot_period = eq_period / (1 - shear * pm.math.sin(lat - np.pi / 2) ** 2)
            phi = 2 * np.pi / spot_period * (self.lc.time[self.mask][::self.skip_n_points][:, None] -
                                             self.lc.time.mean()) - lon

            spot_position_x = (pm.math.cos(phi - np.pi / 2) *
                               pm.math.sin(comp_inclination) *
                               pm.math.sin(lat) +
                               pm.math.cos(comp_inclination) *
                               pm.math.cos(lat))
            spot_position_y = -(pm.math.sin(phi - np.pi/2) *
                                pm.math.sin(lat))
            spot_position_z = (pm.math.cos(lat) *
                               pm.math.sin(comp_inclination) -
                               pm.math.sin(phi) *
                               pm.math.cos(comp_inclination) *
                               pm.math.sin(lat))
            rsq = spot_position_x ** 2 + spot_position_y ** 2
            if contrast is None:
                contrast = pm.TruncatedNormal('contrast',
                                              lower=0.01,
                                              upper=0.99,
                                              mu=0.5,
                                              sigma=0.5)

            spot_model = 1 - pm.math.sum(rspot ** 2 * (1 - contrast) *
                                         pm.math.where(spot_position_z > 0,
                                                       pm.math.sqrt(1 - rsq),
                                                       0),
                                         axis=1)

            gp = GP(
                terms.Matern32Term(sigma=1, rho=rho_factor * rotation_period),
                self.lc.time[self.mask][::self.skip_n_points],
                self.lc.flux_err[self.mask][::self.skip_n_points] ** 2,
                mean=spot_model
            )

            # Condition the GP on the observations and add the marginal likelihood
            # to the model
            gp.marginal("gp",
                observed=self.lc.flux[self.mask][::self.skip_n_points]
            )

        self.pymc_model = model
        self.pymc_gp = gp
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
            raise ValueError('Must first call `Model.construct_model` first.')

    def sample_smc(self, draws, random_seed=42, parallel=True, cores=1,
                   **kwargs):
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
                trace = sample_smc(draws, random_seed=random_seed,
                                   parallel=parallel, cores=cores, **kwargs)
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
