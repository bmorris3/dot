import matplotlib.pyplot as plt
import numpy as np
from celerite import GP
from celerite.terms import Matern32Term
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
        logging.disable(logging.NOTSET)


class Model(object):
    def __init__(self, light_curve, rotation_period, n_spots,
                 skip_n_points=1, latitude_cutoff=10, scale_error=5,
                 verbose=False, min_time=None, max_time=None, contrast=0.7):
        """
        Construct a new instance of `Model`.

        Parameters
        ----------
        light_curve : `~lightkurve.LightCurve`
        rotation_period : float
        n_spots : int
        skip_n_points : int
        latitude_cutoff : float
        scale_error : float
        verbose : bool
        min_time : float
        max_time : float
        contrast : float or None
            If `None`, fit for the contrast of each spot independently
        """
        self.lc = light_curve
        self.model = None
        self.skip_n_points = skip_n_points
        self.rotation_period = rotation_period
        self.n_spots = n_spots
        self.verbose = verbose
        self.scale_error = scale_error
        self.mask = (self.lc.time > min_time) & (self.lc.time < max_time)
        self.contrast = contrast
        self._initialize_model(rotation_period, n_spots,
                               latitude_cutoff=latitude_cutoff,
                               scale_error=scale_error, verbose=verbose,
                               contrast=contrast)

    def gp_normalize(self, log_sigma=1, log_rho=8, plot=False):
        """
        Use a Matern 3/2 kernel to normalize the data by a smooth Gaussian
        process.

        Parameters
        ----------

        Returns
        -------

        """
        gp = GP(Matern32Term(log_sigma=log_sigma, log_rho=log_rho))
        gp.compute(self.lc.time[self.mask] / 100, self.lc.flux_err[self.mask])
        gp_trend = gp.predict(self.lc.flux[self.mask],
                              self.lc.time[self.mask] / 100,
                              return_cov=False)
        if plot:
            plt.plot(self.lc.time[self.mask], self.lc.flux[self.mask])
            plt.plot(self.lc.time[self.mask], gp_trend)
            plt.show()

        self.lc.flux /= gp_trend

    def _initialize_model(self, rotation_period, n_spots, latitude_cutoff=10,
                          scale_error=5, verbose=False, partition_lon=True,
                          contrast=0.7):
        """
        Construct a PyMC3 model instance for use with samplers.

        Parameters
        ----------
        rotation_period : float
        n_spots : int
        latitude_cutoff : float
        scale_error : float
        verbose : bool
        partition_lon : bool
        contrast : float or None
        """
        with DisableLogger(verbose):
            with pm.Model(name=f'{n_spots}') as model:
                f0 = pm.HalfNormal("f0", sigma=1)
                spot_model = 1 + f0
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
                lon_lims = 2 * np.pi * np.arange(n_spots + 1) / n_spots

                for spot_ind in range(n_spots):
                    if partition_lon:
                        lon_lower = lon_lims[spot_ind]
                        lon_upper = lon_lims[spot_ind + 1]
                        mu = 0.5 * (lon_lims[spot_ind] +
                                    lon_lims[spot_ind + 1])
                        sigma = 0.5 * (lon_lims[spot_ind + 1] -
                                       lon_lims[spot_ind])

                        lon = pm.TruncatedNormal(f"lon_{spot_ind}",
                                                 lower=lon_lower,
                                                 upper=lon_upper,
                                                 mu=mu,
                                                 sigma=sigma)
                    else:
                        lon = pm.Uniform(f"lon_{spot_ind}",
                                         lower=0,
                                         upper=2*np.pi,
                                         mu=mu,
                                         sigma=sigma)
                    lat = pm.TruncatedNormal(f"lat_{spot_ind}",
                                             lower=np.radians(latitude_cutoff),
                                             upper=np.radians(
                                                 180 - latitude_cutoff),
                                             mu=np.pi / 2,
                                             sigma=np.pi / 2)
                    rspot = pm.HalfNormal(f"R_spot_{spot_ind}",
                                          sigma=0.1)

                    spot_period = eq_period / (
                                1 - shear * pm.math.sin(lat - np.pi / 2) ** 2)
                    phi = 2 * np.pi / spot_period * (self.lc.time[self.mask][::self.skip_n_points] -
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
                        contrast = pm.TruncatedNormal(f'contrast_{spot_ind}',
                                                      lower=0.01,
                                                      upper=0.99,
                                                      mu=contrast,
                                                      sigma=0.5)
                    spot_model -= rspot ** 2 * (1 - contrast) * pm.math.where(
                        spot_position_z > 0, pm.math.sqrt(1 - rsq), 0)

                pm.Normal("obs", mu=spot_model,
                          sigma=scale_error *
                                self.lc.flux_err[self.mask][::self.skip_n_points],
                          observed=self.lc.flux[self.mask][::self.skip_n_points])
        self.model = model
        return self.model

    def _check_model(self):
        """
        Check that a model instance exists on this object
        """
        if self.model is None:
            raise ValueError('Must first call `Model.construct_model` before '
                             'specifying a sampler.')

    def sample_smc(self, draws, random_seed=42, parallel=True, cores=1,
                   **kwargs):
        """
        Sample the posterior distribution of the model given the data using
        Sequential Monte Carlo.

        Parameters
        ----------
        draws : int
        random_seed : int
        parallel : bool
        cores : int

        Returns
        -------
        trace : `~pymc.MultiTrace`
        """
        self._check_model()
        with DisableLogger(self.verbose):
            with self.model:
                trace = sample_smc(draws, random_seed=random_seed,
                                   parallel=parallel, cores=cores, **kwargs)
        return trace

    def sample_nuts(self, trace_smc, draws, cores=64,
                    target_accept=0.99, **kwargs):
        """
        Sample the posterior distribution of the model given the data using
        the No U-Turn Sampler.

        Parameters
        ----------
        draws : int
        start : `~pymc.MultiTrace` or None
        cores : int
        target_accept : float

        Returns
        -------
        trace : `~pymc.MultiTrace`

        """
        self._check_model()
        with DisableLogger(self.verbose):
            with self.model:
                trace = pm.sample(draws,
                                  start=trace_smc.point(-1), cores=cores,
                                  target_accept=target_accept, **kwargs)
                summary = pm.summary(trace)

        return trace, summary
