import numpy as np
from lightkurve import LightCurve
import pymc3 as pm
import pytest

from ..model import Model

np.random.seed(42)

# Create a grid of spot longitudes, latitudes, radii and stellar inclinations
# for iterating over in the tests below...
n_trials = 10
lon_grid = 2 * np.pi * np.random.rand(n_trials, 2)
lat_grid = 0.5 * np.pi * (np.random.rand(n_trials, 2) - 0.5)
rad_grid = 0.05 + np.max([0.01 * np.random.randn(n_trials, 2),
                          np.zeros((n_trials, 2))], axis=0)
inc_grid = 90 * np.random.rand(n_trials)


# This decorator at the top allows us to iterate over the various pairs
# of spot lons, lats, rads, and stellar inclinations. We compare the result
# of the chi^2 computation with the 1e-6 and make sure it's smaller, i.e.
# we test for excellent agreement between fleck and dot.
@pytest.mark.parametrize("test_input,expected",
                         [((lons, lats, rads, inc), -6)
                          for lons, lats, rads, inc in
                          zip(lon_grid, lat_grid, rad_grid, inc_grid)])
def test_against_fleck(test_input, expected):
    from fleck import Star
    import astropy.units as u

    # Unpack the input
    lons, lats, rads, inc = test_input

    # Create a time axis and compute the light curve on that axis
    times = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    s = Star(spot_contrast=0.3, u_ld=[0, 0], n_phases=len(times),
             rotation_period=2 * np.pi)
    # Note: the longitudes in fleck and dot are different by a constant
    # offset of -pi/2
    lc = s.light_curve((lons - np.pi / 2).T[:, None] * u.rad,
                       lats.T[:, None] * u.rad,
                       rads.T[:, None], inc * u.deg,
                       times=times, time_ref=0)[:, 0]

    # Define (arbitrary) error for the light curve
    errs = np.std(lc) * np.ones_like(lc) / 10

    m = Model(
        light_curve=LightCurve(times, lc - np.median(lc), errs),
        rotation_period=2 * np.pi,
        n_spots=2,
        partition_lon=False,
        contrast=0.3
    )

    # Create a starting point for the dot model in the correctly transformed
    # notation from fleck to dot
    start = {
        "dot_R_spot": np.array([rads]),
        "dot_lat": np.array([np.pi / 2 - lats]),
        "dot_lon": np.array([lons]),
        "dot_comp_inc": np.radians(90 - inc),
        "dot_shear": 1e-4,
        "dot_P_eq": 2 * np.pi,
        "dot_f0": m.lc.flux.max()
    }

    # Need to call this to validate ``start``
    pm.util.update_start_vals(start, m.pymc_model.test_point, m.pymc_model)

    # the fit is not normalized to its median like the input light curve is
    fit, var = m(start)
    # ...so we normalize it before we compare:
    fit -= np.median(fit)

    # Compute the chi^2. This should be super small if both models agree!
    chi2 = np.sum((m.lc.flux - fit) ** 2)
    assert np.log10(chi2) < expected
