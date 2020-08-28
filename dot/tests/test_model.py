import numpy as np

from ..model import Model
from ..io import ab_dor_example_lc

np.random.seed(42)


def test_model_optimizer():

    lc = ab_dor_example_lc()

    lc.time -= np.median(lc.time)
    lc.flux -= np.median(lc.flux)
    # Test that we can construct a Model instance
    m = Model(lc, rotation_period=0.5, n_spots=2, scale_errors=1,
              max_time=lc.time.min()+2, contrast=0.2)

    # Run the optimizer:
    map_soln = m.optimize()

    # Preliminary values here (these aren't necessarily the true best-fit, just
    # what we expect the `optimize` function to return)
    eps = 1e-3
    assert abs(map_soln['dot_P_eq'] - 0.40419959) < eps
    assert abs(map_soln['dot_f0'] - 0.03531389) < eps
    assert abs(map_soln['dot_comp_inc'] - 0.08830853) < eps
