import numpy as np

from ..model import Model
from ..io import ab_dor_example_lc

def test_model_optimizer():

    lc = ab_dor_example_lc()

    # Test that we can construct a Model instance
    m = Model(lc, rotation_period=0.5, n_spots=2, scale_errors=5,
              max_time=lc.time.min()+2, contrast=0.2)

    # Run the optimizer:
    map_soln = m.optimize()

    # Preliminary values here (these aren't necessarily the true best-fit, just
    # what we expect the `optimize` function to return)
    np.testing.assert_allclose(map_soln['dot_P_eq'], 0.45715124)
    np.testing.assert_allclose(map_soln['dot_shear'], 0.22016086)
    np.testing.assert_allclose(map_soln['dot_P_eq'], 0.45715124)
