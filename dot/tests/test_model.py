from ..model import Model
from ..io import ab_dor_example_lc


def test_model_constructor():
    """
    python -c "from dot.tests.test_model import test_model_MAP as f; f()"
    """
    lc = ab_dor_example_lc()

    # Test that we can construct a Model instance
    Model(lc, rotation_period=0.5, n_spots=2, scale_errors=5,
          max_time=lc.time.min()+2, contrast=0.2)
