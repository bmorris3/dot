import os
import pickle
import numpy as np
import pandas as pd
from lightkurve import LightCurve

__all__ = ['save_results', 'load_results', 'ab_dor_example_lc']


def save_results(name, model, trace, summary):
    """
    Save a trace to a pickle file.

    Parameters
    ----------
    name : str
        Path to the directory where results will be saved
    model : `~dot.Model`
        Model object
    trace : `~pymc3.MultiTrace`
        Trace from SMC/NUTS
    summary : `pandas.DataFrame`
        Dataframe containing summary statistics
    """
    with open(os.path.join(name, 'model.pkl'), 'wb') as buff:
        pickle.dump(model, buff)

    with open(os.path.join(name, 'trace_nuts.pkl'), 'wb') as buff:
        pickle.dump(trace, buff)

    summary.to_pickle(os.path.join(name, 'summary.pkl'))


def load_results(name):
    """
    Load a trace from a pickle file

    Parameters
    ----------
    name : str
        Path to the directory where results were saved

    Returns
    -------
    model : `~dot.Model`
        Model object
    trace : `~pymc3.MultiTrace`
        Trace from SMC/NUTS
    summary : `pandas.DataFrame`
        Dataframe containing summary statistics
    """
    with open(os.path.join(name, 'model.pkl'), 'rb') as buff:
        model = pickle.load(buff)

    with open(os.path.join(name, 'trace_nuts.pkl'), 'rb') as buff:
        trace_nuts = pickle.load(buff)

    summary = pd.read_pickle(os.path.join(name, 'summary.pkl'))

    return model, trace_nuts, summary


def ab_dor_example_lc(path=None):
    """
    Return a `~lightkurve.LightCurve` object with the first few TESS
    observations of the rapidly-rotating, spotted star AB Doradus.

    Parameters
    ----------
    path : None or str
        Path to the file to load (optional)

    Returns
    -------
    lc : `~lightkurve.LightCurve`
        Light curve of AB Doradus
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data',
                            'abdor_lc_example.npy')
    return LightCurve(*np.load(path))
