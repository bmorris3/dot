import os
import pickle

__all__ = ['save_results', 'load_results']


def save_results(name, model, trace):
    """
    Save a trace to a pickle file.
    """
    with open(os.path.join(name, 'model.pkl'), 'wb') as buff:
        pickle.dump(model, buff)
    with open(os.path.join(name, 'trace_nuts.pkl'), 'wb') as buff:
        pickle.dump(trace, buff)


def load_results(name):
    """
    Load a trace from a pickle file
    """
    with open(os.path.join(name, 'model.pkl'), 'rb') as buff:
        model = pickle.load(buff)

    with open(os.path.join(name, 'trace_nuts.pkl'), 'rb') as buff:
        trace_nuts = pickle.load(buff)

    return model, trace_nuts
