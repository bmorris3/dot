import pickle

__all__ = ['save_trace', 'load_trace']


def save_trace(name, trace):
    """
    Save a trace to a pickle file.
    """
    with open(name, 'wb') as buff:
        pickle.dump(trace, buff)


def load_trace(name):
    """
    Load a trace from a pickle file
    """
    with open(name, 'rb') as buff:
        trace = pickle.load(buff)
    return trace
