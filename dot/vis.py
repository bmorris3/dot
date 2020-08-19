import matplotlib.pyplot as plt
from corner import corner as dfm_corner
import pymc3 as pm

__all__ = ['corner', 'posterior_predictive']

def corner(trace, **kwargs):
    """
    Make a corner plot
    """
    return dfm_corner(pm.trace_to_dataframe(trace), **kwargs)


def posterior_predictive(model, trace, samples=100, **kwargs):
    """
    Take draws from the posterior predictive given a trace and a model.
    """
    with model.model:
        ppc = pm.sample_posterior_predictive(trace, samples=samples)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    plt.errorbar(model.lc.time,
                 model.lc.flux,
                 model.lc.flux_err,
                 fmt='.', color='k', ecolor='silver')

    plt.plot(model.lc.time[::model.skip_n_points],
             ppc[f'{model.n_spots}_obs'].T,
             color='DodgerBlue', lw=2, alpha=0.1)

    plt.gca().set(xlabel='Time [d]', ylabel='Flux')
    return fig, ax
