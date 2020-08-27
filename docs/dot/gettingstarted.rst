.. _getting-started:

***************
Getting Started
***************

.. note::

    If you plan to copy and paste the code in this tutorial into a notebook,
    no modifications are needed to run the examples. However, if you run it in
    from a Python script (.py file), you need to add the following line
    ``if __name__ == "__main__":`` to the top of the code or else you'll hit a
    `RuntimeError`.

Load a light curve
------------------

`dot` uses `~lightkurve.lightcurve.LightCurve` objects to handle light curves.
For quick and easy access to an example light curve, we include the light curve
of AB Doradus in the package, which you can access with:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from dot import ab_dor_example_lc

    min_time = 0
    max_time = 2

    lc = ab_dor_example_lc()

    # For the Gaussian process regression later on, we'll remove the mean from
    # the time and the median from the flux:
    lc.time -= lc.time.mean()
    lc.flux -= np.median(lc.flux)

    lc.plot()
    plt.xlim([min_time, max_time])


.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from dot import ab_dor_example_lc

    min_time = 0
    max_time = 2

    lc = ab_dor_example_lc()

    # For the Gaussian process regression later on, we'll remove the mean from
    # the time and the median from the flux:
    lc.time -= lc.time.mean()
    lc.flux -= np.median(lc.flux)

    lc.plot()
    plt.xlim([min_time, max_time])

Construct a model
-----------------

Next we construct a `~dot.Model` object to fit to the light curve, which we
initialize with some key parameters particular to this model:

.. code-block:: python

    from dot import Model

    m = Model(
        light_curve=lc,
        rotation_period=0.5,
        n_spots=2,
        skip_n_points=20,
        min_time=min_time,
        max_time=max_time
    )

We've constructed a model light curve which will only compare to every 20th
observation in the ``lc`` object to speed up computation times in this tutorial.
In real observations, you should make ``skip_n_points`` closer to unity.

The first thing we should do is check if our model can approximate the data.
Here's a quick sanity check that our model is defined on the correct bounds,
our errorbar scaling is appropriate, and the number of spots is a good guess:

.. code-block:: python

    m.optimize(plot=True)

.. plot::

    import numpy as np
    from dot import ab_dor_example_lc, Model

    min_time = 0
    max_time = 2

    lc = ab_dor_example_lc()

    # For the Gaussian process regression later on, we'll remove the mean from
    # the time and the median from the flux:
    lc.time -= lc.time.mean()
    lc.flux -= np.median(lc.flux)

    m = Model(
        light_curve=lc,
        rotation_period=0.5,
        n_spots=2,
        skip_n_points=20,
        min_time=min_time,
        max_time=max_time
    )

    map_soln = m.optimize(plot=True)

That fit looks pretty good for an initial guess with no manual-tuning and only
two spots! It looks to me like the model probably has sufficient but not
too much complexity with two spots. Now let's sample the posterior
distributions for the stellar and spot parameters.

Sampling
--------

Next, we can sample the posterior distributions using the
`Sequential Monte Carlo (SMC) <https://en.wikipedia.org/wiki/Particle_filter>`_
sampler in `pymc3 <https://docs.pymc.io>`_ with:

.. code-block:: python

    trace_smc = m.sample_smc(draws=100)

This will give us a quick fit to the light curve while exploring parameter
degeneracies that were ignored in the first optimization step.

Next we'll sample the posterior distributions using the
`No U-Turn Sampler (NUTS) <https://arxiv.org/abs/1701.02434>`_ implemented by
pymc3:

.. code-block:: python

    trace_nuts, summary = m.sample_nuts(trace_smc, draws=1000, cores=2)

The values for ``draws`` and ``tune`` used above are chosen to produce quick
plots, not to give converged publication-ready results. Always make these
parameters as large as you can tolerate!

Finally, let's plot our results:

.. code-block:: python

    from dot.plots import posterior_predictive

    posterior_predictive(m, trace_nuts, samples=10)
    plt.xlim([min_time, max_time])

.. plot::

    import numpy as np
    from dot import ab_dor_example_lc, Model
    from dot.plots import posterior_predictive
    import matplotlib.pyplot as plt

    min_time = 0
    max_time = 2

    lc = ab_dor_example_lc()

    # For the Gaussian process regression later on, we'll remove the mean from
    # the time and the median from the flux:
    lc.time -= lc.time.mean()
    lc.flux -= np.median(lc.flux)

    m = Model(
        light_curve=lc,
        rotation_period=0.5,
        n_spots=2,
        skip_n_points=20,
        min_time=min_time,
        max_time=max_time
    )

    trace_smc = m.sample_smc(draws=50)
    trace_nuts, summary = m.sample_nuts(trace_smc, draws=10,
                                        cores=2, tune=10)
    fig, ax = posterior_predictive(m, trace_nuts, samples=10)
    ax.set_xlim([min_time, max_time])
    fig.tight_layout()

Look at that, the fit is great! Let's save our model, trace, and summary:

.. code-block:: python

    from dot import save_results

    results_dir = 'example'  # this directory will be created

    save_results(results_dir, m, trace_nuts, summary)

.. warning::

    This tutorial is optimized for producing quick results that can be
    rendered online, and does not fully represent best-practices for using
    `dot`. For example, you should make ``draws`` as large as you can tolerate
    when using dot for science. Ye be warned!