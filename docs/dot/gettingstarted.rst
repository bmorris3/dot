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

    import matplotlib.pyplot as plt
    from dot import ab_dor_example_lc

    min_time=1335
    max_time=1337

    lc = ab_dor_example_lc()
    lc.plot()

    plt.xlim([min_time, max_time])

.. plot::

    import matplotlib.pyplot as plt
    from dot import ab_dor_example_lc

    min_time=1335
    max_time=1337

    lc = ab_dor_example_lc()
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

We can sample using the
`Sequential Monte Carlo (SMC) <https://en.wikipedia.org/wiki/Particle_filter>`_
sampler in `pymc3 <https://docs.pymc.io>`_ with:

.. code-block:: python

    trace_smc = m.sample_smc(draws=100)

This will give us a quick preliminary fit to the light curve. Next we'll sample
the posterior distributions using the
`No U-Turn Sampler (NUTS) <https://arxiv.org/abs/1701.02434>`_ implemented by
pymc3:

.. code-block:: python

    trace_nuts, summary = m.sample_nuts(trace_smc, draws=10,
                                        cores=2, tune=100)

Finally, let's plot our results:

.. code-block:: python

    from dot.plots import posterior_predictive

    posterior_predictive(m, trace_nuts, samples=100)
    plt.xlim([min_time, max_time])

.. plot::

    from dot import ab_dor_example_lc, Model
    from dot.plots import posterior_predictive
    import matplotlib.pyplot as plt

    min_time=1335
    max_time=1337

    lc = ab_dor_example_lc()

    m = Model(
        light_curve=lc,
        rotation_period=0.5,
        n_spots=2,
        skip_n_points=20,
        min_time=min_time,
        max_time=max_time
    )

    trace_smc = m.sample_smc(draws=100)
    trace_nuts, summary = m.sample_nuts(trace_smc, draws=100,
                                        cores=2, tune=100)
    posterior_predictive(m, trace_nuts, samples=10)
    plt.xlim([min_time, max_time])

Let's save our model, trace, and summary:

.. code-block:: python

    from dot import save_results

    results_dir = 'example'  # this directory will be created

    save_results(results_dir, m, trace_nuts, summary)

