.. _plots:

*****
Plots
*****

There are a few built-in visualization tools within `dot`. Suppose you just
finished the :ref:`getting-started` tutorial. Here's a quick
example script for the `~dot.plots.movie` function which produces a movie like
the one below:

.. code-block:: python

    from dot import load_results
    from dot.plots import movie

    m, trace_nuts, summary = load_results(results_dir)

    posterior_shear(m, trace_nuts)
    movie(results_dir, m, trace_nuts, xsize=250)

.. raw:: html

    <div style="position: relative; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe width="700" height="200" src="https://www.youtube.com/embed/QRm-WarnBTs" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"></iframe>
    </div>

Neat!
