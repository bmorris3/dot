"""
conda install mkl-service
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dot import Model, ab_dor_example_lc
from dot.plots import gp_from_posterior
import pymc3 as pm
import multiprocessing as mp

# dot parameters
rotation_period = 0.5  # Rotation period in units of light curve time interval
n_spots = 4         # Number of starspots
results_dir = 'test-go'  # Save results here
draws_nuts = 100    # Number of draws from the NUTS
tune = 100          # Tuning steps (default: >1000)
cores = 2           # This controls the NUTS, for SMC we recommend cores=4.
skip_n_points = 5   # skip every n photometric measurements
rho_factor = 0.5    # length scale of GP in units of the rotation period
scale_errors = 3    # scale up the uncertainties by this factor
contrast = 0.4      # If `None` allow to float, else fix it
partition_lon = True  # Bound the longitudes explored by each spot
verbose = True      # Allow PyMC3 logger to print to stdout

# Fetch example light curve from the package:
lc = ab_dor_example_lc()
lc.time -= lc.time.mean()       # Remove the mean from the time and median from
lc.flux -= np.median(lc.flux)   # for efficient/stable GP regression

min_time = lc.time.min()        # Minimum time to fit
max_time = lc.time.min() + 2    # Maximum time to fit

if __name__ == '__main__':
    # If there isn't already a results directory, create one:
    if not os.path.exists(results_dir):
        print('Constructing model...')
        # Construct an instance of `Model` (this is surprisingly expensive)
        m = Model(
            light_curve=lc,
            rotation_period=rotation_period,
            n_spots=n_spots,
            contrast=contrast,
            skip_n_points=skip_n_points,
            min_time=min_time,
            max_time=max_time,
            rho_factor=rho_factor,
            scale_errors=scale_errors,
            partition_lon=partition_lon,
            verbose=verbose
        )

        with m:
            map_soln = pm.find_MAP()

        print('Running NUTS...')
        with m:
            trace_nuts = pm.sample(start=map_soln, draws=draws_nuts,
                                   cores=cores, tune=tune,
                                   init='jitter+adapt_full',
                                   mp_ctx=mp.get_context("fork"))

        gp_from_posterior(m, trace_nuts)
        plt.show()