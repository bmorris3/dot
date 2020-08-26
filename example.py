"""
conda install mkl-service
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dot import Model, save_results, load_results, ab_dor_example_lc
from dot.plots import gp_from_posterior

# dot parameters
rotation_period = 0.5  # Rotation period in units of light curve time interval
n_spots = 4         # Number of starspots
results_dir = 'test-go'  # Save results here
draws_smc = 100     # Number of draws from the NUTS
draws_nuts = 100    # Number of draws from the NUTS
tune = 100          # Tuning steps (default: >1000)
cores = 4           # This controls the NUTS, for SMC we recommend cores=4.
skip_n_points = 5   # skip every n photometric measurements
rho_factor = 0.2    # length scale of GP in units of the rotation period
scale_errors = 15   # scale up the uncertainties by this factor
contrast = None     # If `None` allow to float, else fix it
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
        print('Running SMC...')
        trace_smc = m.sample_smc(draws=draws_smc, parallel=True, cores=4)

        print('Running NUTS...')
        trace_nuts, summary = m.sample_nuts(trace_smc, draws=draws_nuts,
                                            cores=cores, tune=tune)

        os.mkdir(results_dir)
        save_results(results_dir, m, trace_nuts, summary)

    # Otherwise load the previously computed model, results:
    else:
        m, trace_nuts, summary = load_results(results_dir)

    # Times at which to compute the mean model + GP
    xnew = np.linspace(min_time, max_time, 1000)

    # Plot the resulting fit to the light curve with a GP:
    plt.figure(figsize=(20, 5))
    gp_from_posterior(m, trace_nuts, xnew,
                      path=os.path.join(results_dir, 'gp.png'))
    plt.show()
