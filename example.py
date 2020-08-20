"""
conda install mkl-service
"""

import os
import matplotlib.pyplot as plt
from lightkurve import search_lightcurvefile
from dot import Model, save_results, load_results, ab_dor_example_lc

# lightkurve parameters
target = "AB Dor"
mission = 'TESS'
sector = 1

# dot parameters
rotation_period = 0.5
n_spots = 2
results_dir = 'test-example'
draws_smc = 100
draws_nuts = 100
tune = 100
cores = 4
skip_n_points = 5

if __name__ == '__main__':
    # If there isn't already a results directory, create one:
    if not os.path.exists(results_dir):
        # Get the light curve from lightkurve
        # print(f'Load the light curve for {target} from lightkurve...')
        # lcf = search_lightcurvefile(
        #     target=target,
        #     mission=mission,
        #     sector=sector
        # ).download_all()
        # lc = lcf.PDCSAP_FLUX.stitch()

        # Fetch example light curve from the package:
        lc = ab_dor_example_lc()

        print('Constructing model...')
        # Construct an instance of `Model` (this is surprisingly expensive)
        m = Model(
            light_curve=lc,
            rotation_period=rotation_period,
            n_spots=n_spots,
            skip_n_points=skip_n_points,
            min_time=lc.time.min(),
            max_time=lc.time.min() + 5,
        )

        print('Running SMC...')
        trace_smc = m.sample_smc(draws=draws_smc)

        print('Running NUTS...')
        trace_nuts, summary = m.sample_nuts(trace_smc, draws=draws_nuts,
                                            cores=cores, tune=tune)

        os.mkdir(results_dir)
        save_results(results_dir, m, trace_nuts, summary)

    # Otherwise load the previously computed model, results:
    else:
        m, trace_nuts, summary = load_results(results_dir)

    from dot.plots import posterior_shear, movie

    posterior_shear(m, trace_nuts)
    movie(os.path.join('test-example', 'movie.mp4'), m, trace_nuts, xsize=50)
    plt.show()
