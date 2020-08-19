"""
conda install mkl-service
"""

import os
import matplotlib.pyplot as plt
from lightkurve import search_lightcurvefile
from dot import Model, save_results, load_results, corner, posterior_predictive

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


if __name__ == '__main__':
    # Get the light curve from lightkurve
    print(f'Load the light curve for {target} from lightkurve...')
    lcf = search_lightcurvefile(
        target=target,
        mission=mission,
        sector=sector
    ).download_all()
    lc = lcf.PDCSAP_FLUX.stitch()

    # If there isn't already a results directory, create one:
    if not os.path.exists(results_dir):
        print('Constructing model...')
        # Construct an instance of `Model` (this is surprisingly expensive)
        m = Model(
            light_curve=lc,
            rotation_period=rotation_period,
            n_spots=n_spots,
            skip_n_points=50
        )

        print('Running SMC...')
        trace_smc = m.sample_smc(draws=draws_smc)

        print('Running NUTS...')
        trace_nuts = m.sample_nuts(trace_smc, draws=draws_nuts,
                                   cores=cores, tune=tune)

        os.mkdir(results_dir)
        save_results(results_dir, m, trace_nuts)

    # Otherwise load the previously computed model, results:
    else:
        m, trace_nuts = load_results(results_dir)

    corner(trace_nuts)
    posterior_predictive(m, trace_nuts)
    plt.show()

print('Done.')
