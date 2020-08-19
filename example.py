import os
import matplotlib.pyplot as plt
from lightkurve import search_lightcurvefile
from dot import Model, save_trace, load_trace, corner, posterior_predictive

rotation_period = 0.5
n_spots = 5
trace_path = 'example.pkl'

if __name__ == '__main__':

    lcf = search_lightcurvefile("HD 197890", mission='TESS').download_all()
    lc = lcf.PDCSAP_FLUX.stitch()

    m = Model(
        light_curve=lc,
        rotation_period=rotation_period,
        n_spots=n_spots,
        skip_n_points=50
    )

    if not os.path.exists(trace_path):
        trace_smc = m.sample_smc(draws=50)
        trace_nuts = m.sample_nuts(draws=100, tune=100, cores=4)

        save_trace(trace_path, trace_nuts)

    else:
        trace_nuts = load_trace(trace_path)

    corner(trace_nuts)
    posterior_predictive(m, trace_nuts)
    plt.show()