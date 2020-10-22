from argparse import ArgumentParser
from lightkurve import search_lightcurvefile
import numpy as np
from pathlib import Path

home = str(Path.home())


# MATPLOTLIB Settings
import matplotlib as mpl
import matplotlib.pyplot as plt
from dot import Model, ab_dor_example_lc
from dot.plots import gp_from_posterior
from dot.utils.gcp import create_folder, save_pkl, save_file_to_bucket
import pymc3 as pm
import multiprocessing as mp
from dot import save_results

# # SEABORN SETTINGS
# import seaborn as sns

# sns.set_context(context="talk", font_scale=0.7)


def main(args):

    # dot parameters
    rotation_period = 0.5  # Rotation period in units of light curve time interval
    n_spots = 4  # Number of starspots
    results_dir = "test-go"  # Save results here
    draws_nuts = 100  # Number of draws from the NUTS
    tune = 100  # Tuning steps (default: >1000)
    cores = 2  # This controls the NUTS, for SMC we recommend cores=4.
    skip_n_points = 5  # skip every n photometric measurements
    rho_factor = 0.5  # length scale of GP in units of the rotation period
    scale_errors = 3  # scale up the uncertainties by this factor
    contrast = 0.4  # If `None` allow to float, else fix it
    partition_lon = True  # Bound the longitudes explored by each spot
    verbose = True  # Allow PyMC3 logger to print to stdout

    # Fetch example light curve from the package:
    lcf = search_lightcurvefile(
        args.target, mission=args.mission, sector=args.sector
    ).download_all()
    lc = lcf.PDCSAP_FLUX.stitch()
    lc.time -= lc.time.mean()  # Remove the mean from the time and median from
    lc.flux -= np.median(lc.flux)  # for efficient/stable GP regression
    min_time = lc.time.min()  # Minimum time to fit
    max_time = lc.time.max()  # Maximum time to fit

    # ==========================================
    # BUILD MODEL
    # ==========================================
    print("Constructing model...")
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
        verbose=verbose,
    )

    print("Running MAP...")
    with m:
        map_soln = pm.find_MAP()

    # ===================================
    # SAVE RESULTS
    # ===================================
    results_dir = str(Path(args.data_path).joinpath("example"))
    bucket_id = "spotdot_results"
    target = args.target.replace(" ", "_")
    bucketpath = Path(f"{args.mission}/{target}_{args.seed}")

    create_folder(results_dir)

    # save model

    save_pkl(m, str(Path(results_dir).joinpath("map_model")))
    print("Uploading Model to bucket...")
    save_file_to_bucket(
        bucket_id,
        str(Path(results_dir).joinpath("map_model.pkl")),
        str(bucketpath.joinpath("map_model.pkl")),
    )

    # save trace
    print("Saving Trace...")
    save_pkl(map_soln, str(Path(results_dir).joinpath("map_trace")))

    # upload to bucket
    print("Uploading Trace to bucket...")
    save_file_to_bucket(
        bucket_id,
        str(Path(results_dir).joinpath("map_trace.pkl")),
        str(bucketpath.joinpath("map_trace.pkl")),
    )


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)

    # ======================
    # Data parameters
    # ======================
    parser.add_argument(
        "--data-path", type=str, default=f"{home}/", help="path to load light curves.",
    )
    parser.add_argument(
        "--target", type=str, default="HD 222259 A", help="path to load light curves.",
    )
    parser.add_argument(
        "--mission", type=str, default="TESS", help="path to load light curves.",
    )
    parser.add_argument(
        "--sector", type=int, default=27, help="path to load light curves.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="path to load light curves.",
    )

    args = parser.parse_args()

    main(args)
