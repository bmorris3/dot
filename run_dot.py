"""
conda install mkl-service
"""
import os
import math
import json
import wandb
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dot import Model, save_results, load_results, ab_dor_example_lc
from dot import load_light_curve
from dot.plots import corner, posterior_shear,  posterior_predictive, movie

import warnings
warnings.filterwarnings("ignore")

# Adapted from https://github.com/bmorris3/dot/blob/master/example.py

hdf5_archive_disk = '/110k_pdcsap/'
hdf5_index_path = '110k_rotation_mcquillan_pdcsap_smooth_index_0724.csv'

def run_dot(args):

    # --------Extracting arguments--------

    # Data Params
    target = args.target
    mission = args.mission
    sector = args.sector
    # Fit Params
    n_spots = args.n_spots
    skip_n_points = args.skip_n_points  # skip every n photometric measurements
    limit_duration = args.limit_duration  # days
    n_intervals = args.n_intervals
    # SMC Params
    draws_smc = args.draws_smc
    # NUTS Params
    draws_nuts = args.draws_nuts
    tune = args.tune
    cores = args.cores
    # Logging Params
    base_dir = args.base_dir
    bucket_dir = args.bucket_dir

    # --------Getting the data--------

    # Get Kepler data locally or from lightkurve
    if mission=="Kepler":
        # # Convert into integer to load data locally
        # try:
        #     target = int(target)
        # except:
        #     pass
        print(f'Loading light curve for KIC {target}...')
        lc = load_light_curve(target)
        # Extract rotation periods
        try:
            rotation_period = load_rotation_period(target)
        except:
            print(f'Target KIC {target} does not have a McQuillan period.')
            rotation_period = args.rotation_period

    else:
        # Get data from lightkurve
        print(f'Loading light curve for {target} from lightkurve...')
        from lightkurve import search_lightcurvefile
        lc = search_lightcurvefile(
            target=target,
            mission=mission,
            sector=sector
        ).download_all().PDCSAP_FLUX.stitch()

    # --------Setting up model--------

    min_time = lc.time.min()
    max_time = lc.time.max()
    n_poss_intervals = math.floor((max_time-min_time)/limit_duration)

    if n_intervals is None: # Run all intervals
        n_intervals = n_poss_intervals

    print(f'Setting up {n_intervals} intervals to fit...')
    for n in tqdm(range(1, n_intervals+1)):

        min_time_interval = min_time + (n-1)*limit_duration
        max_time_interval = min_time + n*limit_duration
        filename = str(mission) + '_' + str(target) + '_' + str(int(min_time_interval)) + '-' + str(int(max_time_interval))

        print('Constructing model...')

        # Construct an instance of `Model` (this is surprisingly expensive)
        m = Model(
            light_curve=lc,
            rotation_period=rotation_period,
            n_spots=n_spots,
            skip_n_points=skip_n_points,
            min_time=min_time_interval,
            max_time=max_time_interval,
        )

        print('Running SMC...')
        trace_smc = m.sample_smc(draws=draws_smc)

        print('Running NUTS...')
        trace_nuts, summary = m.sample_nuts(trace_smc, draws=draws_nuts,
                                            cores=cores, tune=tune)

        # --------Logging results--------                             

        results_dir = os.path.join(base_dir, filename, "")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        print(f'Saving results to {results_dir}...')
        save_results(results_dir, m, trace_nuts, summary)

        with open(f'{results_dir}/arguments.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print('Logging results to Weights & Biases')

        # Initialize Weights & Biases
        wandb.init(entity="star-spots", project=args.wandb_project, name=filename)
        wandb.config.update(args)

        posterior_shear(m, trace_nuts)
        wandb.log({f'Posterior_Shear_{n_spots}_Spots': wandb.Image(plt)})
        posterior_predictive(m, trace_nuts)
        wandb.log({f'Light_curve_{n_spots}_Spots': wandb.Image(plt)})
        corner(trace_nuts)
        wandb.log({f'Corner_plot_{n_spots}_Spots': wandb.Image(plt)})
        #movie(results_dir, m, trace_nuts, xsize=50)

        if args.copy_to_bucket:
            print(f'Copying results to GCP bucket {results_dir}...')
            ## Change this if needed
            BUCKET = bucket_dir
            subprocess.call(['gsutil', '-m', 'cp', '-r', results_dir, f'gs://{BUCKET}/'])


if __name__ == '__main__':

    import argparse

    # import the parser

    parser = argparse.ArgumentParser()

    # =================================
    # DATA PARAMS
    # =================================

    parser.add_argument(
       "--target", type=str, default='757099', help="Target Star", # McQuillan star
    #    "--target", type=str, default='757076', help="Target Star", # No McQuillan star
    #    "--target", type=str, default='AB Dor', help="Target Star",

    )
    parser.add_argument(
        "--mission", type=str, default='Kepler', help="Mission",
    #    "--mission", type=str, default='Tess', help="Mission",
    )
    parser.add_argument(
        "--sector", type=int, default=None, help="Quarter or Sector",
    )
    parser.add_argument(
        "--rotation-period", type=float, default=0.5, help="Stellar rotation period",
    )
    # =================================
    # FIT PARAMS
    # =================================
    parser.add_argument(
        "--n-spots", type=float, default=2, help="Number of spots to fit",
    )
    parser.add_argument(
        "--skip-n-points", type=int, default=5, help="Skip every n photometric measurement",
    )
    parser.add_argument(
        "--limit-duration", type=int, default=2, help="Length of interval to fit in days",
    )
    parser.add_argument(
        "--n-intervals", type=int, default=2, help="Number of time intervals to fit",
    )
    # =================================
    # SMC PARAMS
    # =================================
    parser.add_argument(
        "--draws-smc", type=int, default=10, help="Number of SMC draws",
    )
    # =================================
    # NUTS PARAMS
    # =================================
    parser.add_argument(
        "--draws-nuts", type=int, default=10, help="Number of NUTS draws",
    )
    parser.add_argument(
        "--tune", type=int, default=10, help="Tune",
    )
    parser.add_argument(
        "--cores", type=int, default=4, help="Cores to run sampling on",
    )
    # =================================
    # LOGGING PARAMS
    # =================================
    parser.add_argument(
        "--base-dir", type=str, default='test-example', help="Directory to save results to",
    )
    parser.add_argument(
        "--bucket-dir", type=str, default='test-example-discard', help="GCP bucket to save results to",
    )
    parser.add_argument(
        "--copy-to-bucket", type=bool, default=False, help="Whether or not to save results to GCP bucket",
    )
    parser.add_argument(
        "--wandb-project", type=str, default='test-discard', help="Weights and Biases project name",
    )

    parser.add_argument("--logging-level", type=str, default="info")
    args = parser.parse_args()

    run_dot(args)