import os
import pickle
import numpy as np
import pandas as pd
from lightkurve import LightCurve, search_lightcurvefile
import h5py


__all__ = ['save_results', 'load_results', 'ab_dor_example_lc',
           'load_light_curve', 'load_rotation_period']

hdf5_archive_disk = '/110k_pdcsap/'
hdf5_index_path = '110k_rotation_mcquillan_pdcsap_smooth_index_0724.csv'


def save_results(name, model, trace, summary):
    """
    Save a trace to a pickle file.

    Parameters
    ----------
    name : str
        Path to the directory where results will be saved
    model : `~dot.Model`
        Model object
    trace : `~pymc3.backends.base.MultiTrace`
        Trace from SMC/NUTS
    summary : `~pandas.DataFrame`
        Dataframe containing summary statistics
    """
    with open(os.path.join(name, 'model.pkl'), 'wb') as buff:
        pickle.dump(model, buff)

    with open(os.path.join(name, 'trace_nuts.pkl'), 'wb') as buff:
        pickle.dump(trace, buff)

    summary.to_pickle(os.path.join(name, 'summary.pkl'))


def load_results(name):
    """
    Load a trace from a pickle file

    Parameters
    ----------
    name : str
        Path to the directory where results were saved

    Returns
    -------
    model : `~dot.Model`
        Model object
    trace : `~pymc3.backends.base.MultiTrace`
        Trace from SMC/NUTS
    summary : `~pandas.DataFrame`
        Dataframe containing summary statistics
    """
    with open(os.path.join(name, 'model.pkl'), 'rb') as buff:
        model = pickle.load(buff)

    with open(os.path.join(name, 'trace_nuts.pkl'), 'rb') as buff:
        trace_nuts = pickle.load(buff)

    summary = pd.read_pickle(os.path.join(name, 'summary.pkl'))

    return model, trace_nuts, summary


def ab_dor_example_lc(path=None):
    """
    Return a `~lightkurve.lightcurve.LightCurve` object with the first few TESS
    observations of the rapidly-rotating, spotted star AB Doradus.

    Parameters
    ----------
    path : None or str
        Path to the file to load (optional)

    Returns
    -------
    lc : `~lightkurve.lightcurve.LightCurve`
        Light curve of AB Doradus
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data',
                            'abdor_lc_example.npy')
    return LightCurve(*np.load(path))


def load_light_curve(kic):
    """
    Load a light curve from its KIC number.

    Parameters
    ----------
    kic : int
        Kepler Input Catalog number

    Returns
    -------
    lc : `~lightkurve.lightcurve.LightCurve`
        PDCSAP light curve object
    """
    on_gcp = os.path.exists(os.path.join(hdf5_archive_disk,
                                         hdf5_index_path))
    # If on Google Cloud
    if on_gcp:
        try:
            # Try to load from HDF5
            return load_from_hdf5(kic)
        except ValueError:
            pass
    # If not on Google Cloud or the loader fails, use lightkurve
    return download_from_lightkurve(kic)


def load_from_hdf5(kic, data_path=None, index_file=None):
    """
    Load a light curve from the HDF5 archive on Google Cloud Platform.
    """
    if data_path is None:
        data_path = hdf5_archive_disk
    if index_file is None:
        index_file = hdf5_index_path
    index_path = os.path.join(data_path, index_file)
    stars_index = pd.read_csv(index_path)

    # Convert into integer to load data locally
    try:
        kic = int(kic)
    except:
        pass

    star_path_list = stars_index.loc[stars_index["KIC"] == kic]["filepath"].values
    if len(star_path_list) == 0:
        raise ValueError(f'Target KIC {kic} not in database.')
    star_path = star_path_list[0]

    with h5py.File(os.path.join(data_path, star_path), "r") as f:
        time = np.array(f[str(kic)].get("PDC_SAP_time"))
        flux = np.array(f[str(kic)].get("PDC_SAP_flux"))
        flux_err = np.array(f[str(kic)].get("PDC_SAP_flux_err"))

    pdcsap = LightCurve(
        time=time, flux=flux, flux_err=flux_err, targetid=kic
    )

    return pdcsap


def download_from_lightkurve(kic):
    """
    Download a light curve from lightkurve
    """
    lc = search_lightcurvefile(
        target=f"KIC {kic}",
        mission='Kepler'
    ).download_all().PDCSAP_FLUX.stitch()
    return lc


def load_rotation_period(kic, data_path=None, index_file=None):
    """
    Extract McQuillan rotation period from KIC number on google cloud platform.

    Parameters
    ----------
    kic : int
        Kepler Input Catalog number
    data_path : str
        Path to locally stored data
    index_file : str
        Index csv containing McQuillan rotation periods
    Returns
    -------
    prot : float
        McQuillan rotation period
    """

    if data_path is None:
        data_path = hdf5_archive_disk
    if index_file is None:
        index_file = hdf5_index_path
    index_path = os.path.join(data_path, index_file)
    stars_index = pd.read_csv(index_path)
    star_prot_list = stars_index.loc[stars_index["KIC"] == kic]["PRot"].values
    if not math.isfinite(star_prot_list[0]):
        raise ValueError(f'Target KIC {kic} does not have a McQuillan period.')
    else:
        print(f'Using McQuillan period for KIC {kic}.')
        star_prot = star_prot_list[0]
        return star_prot