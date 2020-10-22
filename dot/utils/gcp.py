from pathlib import Path
from google.cloud import storage
import pickle


def get_home_dir():
    return str(Path.home())


def save_pkl(data, file) -> None:
    with open(f"{file}.pkl", "wb") as buff:
        pickle.dump(data, buff)
    return None


# def save_summary(summary: pd.DataFrame, file_path: str, name: str = "summary") -> None:

#     summary.to_pickle(os.path.join(file_path, f"{name}.pkl"))
#     return None


def create_folder(directory: str) -> None:
    """Creates directory if doesn't exist"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder '{directory}' Is Already There.")
    else:
        print(f"Folder '{directory}' is created.")


def save_file_from_bucket(bucket_id: str, file_name: str, destination_file_path: str):
    """Saves a file from a bucket
    
    Parameters
    ----------
    bucket_id : str
        the name of the bucket
    file_name : str
        the name of the file in bucket (include the directory)
    destination_file_path : str
        the directory of where you want to save the
        data locally (not including the filename)
    
    Examples
    --------
    
    >>> bucket_id = ...
    >>> file_name = 'path/to/file/and/file.csv'
    >>> dest = 'path/in/bucket/'
    >>> load_file_from_bucket(
        bucket_id=bucket_id,
        file_name=file_name,
        destimation_file_path=dest
    )
    """
    client = storage.Client()

    bucket = client.get_bucket(bucket_id)
    # get blob
    blob = bucket.get_blob(file_name)

    # create directory if needed
    create_folder(destination_file_path)

    # get full path
    destination_file_name = Path(destination_file_path).joinpath(
        file_name.split("/")[-1]
    )

    # download data
    blob.download_to_filename(str(destination_file_name))

    return None


def save_file_to_bucket(bucket_id: str, local_save_file: str, full_bucket_name: str):
    """Saves a file to a bucket
    
    Parameters
    ----------
    bucket_id : str
        the name of the bucket
    local_save_file : str
        the name of the file to save (include the directory)
    destination_file_path : str
        the directory (not including) of the bucket where
        you want to save.
    
    Examples
    --------
    
    >>> bucket_id = ...
    >>> file_name = 'path/to/file/and/file.csv'
    >>> dest = 'path/in/bucket/'
    >>> load_file_from_bucket(
        bucket_id=bucket_id,
        file_name=file_name,
        destimation_file_path=dest
    )
    """
    client = storage.Client()

    bucket = client.get_bucket(bucket_id)

    # get blob
    blob = bucket.blob(full_bucket_name)

    # download data
    blob.upload_from_filename(local_save_file)

    return None

