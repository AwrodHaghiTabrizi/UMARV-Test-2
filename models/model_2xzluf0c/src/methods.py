import dropbox
import sys
import time
from getpass import getpass

model_id = "2xzluf0c"

repo_dir = "/content/UMARV-Test-2"
model_dir = f"{repo_dir}/models/model_{model_id}"

sys.path.append(f'{repo_dir}/src')
from helpers import *

sys.path.append(f'{model_dir}/src')
from methods import *
from architecture import *

def download_datasets_from_dropbox(
    dbx_access_token = None, datasets = None,
    all_datasets = False, unity_datasets = False,
    real_world_datasets = False, benchmarks = False ):

    if dbx_access_token is None:
        dbx_access_token = getpass("Enter your DropBox access token: ")
    
    dbx_datasets_dir = '/UMARV/ML/datasets'

    if datasets is not None:
        dataset_dirs = datasets

    else:

        if all_datasets:
            unity_datasets = True
            real_world_datasets = True
            benchmarks = True
        
        elif (not unity_datasets and not real_world_datasets and not benchmarks):
            dataset_dirs = ["sample/sample_dataset"]
    
        else:
            dataset_dirs = []
            if unity_datasets:
                pass
            if real_world_datasets:
                pass
            if benchmarks:
                pass

    print(f"{dataset_dirs=}")

    start_time = time.time()

    for dataset_dir in dataset_dirs:

        copy_directory_from_dropbox(
            source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
            destination_dir = f"/content/datasets_fast/{dataset_dir}",
            dbx_access_token = dbx_access_token,
            use_thread = True
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Fast Elapsed Time:", elapsed_time, "seconds")

    start_time = time.time()

    copy_directory_from_dropbox(
        source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
        destination_dir = f"/content/datasets_slow/{dataset_dir}",
        dbx_access_token = dbx_access_token,
        use_thread = False
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Slow Elapsed Time:", elapsed_time, "seconds")
