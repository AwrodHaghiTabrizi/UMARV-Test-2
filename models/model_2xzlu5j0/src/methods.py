import dropbox
import sys
import time

model_id = "2xzlu5j0"

repo_dir = "/content/UMARV-Test-2"
model_dir = f"{repo_dir}/models/model_{model_id}"

sys.path.append(f'{repo_dir}/src')
from helpers import *

sys.path.append(f'{model_dir}/src')
from methods import *
from architecture import *

def download_datasets_from_dropbox(
    model_id,
    dbx_access_token,
    datasets = [],
    test_dataset = False,
    all_datasets = False,
    unity_datasets = False,
    real_world_datasets = False,
):
    
    start_time = time.time()

    dbx_datasets_dir = '/UMARV/ML/datasets'

    if test_dataset:
        dataset_dirs = ["unity/test_2", "unity/test"]

    for dataset_dir in dataset_dirs:

        copy_directory_from_dropbox_fast(
            source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
            destination_dir = f"/content/datasets_fast/{dataset_dir}",
            dbx_access_token = dbx_access_token
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Fast Elapsed Time:", elapsed_time, "seconds")

    start_time = time.time()

    dbx_datasets_dir = '/UMARV/ML/datasets'

    if test_dataset:
        dataset_dirs = ["unity/test_2", "unity/test"]

    for dataset_dir in dataset_dirs:

        copy_directory_from_dropbox_slow(
            source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
            destination_dir = f"/content/datasets_slow/{dataset_dir}",
            dbx_access_token = dbx_access_token
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Slow Elapsed Time:", elapsed_time, "seconds")
