import dropbox
import sys
import time
from getpass import getpass

model_id = "2xzlugfa"

repo_dir = "/content/UMARV-Test-2"
model_dir = f"{repo_dir}/models/model_{model_id}"

sys.path.append(f'{repo_dir}/src')
from helpers import *

sys.path.append(f'{model_dir}/src')
from methods import *
from architecture import *

def download_datasets_from_dropbox(
    dbx_access_token = None, use_thread = False,
    datasets = None, all_datasets = False,
    unity_datasets = False, real_world_datasets = False, benchmarks = False ):

    if dbx_access_token is None:
        dbx_access_token = getpass("Enter your DropBox access token: ")
    dbx = dropbox.Dropbox(dbx_access_token)
    
    dbx_datasets_dir = '/UMARV/ML/datasets'

    if datasets is not None:
        dataset_dirs = datasets

    else:

        if all_datasets:
            unity_datasets = True
            real_world_datasets = True
            benchmarks = True
        
        elif (not unity_datasets) and (not real_world_datasets) and (not benchmarks):
            dataset_dirs = ["sample/sample_dataset"]
    
        else:

            dataset_dirs = []

            for dataset_category in ["unity", "real_world", "benchmarks"]:
                if dataset_category == "unity" and not unity_datasets:
                    continue
                if dataset_category == "real_world" and not real_world_datasets:
                    continue
                if dataset_category == "benchmarks" and not benchmarks:
                    continue

            unity_folder_path = f"{dbx_datasets_dir}/{dataset_category}"
            result = dbx.files_list_folder(unity_folder_path)
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FolderMetadata):
                    dataset_dirs.append(f"{dataset_category}/{entry.path_display}")
            while result.has_more:
                result = dbx.files_list_folder_continue(result.cursor)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FolderMetadata):
                        dataset_dirs.append(f"{dataset_category}/{entry.path_display}")

    for dataset_dir in dataset_dirs:

        copy_directory_from_dropbox(
            source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
            destination_dir = f"/content/datasets_fast/{dataset_dir}",
            dbx_access_token = dbx_access_token,
            use_thread = use_thread
        )