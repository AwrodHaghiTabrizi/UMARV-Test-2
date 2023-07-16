import dropbox
import os
import sys
import time
from getpass import getpass

def print_environment_variables():
    print("Environment variables:")
    print(f"MODEL_ID: {os.environ['MODEL_ID']}")
    print(f"MODEL_DIR: {os.environ['MODEL_DIR']}")
    print(f"ROOT_DIR: {os.environ['ROOT_DIR']}")
    print(f"REPO_DIR: {os.environ['REPO_DIR']}")
    print(f"ENVIRONMENT: {os.environ['ENVIRONMENT']}")

# # Insert model_id here
# model_id = ""

# model_dir = f"{repo_dir}/models/model_{model_id}"

sys.path.append(f"{os.getenv('REPO_DIR')}/src")
from helpers import *

sys.path.append(f"{os.getenv('MODEL_DIR')}/src")
from methods import *
from architecture import *

def download_datasets_from_dropbox(
    dbx_access_token = None, use_thread = False, datasets = None,
    include_all_datasets = False, include_unity_datasets = False,
    include_real_world_datasets = False, include_benchmarks = False ):

    if dbx_access_token is None:
        dbx_access_token = getpass("Enter your DropBox access token: ")
    dbx = dropbox.Dropbox(dbx_access_token)
    
    dbx_datasets_dir = '/UMARV/ML/datasets'

    # if environment == "colab":
    #     repo_dir = "/content/UMARV-Test-2"
    #     destination_dir = f"/content/datasets/{dataset_dir}"
    # elif environment == "local":
    #     pass
    # elif environment == "lambda":
    #     pass

    if datasets is not None:
        dataset_dirs = datasets

    else:
        
        if not (include_all_datasets or include_unity_datasets or include_real_world_datasets or include_benchmarks):
            dataset_dirs = ["sample/sample_dataset"]
    
        else:

            dataset_dirs = []

            for dataset_category in ["unity", "real_world", "benchmarks"]:

                # Check to skip the category if not requested
                if  not include_all_datasets and (
                    (dataset_category == "unity" and not include_unity_datasets)
                    or (dataset_category == "real_world" and not include_real_world_datasets)
                    or (dataset_category == "benchmarks" and not include_benchmarks) ):
                    continue

                # Collect dataset directories in DropBox for category
                dataset_category_dir = f"{dbx_datasets_dir}/{dataset_category}"
                result = dbx.files_list_folder(dataset_category_dir)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FolderMetadata):
                        found_dataset_dir = entry.path_display.lower().replace(dbx_datasets_dir.lower(),"")
                        dataset_dirs.append(found_dataset_dir)
                while result.has_more:
                    result = dbx.files_list_folder_continue(result.cursor)
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FolderMetadata):
                            found_dataset_dir = entry.path_display.lower().replace(dbx_datasets_dir.lower(),"")
                            dataset_dirs.append(found_dataset_dir)

    # Download datasets
    for dataset_dir in dataset_dirs:

        copy_directory_from_dropbox(
            source_dir = f"{dbx_datasets_dir}/{dataset_dir}",
            destination_dir = f"{os.getenv('ROOT_DIR')}/datasets/{dataset_dir}",
            dbx_access_token = dbx_access_token,
            use_thread = use_thread
        )