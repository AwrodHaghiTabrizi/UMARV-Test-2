import dropbox

def download_datasets_from_dropbox(
    dbx_access_token,
    datasets = [],
    test_dataset = False,
    all_datasets = False,
    unity_datasets = False,
    real_world_datasets = False,
):
    
    dbx_datasets_dir = '/UMARV/ML/datasets'

    if test_dataset:
        datasets_dir = [f"{dbx_datasets_dir}/unity/test"]

    for dataset_dir in datasets_dir:

        copy_directory_from_dropbox_slow(
            dbx_access_token = dbx_access_token,
            source_dir = dataset_dir,
            destination_dir = "/content"
        )
