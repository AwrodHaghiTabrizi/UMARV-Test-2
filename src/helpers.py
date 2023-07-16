import os
import shutil
from datetime import datetime
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
from tqdm.notebook import tqdm
from getpass import getpass
import dropbox

def copy_directory_from_dropbox(source_dir, destination_dir, dbx_access_token=None, use_thread=False):
    if dbx_access_token is None:
        dbx_access_token = getpass.getpass("Enter your DropBox access token: ")
    dbx = dropbox.Dropbox(dbx_access_token)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Retrieve the initial list of entries
    result = dbx.files_list_folder(source_dir)
    entries = result.entries

    # Process entries in batches until all entries are fetched
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)

    # Get the total number of items in the source directory
    total_items = len(entries)

    def download_and_save(item):
        source_item_path = item.path_display
        destination_item_path = os.path.join(destination_dir, os.path.basename(source_item_path))
        description = f"Copying {source_item_path} ..."

        if isinstance(item, dropbox.files.FolderMetadata):
            # Recursive call to copy subdirectory
            copy_directory_from_dropbox(source_item_path, destination_item_path, dbx_access_token, use_thread)
        else:
            # Download image file from Dropbox
            try:
                _, response = dbx.files_download(source_item_path)
                content = response.content
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imwrite(destination_item_path, image)

            except dropbox.exceptions.ApiError as e:
                print(f"Error retrieving image: {e}")

    if use_thread:
        # Use ThreadPoolExecutor to parallelize the processing of items
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(download_and_save, entries), total=total_items, desc=f"Copying {source_dir} :"))
    else:
        # Use the slow method, download and save files one by one
        for item in tqdm(entries, total=total_items, desc=f"Copying {source_dir} :"):
            download_and_save(item)

def copy_directory_from_dropbox_fast(source_dir, destination_dir, dbx_access_token=None):
    if dbx_access_token is None:
        dbx_access_token = getpass.getpass("Enter your DropBox access token: ")
    dbx = dropbox.Dropbox(dbx_access_token)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Retrieve the initial list of entries
    result = dbx.files_list_folder(source_dir)
    entries = result.entries

    # Process entries in batches until all entries are fetched
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)

    # Get the total number of items in the source directory
    total_items = len(entries)

    def download_and_save(item):
        source_item_path = item.path_display
        destination_item_path = os.path.join(destination_dir, os.path.basename(source_item_path))
        description = f"Copying {source_item_path} ..."

        if isinstance(item, dropbox.files.FolderMetadata):
            # Recursive call to copy subdirectory
            copy_directory_from_dropbox_fast(source_item_path, destination_item_path, dbx_access_token)
        else:
            # Download image file from Dropbox
            try:
                _, response = dbx.files_download(source_item_path)
                content = response.content
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imwrite(destination_item_path, image)

            except dropbox.exceptions.ApiError as e:
                print(f"Error retrieving image: {e}")

    # Use ThreadPoolExecutor to parallelize the processing of items
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(download_and_save, entries), total=total_items, desc=f"Copying {source_dir} :"))

# def copy_directory_from_dropbox_fast(source_dir, destination_dir):
#     dbx_access_token = getpass("Enter your DropBox access token: ")
#     dbx = dropbox.Dropbox(dbx_access_token)

#     # Create the destination directory if it doesn't exist
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)

#     # Create a list to hold the submitted tasks
#     tasks = []

#     # tqdm.monitor_interval = 0

#     # Get the total number of files and subdirectories
#     total_files = 0
#     for _ in dbx.files_list_folder(source_dir).entries:
#         total_files += 1

#     # Download files and subdirectories recursively from Dropbox
#     with ThreadPoolExecutor(max_workers=10) as executor, tqdm(total=total_files, unit='file') as pbar:
#         for item in dbx.files_list_folder(source_dir).entries:
#             source_item_path = item.path_display
#             destination_item_path = os.path.join(destination_dir, os.path.relpath(source_item_path, source_dir))

#             if isinstance(item, dropbox.files.FolderMetadata):
#                 # Recursive call to copy subdirectory
#                 tasks.append(executor.submit(copy_directory_from_dropbox_fast, source_item_path, destination_item_path))
#             else:
#                 # Download image file from Dropbox
#                 try:
#                     _, response = dbx.files_download(source_item_path)
#                     content = response.content
#                     nparr = np.frombuffer(content, np.uint8)
#                     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                     tasks.append(executor.submit(cv2.imwrite, destination_item_path, image))

#                 except dropbox.exceptions.ApiError as e:
#                     print(f"Error retrieving image: {e}")

#                 # Update the progress bar
#                 pbar.update(1)

#         # Wait for all tasks to complete
#         wait(tasks)

def copy_directory_from_dropbox_slow(source_dir, destination_dir, dbx_access_token=None):
    if dbx_access_token is None:
        dbx_access_token = getpass("Enter your DropBox access token: ")
    dbx = dropbox.Dropbox(dbx_access_token)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Retrieve the initial list of entries
    result = dbx.files_list_folder(source_dir)
    entries = result.entries

    # Process entries in batches until all entries are fetched
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(result.entries)

    # Get the total number of items in the source directory
    total_items = len(entries)

    # Download files and subdirectories recursively from Dropbox
    for item in tqdm(entries, total=total_items, desc=f"Copying {source_dir} :"):
        source_item_path = item.path_display
        destination_item_path = os.path.join(destination_dir, os.path.basename(source_item_path))
        description = f"Copying {source_item_path} ..."

        if isinstance(item, dropbox.files.FolderMetadata):
            # Recursive call to copy subdirectory
            copy_directory_from_dropbox_slow(source_item_path, destination_item_path, dbx_access_token)
        else:
            # Download image file from Dropbox
            try:
                _, response = dbx.files_download(source_item_path)
                content = response.content
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imwrite(destination_item_path, image)

            except dropbox.exceptions.ApiError as e:
                print(f"Error retrieving image: {e}")

def copy_directory(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Copy files and subdirectories recursively
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        destination_item = os.path.join(destination_dir, item)

        if os.path.isdir(source_item):
            # Recursive call to copy subdirectory
            copy_directory(source_item, destination_item)
        else:
            # Copy file
            shutil.copy2(source_item, destination_item)

def base10_to_base36(number):
    if number == 0:
        return '0'
    
    base36_chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ''
    
    while number > 0:
        number, remainder = divmod(number, 36)
        result = base36_chars[remainder] + result
    
    return result