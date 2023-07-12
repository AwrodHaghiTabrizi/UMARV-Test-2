import os
import shutil
from datetime import datetime
import logging
import sys
import json
import time

repo_dir = os.getcwd()

src_dir = os.path.join(repo_dir, "src")
sys.path.append(src_dir)
from helpers import *

def main():

    start_time = time.time()

    copy_directory_from_dropbox_slow(
        source_dir = "/UMARV/ML/datasets/unity/test",
        destination_dir = f"{repo_dir}/test_dataset"
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()