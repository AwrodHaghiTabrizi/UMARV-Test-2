import os
import shutil
from datetime import datetime
import logging
import sys
import json

repo_dir = os.getcwd()

src_dir = os.path.join(repo_dir, "src")
sys.path.append(src_dir)
from helpers import *

def main():

    creation_date = datetime.now()

    model_id_b10 = int(creation_date.strftime("%y%m%d%H%M%S"))
    model_id = base10_to_base36(model_id_b10)

    template_model_dir = f"{repo_dir}/models/model_template"
    new_model_dir = f"{repo_dir}/models/model_{model_id}"
    copy_directory(
        source_dir = template_model_dir,
        destination_dir = new_model_dir
    )

    model_author = "Awrod" #AHT#input("Author: ")

    model_info_dir = f"{new_model_dir}/info.json"
    with open(model_info_dir, 'r') as file:
        model_info = json.load(file)
    model_info['author'] = model_author
    model_info['model_id'] = model_id
    model_info['creation_date'] = creation_date.strftime("%B %d, %Y, %H:%M:%S")
    with open(model_info_dir, 'w') as file:
        json.dump(model_info, file, indent=4)


    # logging.basicConfig(
    #     filename=f'{destination_directory}/history.log', 
    #     level=logging.INFO, 
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )
    # logging.info(f'Model {model_id} initialized')

    # logging.info(f'Log preserved')
    # logging.info(f'Checking duplicate log')

if __name__ == "__main__":
    main()