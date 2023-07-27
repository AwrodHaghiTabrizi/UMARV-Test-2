import os
import shutil
from datetime import datetime
import logging
import sys
import json
import nbformat

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

    # Fill in info.json
    model_info_dir = f"{new_model_dir}/content/info.json"
    with open(model_info_dir, 'r') as file:
        model_info = json.load(file)
    model_info['author'] = model_author
    model_info['model_id'] = model_id
    model_info['creation_date'] = creation_date.strftime("%B %d, %Y, %H:%M:%S")
    with open(model_info_dir, 'w') as file:
        json.dump(model_info, file, indent=4)

    # Add model_id to methods.py
    methods_dir = f"{new_model_dir}/src/methods.py"
    with open(methods_dir, "r") as file:
        content = file.read()
    content = content.replace(
        '# Insert MODEL_ID here\nos.environ["MODEL_ID"] = ""',
        f'os.environ["MODEL_ID"] = "{model_id}"'
    )
    with open(methods_dir, 'w') as file:
        file.write(content)

    # Add model_id to notebooks
    notebook_names = ["colab_env", "lambda_env", "mac_env", "windows_env"]
    for notebook_name in notebook_names:
        notebook_dir = f"{new_model_dir}/src/notebooks/{notebook_name}.ipynb"
        with open(notebook_dir, 'r') as file:
            notebook = nbformat.read(file, as_version=4)
        cell = notebook['cells'][3]
        cell['source'] = cell['source'].replace(
            '# Insert MODEL_ID here\nos.environ["MODEL_ID"] = ""',
            f'os.environ["MODEL_ID"] = "{model_id}"'
        )
        with open(notebook_dir, 'w') as file:
            nbformat.write(notebook, file)

    print(f"Model {model_id} initialized")


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