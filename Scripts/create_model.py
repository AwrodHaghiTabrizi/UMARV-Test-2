import os
import shutil
from datetime import datetime
import logging

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

def main():

    # Example usage
    model_id = datetime.now().strftime("%y%m%d.%H%M%S")
    base_directory = os.getcwd()
    source_directory = f'{base_directory}/Template'
    destination_directory = f'{base_directory}/TemplateCopy_{model_id}'
    copy_directory(source_directory, destination_directory)

    logging.basicConfig(
        filename=f'{destination_directory}/history.log', 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f'Model {model_id} initialized')

    logging.info(f'Log preserved')
    logging.info(f'Checking duplicate log')

if __name__ == "__main__":
    main()