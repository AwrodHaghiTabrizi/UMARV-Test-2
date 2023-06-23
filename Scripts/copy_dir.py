import os
import shutil
import datetime

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

# Example usage
source_directory = '/Template'
destination_directory = f'/TemplateCopy_{datetime.now().strftime("%y%m%d.%H%M%S")}'

copy_directory(source_directory, destination_directory)