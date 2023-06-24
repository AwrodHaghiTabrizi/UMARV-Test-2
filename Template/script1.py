print('hello world, I am script1')

import logging
import os

script_directory = os.path.abspath(__file__)
model_directory = os.path.dirname(script_directory)
file_name = os.path.basename(script_directory)
model_id =  os.path.splitext(file_name)[0].split('_')[-1]

logging.basicConfig(
    filename=f'{model_directory}/history.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f'Checking external log access')