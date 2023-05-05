from pathlib import Path
import os

                              
''' ######### PATHS ######### '''

# Get the project directory as the parent of this module location
PATH = Path(os.path.abspath(__file__)).parent
print(PATH)

CREDENTIALS_PATH = PATH / 'dokuso-dba4e049f89b.json'