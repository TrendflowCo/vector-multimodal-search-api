import os
from google.oauth2 import service_account
from paths import *

credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH,
    scopes=['https://www.googleapis.com/auth/devstorage.full_control', 
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/spreadsheets',
            "https://www.googleapis.com/auth/drive"]
    )

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(CREDENTIALS_PATH)