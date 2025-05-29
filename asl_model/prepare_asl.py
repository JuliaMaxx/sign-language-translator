# prepare.py
import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = 'grassknoted/asl-alphabet'
ZIP_PATH = '../data/asl-alphabet.zip'
EXTRACT_DIR = '../data/asl_alphabet_train'

os.makedirs('data', exist_ok=True)

if not os.path.exists(EXTRACT_DIR):
    api = KaggleApi()
    api.authenticate()
    print("Downloading...")
    api.dataset_download_files(DATASET, path='data', unzip=False)

    print("Unzipping...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('data')
    print("Done.")
else:
    print("Dataset already exists.")
