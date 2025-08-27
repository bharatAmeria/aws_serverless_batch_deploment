import os
import sys
import zipfile
import gdown
from dotenv import load_dotenv
from src.logger import logging
from src.exception import MyException
from src.config import CONFIG

load_dotenv()

class Data_Ingestion:
    """
    Data ingestion class which downloads data from Google Drive, extracts it,
    and uploads the contents to AWS S3.
    """

    def __init__(self):
        self.config = CONFIG["DATA_INGESTION"]
        logging.info("Data Ingestion class initialized.")

    def download_file(self):
        """Download dataset ZIP file from Google Drive"""
        try:
            dataset_url = os.getenv("DATASET_URI")
            zip_download_path = self.config["UNZIP_DIR"]  # artifacts/archive.zip
            os.makedirs(os.path.dirname(zip_download_path), exist_ok=True)

            logging.info(f"Downloading data from {dataset_url} into {zip_download_path}")
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?export=download&id='
            gdown.download(prefix + file_id, zip_download_path, quiet=False)

            logging.info(f"Successfully downloaded data to {zip_download_path}")
        except Exception as e:
            logging.error("Error occurred while downloading file", exc_info=True)
            raise MyException(e, sys)

    def extract_zip_file(self):
        """Extract downloaded ZIP file"""
        try:
            unzip_path = self.config["ZIP_FILE"]      
            local_data_file = self.config["UNZIP_DIR"] 

            os.makedirs(unzip_path, exist_ok=True)
            logging.info(f"Extracting {local_data_file} to {unzip_path}")

            with zipfile.ZipFile(local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logging.info(f"Successfully extracted to {unzip_path}")
        except Exception as e:
            logging.error("Error occurred while extracting zip file", exc_info=True)
            raise MyException(e, sys)
