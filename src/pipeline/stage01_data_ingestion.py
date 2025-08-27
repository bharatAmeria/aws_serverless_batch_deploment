import sys
import pandas as pd
from src.components.ingestion import Data_Ingestion
from src.logger import logging
from src.exception import MyException

class Data_Ingestion_Pipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        upload = Data_Ingestion()
        upload.download_file()
        upload.extract_zip_file()



if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage Data Ingestion started <<<<<<")
        obj = Data_Ingestion_Pipeline()
        obj.main()
    except MyException as e:
            raise MyException(e, sys)