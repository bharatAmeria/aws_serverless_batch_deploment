import sys
from src.logger import logging
from src.exception import MyException
# from src.pipeline.stage01_data_ingestion import Data_Ingestion_Pipeline
# from src.pipeline.stage02_processing import Processing_Pipeline
from src.pipeline.stage03_training import Training_Pipeline

# try:
#     data_ingestion = Data_Ingestion_Pipeline()
#     data_ingestion.main()
# except MyException as e:
#     logging.exception(e, sys)
#     raise e

# try:
#     data_processing = Processing_Pipeline()
#     data_processing.main()
# except MyException as e:
#     logging.exception(e, sys)
#     raise e

try:
    training = Training_Pipeline()
    training.main()
except MyException as e:
    logging.exception(e, sys)
    raise e