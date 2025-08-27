import sys
from src.config import CONFIG
from src.components.processing import CreditCardFraudProcessor
from src.exception import MyException

class Processing_Pipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = CONFIG.get("DATA_PROCESSING", {})

        data = config.get("RAW_DATA")
        balanced_data = config.get("BALANCED_DATA")

        processor = CreditCardFraudProcessor(data)
        processor.load_data()
        processor.balance_data(n_samples=492)
        processor.save_data(balanced_data)


if __name__ == '__main__':
    try:
        obj = Processing_Pipeline()
        obj.main()
    except MyException as e:
            raise MyException(e, sys)