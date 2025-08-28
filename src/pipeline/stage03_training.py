import sys
import pandas as pd
from src.config import CONFIG, PARMS
from src.components.training import ModelTrainer
from src.exception import MyException

class Training_Pipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        try:
            config = CONFIG.get("MODEL", {})
            parms = PARMS.get("TRAINING_PARMS", {})

            # Get file paths
            balanced_data_path = config.get("BALANCED_DATA")
            model_path = config.get("TRAINED_MODEL")
            
            # Load the CSV into DataFrame
            balanced_data = pd.read_csv(balanced_data_path)

            # Separate features and labels
            X = balanced_data.drop(columns='Class', axis=1)
            Y = balanced_data['Class']

            # Training
            trainer = ModelTrainer(
                data_path=balanced_data_path,
                model_path=model_path,
                X=X,
                Y=Y,
                config=config 
            )

            test_size = parms.get("TEST_SIZE", 0.2)
            trainer.split_data(test_size=test_size)
            trainer.train_model()
            train_acc, test_acc = trainer.evaluate_model()
            trainer.export_model(model_path)

            print(f"✅ Training Accuracy: {train_acc:.4f}")
            print(f"✅ Test Accuracy: {test_acc:.4f}")

        except MyException as e:
            raise MyException(e, sys)
        
if __name__ == '__main__':
    try:
        obj = Training_Pipeline()
        obj.main()
    except MyException as e:
            raise MyException(e, sys)