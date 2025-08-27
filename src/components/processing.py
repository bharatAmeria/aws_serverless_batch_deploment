import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import MyException
from typing import Optional


class CreditCardFraudProcessor:
    def __init__(self, file_path: str):
        """
        Initialize the processor with the dataset path.
        """
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        self.new_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data into pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"✅ Data loaded successfully | Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError as e:
            logging.error(f"❌ File not found: {self.file_path}")
            raise MyException(e, sys)
        except Exception as e:
            logging.error(f"❌ Unexpected error while loading data: {e}")
            raise MyException(e, sys)

    def balance_data(self, n_samples: int = 492) -> pd.DataFrame:
        """
        Balance dataset by undersampling legitimate transactions.
        """
        try:
            if self.data is None:
                raise ValueError("Data not loaded. Call load_data() first.")

            legit = self.data[self.data['Class'] == 0]
            fraud = self.data[self.data['Class'] == 1]

            if len(legit) < n_samples:
                raise ValueError("Not enough legitimate samples to downsample.")

            legit_sample = legit.sample(n=n_samples, random_state=42)
            self.new_df = (
                pd.concat([legit_sample, fraud], axis=0)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

            logging.info(f"✅ Balanced dataset created | Shape: {self.new_df.shape}")
            return self.new_df

        except Exception as e:
            logging.error(f"❌ Error while balancing dataset: {e}")
            raise MyException(e, sys)

    def compute_class_means(self) -> pd.DataFrame:
        """
        Compute mean values for each class.
        """
        try:
            if self.new_df is None:
                raise ValueError("Balanced dataset not created. Call balance_data() first.")

            class_means = self.new_df.groupby('Class').mean(numeric_only=True)
            logging.info("✅ Computed mean values for each class.")
            return class_means

        except Exception as e:
            logging.error(f"❌ Error while computing class means: {e}")
            raise MyException(e, sys)

    def save_data(self, file_name: str = "balanced_data.csv") -> None:
        """
        Save the balanced dataset to a CSV file.
        """
        try:
            if self.new_df is None:
                raise ValueError("Balanced dataset not created. Call balance_data() first.")
            
            dir_name = os.path.dirname(file_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                
            self.new_df.to_csv(file_name, index=False)
            logging.info(f"✅ Balanced dataset saved successfully as '{file_name}'")

        except Exception as e:
            logging.error(f"❌ Error while saving data: {e}")
            raise MyException(e, sys)
