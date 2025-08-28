import os
import sys
import boto3
import pandas as pd
import pickle
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.logger import logging
from src.exception import MyException
from typing import Optional
load_dotenv()

class ModelTrainer:
    def __init__(self, data_path: str, model_path: str, X: pd.DataFrame, Y: pd.Series, model: Optional[object] = None, config=None):
        """
        Initialize the trainer with features X, labels Y, and optional model.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.X = X
        self.Y = Y
        self.model = model if model else LogisticRegression(solver='liblinear', random_state=42)
        self.X_train = self.X_test = self.Y_train = self.Y_test = None
        self.config = config or {} 

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def split_data(self, test_size: float = 0.2, stratify: bool = True, random_state: int = 42):
        """
        Split the data into train and test sets.
        """
        try:
            stratify_labels = self.Y if stratify else None
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self.X, self.Y, test_size=test_size, stratify=stratify_labels, random_state=random_state
            )
            logging.info(f"✅ Data split completed | Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        except Exception as e:
            logging.error(f"❌ Error during data split: {e}")
            raise MyException(e, sys)

    def train_model(self):
        """
        Train the model on the training data.
        """
        try:
            if self.X_train is None or self.Y_train is None:
                raise ValueError("Training data not available. Call split_data() first.")

            self.model.fit(self.X_train, self.Y_train)
            logging.info("✅ Model training completed.")
        except Exception as e:
            logging.error(f"❌ Error during model training: {e}")
            raise MyException(e, sys)

    def evaluate_model(self):
        """
        Evaluate the model on training and test data.
        Returns: (training_accuracy, test_accuracy)
        """
        try:
            if self.X_train is None or self.X_test is None:
                raise ValueError("Data not split. Call split_data() first.")

            # Training accuracy
            train_preds = self.model.predict(self.X_train)
            train_accuracy = accuracy_score(self.Y_train, train_preds)
            logging.info(f"✅ Training Accuracy: {train_accuracy}")

            # Test accuracy
            test_preds = self.model.predict(self.X_test)
            test_accuracy = accuracy_score(self.Y_test, test_preds)
            logging.info(f"✅ Test Accuracy: {test_accuracy}")

            return train_accuracy, test_accuracy
        except Exception as e:
            logging.error(f"❌ Error during model evaluation: {e}")
            raise MyException(e, sys)

    def export_model(self, file_name: str = "model.pkl"):
        """
        Export the trained model locally and upload to AWS S3.
        """
        try:
            # ✅ Save locally
            os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

            with open(file_name, 'wb') as f:
                pickle.dump(self.model, f)
            logging.info(f"✅ Model exported locally at {file_name}")

            # ✅ AWS details from env/config
            bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
            s3_upload_prefix = self.config.get("s3_upload_prefix", "models")

            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_DEFAULT_REGION")

            if not aws_access_key or not aws_secret_key or not bucket_name:
                logging.warning("⚠️ AWS credentials or bucket name not set. Skipping S3 upload.")
                return

            # ✅ Initialize S3 client
            s3 = boto3.client(
                "s3",
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )

            # ✅ Define S3 key
            filename = os.path.basename(file_name)
            s3_key = os.path.join(s3_upload_prefix, filename).replace("\\", "/")

            # ✅ Upload to S3
            s3.upload_file(file_name, bucket_name, s3_key)
            logging.info(f"📤 Uploaded model to s3://{bucket_name}/{s3_key}")

        except Exception as e:
            logging.error(f"❌ Error exporting model: {e}")
            raise MyException(e, sys)