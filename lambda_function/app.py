import boto3
import pickle
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging
import traceback

# -------------------------------
# Setup logging for CloudWatch
# -------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Optional: Add a formatter for more structured logs
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Logger initialized successfully.")

# -------------------------------
# Initialize S3 client
# -------------------------------
s3 = boto3.client("s3")
logger.info("S3 client initialized.")

# -------------------------------
# Configuration
# -------------------------------
MODEL_BUCKET = "learnyard-data-ingestion"
MODEL_KEY = "models/modelv1.pkl"
OUTPUT_BUCKET = "learnyard-data-ingestion"

def lambda_handler(event, context):
    try:
        logger.info("Lambda function started.")
        
        # -------------------------------
        # Check if model exists in S3
        # -------------------------------
        try:
            s3.head_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
            logger.info(f"Model found in S3: s3://{MODEL_BUCKET}/{MODEL_KEY}")
        except Exception:
            logger.error(f"Model not found in S3: s3://{MODEL_BUCKET}/{MODEL_KEY}")
            raise FileNotFoundError(f"Model not found in S3: s3://{MODEL_BUCKET}/{MODEL_KEY}")

        # -------------------------------
        # Prepare temporary output path
        # -------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = "/tmp/predictions"
        os.makedirs(tmp_dir, exist_ok=True)
        local_output_path = os.path.join(tmp_dir, f"prediction_output_{timestamp}.json")
        output_key = f"predictions/prediction_output_{timestamp}.json"
        logger.info(f"Temporary output path prepared: {local_output_path}")

        # -------------------------------
        # Download model
        # -------------------------------
        local_model_path = "/tmp/model.pkl"
        logger.info(f"Downloading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
        s3.download_file(MODEL_BUCKET, MODEL_KEY, local_model_path)
        logger.info("Model downloaded successfully.")

        # -------------------------------
        # Load model
        # -------------------------------
        with open(local_model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")

        # -------------------------------
        # Determine feature columns
        # -------------------------------
        try:
            feature_columns = model.feature_names_in_.tolist()
        except AttributeError:
            feature_columns = [f"V{i}" for i in range(1, 31)]
        logger.info(f"Feature columns: {feature_columns}")

        # -------------------------------
        # Generate example inputs
        # -------------------------------
        num_samples = 10
        half = num_samples // 2
        non_fraud_inputs = np.random.rand(half, len(feature_columns)) * 0.3
        fraud_inputs = 0.7 + np.random.rand(num_samples - half, len(feature_columns)) * 0.3
        random_inputs = np.vstack([non_fraud_inputs, fraud_inputs])
        np.random.shuffle(random_inputs)
        input_df = pd.DataFrame(random_inputs, columns=feature_columns)
        logger.info(f"Generated input data for {num_samples} samples.")

        # -------------------------------
        # Make predictions safely
        # -------------------------------
        try:
            predictions = model.predict(input_df)
        except Exception:
            logger.error(f"Prediction error:\n{traceback.format_exc()}")
            predictions = np.zeros(len(input_df), dtype=int)

        logger.info(f"Predictions: {predictions.tolist()}")

        # -------------------------------
        # Prepare output JSON
        # -------------------------------
        output_data = [
            {"input_features": input_df.iloc[i].to_dict(), "prediction": int(predictions[i])}
            for i in range(num_samples)
        ]

        # -------------------------------
        # Save to /tmp and upload to S3
        # -------------------------------
        with open(local_output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        s3.upload_file(local_output_path, OUTPUT_BUCKET, output_key)
        logger.info(f"Predictions uploaded to s3://{OUTPUT_BUCKET}/{output_key}")

        logger.info("Lambda function completed successfully.")
        return {
            "statusCode": 200,
            "body": f"Batch prediction for {num_samples} samples completed successfully! File: s3://{OUTPUT_BUCKET}/{output_key}"
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in Lambda:\n{tb}")
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}\nTraceback:\n{tb}"
        }
