import boto3
import pickle
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging

# -------------------------------
# Setup logging
# -------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# -------------------------------
# Initialize S3 client
# -------------------------------
s3 = boto3.client("s3")

# -------------------------------
# Configuration
# -------------------------------
MODEL_BUCKET = "learnyard-data-ingestion"
MODEL_KEY = "models/modelv1.pkl"
OUTPUT_BUCKET = "learnyard-data-ingestion"

def lambda_handler(event, context):
    try:
        # -------------------------------
        # Prepare temporary output path
        # -------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = "/tmp/predictions"
        os.makedirs(tmp_dir, exist_ok=True)
        local_output_path = os.path.join(tmp_dir, f"prediction_output_{timestamp}.json")
        output_key = f"predictions/prediction_output_{timestamp}.json"

        # -------------------------------
        # Download the model from S3
        # -------------------------------
        local_model_path = "/tmp/model.pkl"
        logger.info(f"Downloading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
        s3.download_file(MODEL_BUCKET, MODEL_KEY, local_model_path)

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

        # -------------------------------
        # Make predictions
        # -------------------------------
        predictions = model.predict(input_df)
        logger.info(f"Predictions: {predictions.tolist()}")

        # -------------------------------
        # Prepare output JSON
        # -------------------------------
        output_data = [
            {"input_features": input_df.iloc[i].to_dict(), "prediction": int(predictions[i])}
            for i in range(num_samples)
        ]

        # -------------------------------
        # Save to local /tmp and upload to S3
        # -------------------------------
        with open(local_output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        s3.upload_file(local_output_path, OUTPUT_BUCKET, output_key)
        logger.info(f"Predictions uploaded to s3://{OUTPUT_BUCKET}/{output_key}")

        return {
            "statusCode": 200,
            "body": f"Batch prediction for {num_samples} samples completed successfully! File: s3://{OUTPUT_BUCKET}/{output_key}"
        }

    except Exception as e:
        logger.error(f"Error in Lambda: {str(e)}")
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
