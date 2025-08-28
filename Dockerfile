# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential gcc g++ \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt --target /app/package \
    && apt-get remove -y build-essential gcc g++ \
    && apt-get autoremove -y \
    && apt-get clean

# Lambda runtime stage
FROM public.ecr.aws/lambda/python:3.10

# Copy dependencies
COPY --from=builder /app/package ${LAMBDA_TASK_ROOT}

# Copy Lambda handler code
COPY lambda_batch_prediction/lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set Lambda handler
CMD ["lambda_function.lambda_handler"]
