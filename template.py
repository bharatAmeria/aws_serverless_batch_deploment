import os
from pathlib import Path

project_name = "src"

list_of_files = [

    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/ingestion.py",
    f"{project_name}/components/processing.py",
    f"{project_name}/components/training.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/stage01_data_ingestion.py",
    f"{project_name}/pipeline/stage02_processing.py",
    f"{project_name}/pipeline/stage03_training.py",
    "lambda_function/app.py",
    "lambda_function/Dockerfile",
    "lambda_function/requirements.txt",
    ".github/workflows/docker.yaml",
    ".github/workflows/training.yaml",
    "main.py",
    ".dockerignore",
    ".env",
    ".project-root",
    "config.json",
    "parms.json",
    "project_flow.txt",
    "pyproject.toml",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")