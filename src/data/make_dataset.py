import pandas as pd 
import zipfile
import os
import urllib.request
from pandas import DataFrame
import numpy as np
import logging
from pathlib import Path

# LOGGING
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# PATHING
current_dir = Path(__file__).resolve().parent.parent.parent

# raw_data_path = current_dir / "Data" / "Raw"
# process_data_path = current_dir / "Data" / "Interim"
# processed_data_path = current_dir / "Data" / "Processed"
data_dir = current_dir/ "Data" / "Raw"
interim_dir = current_dir / "Data" / "Interim"
processed_dir = current_dir/ "Data" / "Processed"
zip_name = "heart_disease.zip"
zip_name_two = "heart.zip"

UCI_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
UCI_ZIP = data_dir / "heart_disease.zip"

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/johnsmith88/heart-disease-dataset"
KAGGLE_ZIP = data_dir / "heart.zip"

# COLUMNS
# The attribute names given to the data that will be utilized
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

def ensure_dirs():
    """Create data directories if they do not exist"""
    for d in (data_dir, interim_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory: {d}")

def download_zip(url: str, dest: Path):
    """ Download file from given URL if not present """
    if dest.exists():
        logger.info(f"Already Downloaded; {dest.name}")
        return
    logger.info(f"Downloading {dest.name}...")
    urllib.request.urlretrieve(url,dest)
    logger.info(f"Downloaded to {dest}")

def unzip_zip_file(zip_path: Path, target_dir: Path):
    """ Extracting ZIP file contents into the target_dir """
    if not zip_path.exists():
        logger.warning(f"ZIP file not found: {zip_path}")
        return
    with zipfile.ZipFile(zip_path, 'r') as zipper:
        zipper.extractall(path=target_dir)
    logger.info(f"Extracted {zip_path.name} to {target_dir}")



def process_data():
    """ Reads raw data and assigns columns and saves to interim as a CSV 
        processed_cleveland.data -> test.csv
        heart.csv                -> train.csv
    """
    clev_file = data_dir / "processed.cleveland.data"
    if not clev_file.exists():
        logger.error(f"Missing Cleveland data: {clev_file}")
        return 
    df_clev = pd.read_csv(clev_file, header=None, names=columns)
    df_clev.to_csv(interim_dir/ "Cleveland_Heart_Disease.csv", index=False)
    logger.info(f"Saved interim Cleveland data {df_clev.shape}")

    train_file = data_dir / "heart.csv"
    if not train_file.exists():
        logger.error(f"Missing train CSV: {train_file}")
        return
    df_train = pd.read_csv(train_file)
    df_train.columns = columns
    df_train.to_csv(interim_dir / "Train_Set.csv", index=False)
    logger.info(f"Saved interim Train Data {df_train.shape}")

    print("Head of clev data\n")
    print(df_clev.head)

    print("Head of train data\n")
    print(df_train.head)

def clean_data():
    """
    Clean interim CSV by:
    * remove and replace '?' with NaN 
    * Dropping rows with any NaN 
    * Dropping duplicates 
    * Saving the final Train/Test sets to processed directory
    """
    df_train = pd.read_csv(interim_dir / "Train_Set.csv")
    df_test = pd.read_csv(interim_dir / "Cleveland_Heart_Disease.csv")

    # num_rows_with_missing_train = df_train.eq("?").any(axis=1).sum()
    # num_rows_with_missing_test = df_test.eq("?").any(axis=1).sum()
    # print(num_rows_with_missing_train)
    # No missing values
    # print(num_rows_with_missing_test)
    # 6 missing values
    df_train.replace('?', np.nan, inplace=True)
    df_test.replace('?', np.nan, inplace=True)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    before_train = df_train.shape[0]
    df_train.drop_duplicates(inplace=True)
    after_train = df_train.shape[0]
    logger.info(f"Train rows: {before_train} -> {after_train} after de-duplication")

    before_test = df_test.shape[0]
    df_test.drop_duplicates(inplace=True)
    after_test = df_test.shape[0]
    logger.info(f"Test rows: {before_test} -> {after_test} after de-duplication")

    df_train.to_csv(processed_dir / "Train.csv", index=False)
    df_test.to_csv(processed_dir / "Test.csv", index=False)
    logger.info("Saved Train and Test Sets")

    for col in df_train.columns:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col]  = pd.to_numeric(df_test[col], errors='coerce')

    common = pd.merge(
        df_train,
        df_test,
        on=list(df_train.columns),
        how="inner"
    )

    print(f"Overlapping rows: {len(common)}")
    # No overlap between train and test set so the test set will be be new to the model after training 
    # However the problem with the training set being small could end up being problematic 




def main():
    ensure_dirs()

    download_zip(UCI_URL, UCI_ZIP)

    try:
        download_zip(KAGGLE_URL, KAGGLE_ZIP)
    except Exception: 
        logger.warning("kaggle download failed")
    
    unzip_zip_file(UCI_ZIP, data_dir)
    unzip_zip_file(UCI_ZIP, data_dir)

    process_data()

    clean_data()
    
    

    


if __name__ == "__main__":
    main()

