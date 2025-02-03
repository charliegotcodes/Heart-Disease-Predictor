import pandas as pd 
import zipfile
import os
import urllib.request
from pandas import DataFrame
import numpy as np

from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent.parent

raw_data_path = current_dir / "Data" / "Raw"
process_data_path = current_dir / "Data" / "Interim"
processed_data_path = current_dir / "Data" / "Processed"
data_dir = os.listdir(raw_data_path)
interim_dir = os.listdir(process_data_path)
processed_dir = os.listdir(processed_data_path)
zip_name = "heart_disease.zip"
zip_name_two = "heart.zip"

# train_data = Kaggle Dataset
# test_data = Cleaveland Dataset

# The attribute names given to the data that will be utilized
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

def download_zip():
    # Downloads the zip from the given URL if the data_dir is empty only and utilizes urllib.requests
    # To retrieve the zip file and names it heart_disease.zip
    if len(data_dir) == 0:
        URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
        urllib.request.urlretrieve(URL,"heart_disease.zip")
        URL_Two = "https://www.kaggle.com/api/v1/datasets/download/johnsmith88/heart-disease-dataset"
        urllib.request.urlretrieve(URL_Two,"heart.zip")

def unzip_zip_file():
    # Unzips the zipfile collected from the URL and places them in the Raw data folder
    # which is found using the raw_data_path
    zip_obj = zipfile.ZipFile(zip_name, "r")
    with zipfile.ZipFile(zip_name) as zip_obj: 
        zip_obj.extractall(path=raw_data_path)
    zip_obj = zipfile.ZipFile(zip_name_two, "r")
    with zipfile.ZipFile(zip_name_two) as zip_obj: 
        zip_obj.extractall(path=raw_data_path)
    # dataset = pd.read_csv("heart.csv")
    # dataset.to_csv(raw_data_path / "Train_Set.csv")


def clean_up_file():
    #Removing the zip file since its unecessary
    os.remove(zip_name)
    os.remove(zip_name_two)


def process_data():
    # Reads all the .data files as CSV's into dataframes and incorporates corresponding 
    # column names and combines the data for training into df_combined and df_cleveland for testing 
    # Are both saved to Interim data as a CSV
    df_cleveland = pd.read_csv(raw_data_path / "processed.cleveland.data")
    df_Train_Set = pd.read_csv(raw_data_path / "heart.csv")
    # Ensuring the Shape of the data all matched columns wise before setting
    print("Shape of Cleveland Data: ",df_cleveland.shape)
    print("Shape of Train Set Data: ",df_Train_Set.shape)
    df_cleveland.columns = columns
    df_Train_Set.columns = columns
    print(df_cleveland.head())
    print(df_Train_Set.head())
    df_cleveland.to_csv(process_data_path / "Cleveland_Heart_Disease.csv")
    df_Train_Set.to_csv(process_data_path / "Train_Set.csv")

def clean_data():
    # Will remove datapoints with null variables as they are not useable
    df_unclean_cleveland = pd.read_csv(process_data_path / "Cleveland_Heart_Disease.csv")
    df_unclean_train_set = pd.read_csv(process_data_path / "Train_Set.csv")
    df_unclean_cleveland.replace(to_replace='?', value=np.nan, inplace=True)
    df_unclean_train_set.replace(to_replace='?', value=np.nan, inplace=True)
    missing_vals_cle = df_unclean_cleveland.isnull().sum()
    print("Cleveland Missing values: ", missing_vals_cle)
    missing_vals_train = df_unclean_train_set.isnull().sum()
    print("Training Missing Values: ", missing_vals_train)
    df_clean_train_set = df_unclean_train_set.dropna()
    df_clean_cleveland= df_unclean_cleveland.dropna()
    df_clean_cleveland.drop_duplicates()
    df_clean_train_set.drop_duplicates()
    print(df_clean_cleveland.shape)

    df_clean_cleveland.to_csv(processed_data_path / "Test.csv")
    df_clean_train_set.to_csv(processed_data_path / "Train.csv")

def main():
    if len(data_dir) == 0:
        download_zip()
        unzip_zip_file()
        clean_up_file()
    if len(interim_dir) == 0:
        process_data()
    if len(processed_dir) == 0:
        clean_data()
    
    

    


if __name__ == "__main__":
    main()

