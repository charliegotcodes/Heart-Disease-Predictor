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
    df_Train_Set = pd.read_csv(raw_data_path / "heart.csv")
    df_Test_Set = pd.read_csv(raw_data_path / "processed.cleveland.data")
    # Ensuring the Shape of the data all matched columns wise before setting
    print("Shape of Test Set Data in process_data: ",df_Test_Set.shape)
    print("Shape of Train Set Data in process_data: ",df_Train_Set.shape)
    df_Test_Set.columns = columns
    df_Train_Set.columns = columns
    print(df_Test_Set.head())
    print(df_Train_Set.head())
    df_Test_Set.to_csv(process_data_path / "Test_Set.csv", index=False)
    df_Train_Set.to_csv(process_data_path / "Train_Set.csv", index=False)

def clean_data():
    # Will remove datapoints with null variables as they are not useable
    df_unclean_train= pd.read_csv(process_data_path / "Train_Set.csv")
    df_unclean_test_set = pd.read_csv(process_data_path / "Test_Set.csv")
    missing_vals_cle = df_unclean_train.isnull().sum()
    print("Cleveland Missing values: ", missing_vals_cle)
    missing_vals_test = df_unclean_test_set.isnull().sum()
    print("Training Missing Values: ", missing_vals_test)
    df_unclean_train.replace(to_replace='?', value=np.nan, inplace=True)
    df_unclean_test_set.replace(to_replace='?', value=np.nan, inplace=True)
    df_clean_test_set = df_unclean_test_set.dropna()
    df_clean_train= df_unclean_train.dropna()
    df_clean_test_set.drop_duplicates()
    df_clean_train.drop_duplicates()
    # print("pre dropping column 0", df_clean_test_set.shape)
    # df_clean_test_set = df_clean_test_set.drop(df_clean_test_set.columns[0], axis=1)
    # print("Post dropping column 0", df_clean_test_set.shape)
    # df_clean_train = df_clean_train.drop(df_clean_train.columns[0], axis=1)


    df_clean_train.to_csv(processed_data_path / "Train.csv", index=False)
    df_clean_test_set.to_csv(processed_data_path / "Test.csv", index=False)

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

