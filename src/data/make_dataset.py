import pandas as pd 
import zipfile
import os
import urllib.request
from pandas import DataFrame

from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent.parent

print(current_dir)
raw_data_path = current_dir / "Data" / "Raw"
process_data_path = current_dir / "Data" / "Interim"
data_dir = os.listdir(raw_data_path)
interim_dir = os.listdir(process_data_path)
zip_name = "heart_disease.zip"

# train_data = {"Hungarian":[], "Switzerland":[], "Long Beach VA":[]}
# test_data = {"Cleavand":[]}

# The attribute names given to the data that will be utilized
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

def download_zip():
    # Downloads the zip from the given URL if the data_dir is empty only and utilizes urllib.requests
    # To retrieve the zip file and names it heart_disease.zip
    if len(data_dir) == 0:
        URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
        urllib.request.urlretrieve(URL,"heart_disease.zip")

def unzip_zip_file():
    # Unzips the zipfile collected from the URL and places them in the Raw data folder
    # which is found using the raw_data_path
    zip_obj = zipfile.ZipFile(zip_name, "r")
    with zipfile.ZipFile(zip_name) as zip_obj: 
        zip_obj.extractall(path=raw_data_path)

def clean_up_file():
    #Removing the zip file since its unecessary
    os.remove(zip_name)


def process_data():
    # BETTER EXPLANATION NEEDED HERE
    # Build Model On Hungarian, Switzerland and Long Beach VA 
    # Test Model On Cleavand
    # Convert the .data to a .csv file 
    df_Hungarian = pd.read_csv(raw_data_path / "processed.hungarian.data")
    df_Switzerland = pd.read_csv(raw_data_path / "processed.switzerland.data")
    df_LongBeach = pd.read_csv(raw_data_path / "processed.va.data")
    df_cleveland = pd.read_csv(raw_data_path / "processed.cleveland.data")
    # Ensuring the Shape of the data all matched columns wise before setting
    print("Shape of Hungarian Data: ",df_Hungarian.shape)
    print("Shape of Switzerland Data: ",df_Switzerland.shape)
    print("Shape of LongBeach Data: ",df_LongBeach.shape)
    df_Hungarian.columns = columns
    df_Switzerland.columns = columns
    df_LongBeach.columns = columns
    df_cleveland.columns = columns
    print(df_cleveland.head())
    df_combined= pd.concat([df_Hungarian, df_Switzerland, df_LongBeach], ignore_index=True)
    print("Combined Heart Disease Shape: ", df_combined.shape)
    df_combined.to_csv(process_data_path / "Combined_Heart_Disease.csv")
    df_cleveland.to_csv(process_data_path / "Cleveland_Heart_Disease.csv")

# def clean_data():
#     # Will remove datapoints with null variables as they are not useable
    



def main():
    if len(data_dir) == 0:
        download_zip()
        unzip_zip_file()
        clean_up_file()
    if len(interim_dir) == 0:
        process_data()
    # clean_data()
    
    

    


if __name__ == "__main__":
    main()

