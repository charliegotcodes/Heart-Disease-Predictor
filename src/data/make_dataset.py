import pandas as pd 
import zipfile
import os
import urllib.request
from pandas import DataFrame

raw_data_path = "/Users/kevincharles/Desktop/Heart Disease Predictor/Data/Raw"

data_dir = os.listdir(raw_data_path)
zip_name = "heart_disease.zip"
if len(data_dir) == 0:
    URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    urllib.request.urlretrieve(URL,"heart_disease.zip")
    zip_obj = zipfile.ZipFile(zip_name, "r")
    with zipfile.ZipFile(zip_name) as zip_obj: 
        zip_obj.extractall(path=raw_data_path)
