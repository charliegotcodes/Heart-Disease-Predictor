import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import os
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns 
import re

from pandas import DataFrame
import numpy as np

from pathlib import Path

# Possible problem is the hot encoding of categorical values to numerical values
# Fix this and then run the model possibly
current_dir = Path(__file__).resolve().parent.parent.parent
feature_dir = os.listdir(current_dir / "Data" / "Selected_Features") 

def load_processed_data(filepath):
    df_Train = pd.read_csv(filepath / "Data" / "Processed" / "Train.csv")
    print("Training Data's shape: ", df_Train.shape)
    return df_Train


def correlation_analysis(df):
    print("Correlation analysis of the dataframes")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Features Correlation Matrix")
    plt.savefig(current_dir /"src" / "visualization" / "heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Since we are converting categorical variables to numerical values these are the important features according to correlation
    print("categorical to numerical \n")
    df = pd.get_dummies(df, columns=['sex', 'cp','fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    return df

def chi_square_feat_selection(X, y, k=7):
    chi_selector = SelectKBest(score_func=chi2, k=k)
    chi_selector.fit(X, y)
    selected_features = X.columns[chi_selector.get_support()]
    print("Best Features: ", selected_features)
    return selected_features

def rfe_feature_selection(X, y, k=7):
    model = RandomForestClassifier(random_state=69)
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    print("Best Features: ", selected_features)
    return selected_features

def save_selected_features(df, selected_features, out_file):
    df_selected = df[selected_features].copy()
    df_selected['target'] = df['target']
    df_selected.rename(columns=lambda x: re.sub(r'_\d+(\.\d+)?$','', x), inplace=True)
    df_selected.to_csv(out_file, index=False)


def main():
    df_Train = load_processed_data(current_dir)
    df_Train = df_Train.drop(["Unnamed: 0.1","Unnamed: 0"], axis= 1)
    df_Train = correlation_analysis(df_Train)
    X = df_Train.drop(columns=["target"])
    y = df_Train["target"]

    selected_chi2 = chi_square_feat_selection(X, y)
    selected_rfe = rfe_feature_selection(X, y)

    finalized_Features =  selected_rfe

    if len(feature_dir) == 0:
        save_selected_features(df_Train, finalized_Features, current_dir / "Data" / "Selected_Features"/ "selected_features.csv")






if __name__ == "__main__":
    main()