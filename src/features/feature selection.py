import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import os
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns 
import re
from sklearn.preprocessing import OneHotEncoder

from pandas import DataFrame
import numpy as np

from pathlib import Path



# Features Utilized in the dataset:
# 3 age: age in years
# 4 sex: sex (1 = male; 0 = female)
# 9 cp: chest pain type
#         -- Value 1: typical angina
#         -- Value 2: atypical angina
#         -- Value 3: non-anginal pain
#         -- Value 4: asymptomatic
# 10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 12 chol: serum cholestoral in mg/dl
# 16 fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
# 19 restecg: resting electrocardiographic results
#         -- Value 0: normal
#         -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# 32 thalach: maximum heart rate achieved
# 38 exang: exercise induced angina (1 = yes; 0 = no)
# 40 oldpeak = ST depression induced by exercise relative to rest
# 41 slope: the slope of the peak exercise ST segment
        # -- Value 1: upsloping
        # -- Value 2: flat
        # -- Value 3: downsloping
# 44 ca: number of major vessels (0-3) colored by flourosopy
# 51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 58 target

# Possible problem is the hot encoding of categorical values to numerical values
# Fix this and then run the model possibly
current_dir = Path(__file__).resolve().parent.parent.parent
feature_dir = os.listdir(current_dir / "Data" / "Selected_Features") 

def load_processed_data(filepath):
    df_Track = pd.read_csv(filepath / "Data" / "Processed" / "Train.csv")
    print("Before encoding:")
    print(df_Track.head())
    print("Columns:", df_Track.columns.tolist())
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    cols_to_encode = [col for col in categorical_cols if col in df_Track.columns]
    # if cols_to_encode:
    #     # Create an instance of OneHotEncoder.
    #     # Set sparse=False to return a dense array.
    #     # Optionally, you can use drop='first' to avoid dummy variable trap.
    #     encoder = OneHotEncoder(sparse_output=False, drop=None)
        
    #     # Fit and transform the categorical columns
    #     encoded_array = encoder.fit_transform(df_Track[cols_to_encode])
    #     # Get new column names from the encoder
    #     encoded_col_names = encoder.get_feature_names_out(cols_to_encode)
    #     # Create a DataFrame from the encoded array with the proper column names
    #     encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df_Track.index)
        
    #     # Combine the non-categorical columns with the encoded DataFrame
    #     df = pd.concat([df_Track.drop(columns=cols_to_encode), encoded_df], axis=1)
    # else:
    #     # If no columns to encode, copy the DataFrame
    #      df = df_Track.copy()
    # print("Below here\n")
    # print(df.head)
    return df_Track


def correlation_analysis(df):
    print("Correlation analysis of the dataframes")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Features Correlation Matrix")
    plt.savefig(current_dir /"src" / "visualization" / "heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Since we are converting categorical variables to numerical values these are the important features according to correlation
    # print("categorical to numerical \n")
    # df = pd.get_dummies(df, columns=['sex', 'cp','fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    return df

def chi_square_feat_selection(X, y, k=9):
    chi_selector = SelectKBest(score_func=chi2, k=k)
    chi_selector.fit(X, y)
    selected_features = X.columns[chi_selector.get_support()]
    print("Best Features: ", selected_features)
    return selected_features

def rfe_feature_selection(X, y, k=9):
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
    print(df_Train.head)
    print("Here\n")
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