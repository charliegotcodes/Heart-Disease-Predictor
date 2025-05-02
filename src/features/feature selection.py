import pandas as pd 
import matplotlib.pyplot as plt 
import os
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns 
from pathlib import Path


#Directories 
current_dir  = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = current_dir / "Data" / "Processed"
SELECTED_DIR = current_dir / "Data" / "Selected_Features"
SELECTED_FILE= SELECTED_DIR / "selected_features.csv"
HEATMAP_DIR = current_dir / "src" / "visualization"

HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_DIR.mkdir(parents=True, exist_ok=True)

# Explanation of the values from the dataset and what they represent 

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

#Renaming fields to help better understand

NAME_DICT = {
    'cp': 'chest_pain_type',
    'trestbps': 'resting_blood_pressure',
    'chol': 'cholesterol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'rest_ecg',
    'thalach': 'max_heart_rate_achieved',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'st_depression',
    'slope': 'st_slope',
    'ca': 'num_major_vessels',
    'thal': 'thalassemia'
}

MAPPING_DICTS = {
    'sex': {0: 'female', 1: 'male'},
    'chest_pain_type': {
        1: 'typical angina',
        2: 'atypical angina',
        3: 'non-anginal pain',
        4: 'asymptomatic'
    },
    'fasting_blood_sugar': {0: 'lower than 120mg/dl', 1: 'greater than 120mg/dl'},
    'rest_ecg': {0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'},
    'exercise_induced_angina': {0: 'no', 1: 'yes'},
    'st_slope': {1: 'upsloping', 2: 'flat', 3: 'downsloping'},
    'thalassemia': {3: 'normal', 6: 'fixed defect', 7: 'reversible defect'}
}


def load_data():
    """Load the cleaned data for Training into the Dataframe"""
    df = pd.read_csv(PROCESSED_DIR / "Train.csv")
    print(f"Loaded Train.csv: {df.shape[0]} rows, {df.shape[1]} columns")

    return df

def corr_analysis(df):
    """Generate and save the heatmap of the numeric feature correlations"""
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig(HEATMAP_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {HEATMAP_DIR}")

def perform_eda(df):
    """Run EDA plots and tables"""
    # Roughly around 140 for "no disease" and 165 for "disease" and the data is clearly close to balanced
    # that there is no need for oversampling or dropping
    print("1. Class Balance")
    sns.countplot(x='target', data=df, palette='pastel',legend=False, hue='target')
    plt.title('Target Class Distribution')
    plt.show()

    # Noticed outliers in the boxplot of blood pressure vs target
    # The gender distribution is 68% male to 32% female 
    # When viewing the violin plot the cholesterol distribution overlaps so it doesn't define the classes well
    # The stacked bar chart for the classes shows that the ST-T wave increases in target 1 indicative of underlying heart disease
    print("2. Numeric Pairplot")
    num_cols= ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
    sns.pairplot(df[num_cols + ['target']], hue='target', corner=True, diag_kind='hist')
    plt.suptitle('Numeric Feature Pairplot', y=1.02)
    plt.show()

    print("3. Distribution of Age")
    sns.histplot(df['age'], kde=True)
    plt.title('Distribution of Age')
    plt.show()

    print("4. Pie Chart of Sex")
    counts = df['sex'].value_counts()
    counts.plot.pie(autopct='%.2f%%', startangle=90, pctdistance=0.85)
    centre_circle=plt.Circle((0,0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)
    plt.title('Gender Distribution')
    plt.ylabel('')
    plt.show()

    print("5. Boxplot blood pressure vs target")
    sns.boxplot(x='target', y='resting_blood_pressure', data=df)
    plt.title('Resting Blood Pressure by Target')
    plt.show()

    print("6. Cholesterol vs Target")
    sns.violinplot(x='target', y='cholesterol', data=df)
    plt.title('Cholesterol by Target')
    plt.show()

    print("7. rest_ecg vs target")
    ctab = pd.crosstab(df['target'], df['rest_ecg'], normalize='index')
    ctab.plot(kind='bar', stacked=True, figsize=(8,6))
    plt.title('ECG Results by Target (%)')
    plt.show()

def feature_selection(df):

    df_enc = pd.get_dummies(df.drop('target', axis=1), drop_first=1)
    df_enc['target'] = df['target']
    X = df_enc.drop('target', axis=1)
    y = df_enc['target']

    chi = SelectKBest(chi2, k=9).fit(X,y)
    chi_feats = X.columns[chi.get_support()].tolist()
    print('Chi-selected features:', chi_feats)

    rf = RandomForestClassifier(n_estimators=100, random_state=65)
    rfe = RFE(rf, n_features_to_select=9)
    rfe.fit(X, y)
    rfe_feats = X.columns[rfe.support_].tolist()
    print('RFE-selected features:', rfe_feats)

    # Conducted the following code with both RFE_feats and chi_feats and the resulting
    # score for the cross validation gave a higher score for RFE_feats so we will proceed with RFE_feats
    # chi_feats score : 0.877 RFE_feats score : 0.892
    chosen = rfe_feats
    df_selected = df_enc[chosen + ['target']]
    df_selected.to_csv(SELECTED_FILE, index=False)
    print(f"Saved selected features to {SELECTED_FILE}")

    score = cross_val_score(rf, X[chosen], y, cv=5, scoring='roc_auc').mean()
    print(f"RF 5-fold CV AUC on selected features: {score: .3f}")

    return chi_feats, rfe_feats



def main():
    # Saves the loaded dataframe into df_raw
    df_raw = load_data()
    # Added to remove the outliers found in our EDA for the boxplot from Resting Blood Pressure by target
    Q1 = df_raw['trestbps'].quantile(0.25)
    Q3 = df_raw['trestbps'].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df_raw['trestbps'] = df_raw['trestbps'].clip(lower, upper)
    # Does Correlation Analysis between the attributes
    corr_analysis(df_raw)

    # Does further Exploratory Data Analysis on the data before building the model
    # Helps Find outliers and other imbalances within the data
    df_eda = df_raw.rename(columns=NAME_DICT)
    for col, mapping in MAPPING_DICTS.items():
        df_eda[col] = df_eda[col].map(mapping)
    
    perform_eda(df_eda)

    feature_selection(df_raw)



if __name__ == "__main__":
    main()