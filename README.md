## ** Heart Disease Predictor - UCI Dataset **

An end-to-end machine learning pipeline for predicting the presence of heart disease using UCI dataset. 

## ** Overview ** 

This project automates data ingestion, cleaning, exploratory analysis, feature selection, and model training to build a robust heart-disease classifier. The models included in this project is 
Logistic Regression, Random Forest, and Gradient Boosting, evaluated via cross-validated ROC-AUC and hold-out testing.

## ** Attribute Information **
```
 Features Utilized in the dataset:
 3 age: age in years
 4 sex: sex (1 = male; 0 = female)
 9 cp: chest pain type
         -- Value 1: typical angina
         -- Value 2: atypical angina
         -- Value 3: non-anginal pain
         -- Value 4: asymptomatic
 10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)
 12 chol: serum cholestoral in mg/dl
 16 fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
 19 restecg: resting electrocardiographic results
         -- Value 0: normal
         -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
 32 thalach: maximum heart rate achieved
 38 exang: exercise induced angina (1 = yes; 0 = no)
 40 oldpeak = ST depression induced by exercise relative to rest
 41 slope: the slope of the peak exercise ST segment
        # -- Value 1: upsloping
        # -- Value 2: flat
        # -- Value 3: downsloping
 44 ca: number of major vessels (0-3) colored by flourosopy
 51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
 58 target
```

## **Repository Structure**
```
Heart-Disease-Predictor/
├── Data/
│   ├── Raw/                 # Downloaded ZIPs and extracted files
│   ├── Interim/             # Combined/raw CSVs with headers
│   └── Processed/           # Cleaned Train.csv and Test.csv
│
├── src/
│   ├── data/
│   │   └── make_dataset.py  # Download & clean raw data pipeline
│   ├── features/
│   │   └── feature_selection.py  # EDA & feature-selection scripts
│   └── models/
│       └── model_creation.py    # Model training & evaluation
│
├── Data/Selected_Features/  # CSV of final selected features
├── src/visualization/       # Correlation heatmaps, EDA plots
├── README.md                # Project overview & instructions
└── requirements.txt         # Python dependencies

** Installation** 
1. Clone Repository
git clone https://github.com/charliegotcodes/Heart-Disease-Predictor.git
cd Heart-Disease-Predictor
3. Create virtual environment and install dependencies
python3 -m venv venv
source venv.bin.activate
pip install -r requirements.txt
```

** Results ** 

Logistic Regression: CV AUC = 0.89, Test AUC = 0.87, Accuracy = 82% 
Random Forest: CV AUC = 0.86, Test AUC = 0.87
Gradient Boosting: CV AUC = 0.85, Test AUC = 0.87
Best Model: Gradient Boosting (ROC-AUC = 0.874) or untuned Random Forest by Occam's razor
