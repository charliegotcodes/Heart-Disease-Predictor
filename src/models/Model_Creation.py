from pathlib import Path
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

BASE = Path(__file__).resolve().parent.parent.parent
RAW  = BASE / "Data" / "Processed"
FEATURES = BASE / "Data" / "Selected_Features" / "selected_features.csv"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "Data" / "Processed"
FEATURES_FILE = BASE_DIR / "Data" / "Selected_Features" / "selected_features.csv"

df = pd.read_csv(FEATURES_FILE)
X = df.drop('target', axis=1)
y = df['target']

# Train & Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=64)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Logistic Regression
print("\n Logistic Regression (CV ROC-AUC)")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Mean AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Fit & evaluate on test
lr.fit(X_train, y_train)
y_predict_probability = lr.predict_proba(X_test)[:,1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_predict_probability):.3f}")
print(classification_report(y_test, lr.predict(X_test)))

# Random Forest 
print("\n Random Forest (CV ROC-AUC)")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=64)
cvs_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring = 'roc_auc')
print(f"Mean AUC: {cvs_rf.mean():.3f} +/- {cvs_rf.std():.3f}")

# Fit & evaluate on test
rf.fit(X_train, y_train)
y_predict_probability = rf.predict_proba(X_test)[:,1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_predict_probability):.3f}")
print(classification_report(y_test, rf.predict(X_test)))

# Gradient Boosting
# When looking at the results from the classification report there is slightly lower precision/recall on the negative class
print("\n Gradient Boosting (CV ROC-AUC)")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=64)
cvs_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Mean AUC: {cvs_gb.mean():.3f} +/- {cvs_gb.std():.3f}")

gb.fit(X_train, y_train)
y_predict_probability = gb.predict_proba(X_test)[:,1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_predict_probability):.3f}")
print(classification_report(y_test, gb.predict(X_test)))

# Random Forest Hyper-parameter tuning

print("\n Random Forest Hyper Tuning")
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 5, 8, 10],
    'min_samples_split': [2, 4, 5]
}

grid = GridSearchCV( RandomForestClassifier(class_weight='balanced', random_state=64), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV AUC: {grid.best_score_:.3f}")

best_rf = grid.best_estimator_
y_predict_probability = best_rf.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC (best RF): {roc_auc_score(y_test, y_predict_probability):.3f}")
print(classification_report(y_test, best_rf.predict(X_test)))

# After viewing the following results its clear that the test size is quite small with 61 samples and thus test results are noisy.
# Another outlook is that Logistic Regression is ideal but with tuning CV performance of Random Forest increased to 0.88 on training data.
# However, Random forest untuned on test values had a higher AUC score compared to tuned so preferably untuned performed better on Highest hold-out AUC 
# Therefore with the possible problem of over-fitting its preferable by occams razor to choose untuned