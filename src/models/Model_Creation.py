from pathlib import Path
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score

current_dir = Path(__file__).resolve().parent.parent.parent



def load_selected_features(filepath):
    print(filepath / "Data" / "Selected_Features")
    final_features = pd.read_csv(filepath / "Data" / "Selected_Features" / "selected_features.csv")
    
    return final_features.columns.tolist()

def load_Train_Test(filepath, file_name):
    df = pd.read_csv(filepath / "Data" / "Processed" / file_name)
    return df

def train_Model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state = 42)
    model.fit(X_train, y_train)
    return model

def test_model_evaluation(model, X_test, y_test):
    y_prediction = model.predict(X_test)
    accuracy = accuracy_score(y_prediction, y_test)
    print("Accuracy of the model: ", accuracy, "\n")
    return accuracy



def main():
    final_features =  load_selected_features(current_dir)
    print(final_features)

    df_Train = load_Train_Test(current_dir, "Train.csv")
    df_Test = load_Train_Test(current_dir, "Test.csv")
    
    X_train, y_train = df_Train[final_features], df_Train["target"]
    X_test, y_test = df_Test[final_features], df_Test["target"]

    model = train_Model(X_train, y_train)
    accuracy = test_model_evaluation(model, X_test, y_test)
    print("Initial Iteration of RFC's accuracy: ", accuracy, "\n")

    # Attempt to increase the accuracy of the model through Tuning Hyperparameters
    








if __name__ == "__main__":
    main()