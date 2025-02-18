from pathlib import Path
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
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
    model = RandomForestClassifier( max_depth =5, random_state = 42)
    model.fit(X_train, y_train)
    return model

def test_model_evaluation(model, X_test, y_test):
    y_prediction = model.predict(X_test)
    accuracy = accuracy_score(y_prediction, y_test)
    print("Accuracy of the model: ", accuracy, "\n")
    return accuracy

def hyperparam_tuning(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid, cv = 5, n_jobs = 1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)



def main():
    final_features =  load_selected_features(current_dir)
    final_features = final_features[:-1]
    print(final_features)

    df_Train = load_Train_Test(current_dir, "Test.csv")
    df_Test = load_Train_Test(current_dir, "Train.csv")

    print("Shape of Test Data :", df_Test.shape)
    print("Shape of Train Data :", df_Train.shape)
    # X_train, X_test, Y_train, Y_test = train_test_split(df_Train[final_features], df_Train["target"], test_size=0.2, random_state=0)
    # print("Shape of Test Data :", X_train.shape)
    train_predictor = df_Train.drop("target", axis=1)
    test_predictor = df_Test.drop("target", axis=1)
    X_train, y_train = train_predictor, df_Train["target"]
    X_test, y_test = test_predictor, df_Test["target"]

    model = train_Model(X_train, y_train)
    accuracy = test_model_evaluation(model, X_test, y_test)
    print("Initial Iteration of RFC's accuracy: ", accuracy, "\n")

    # Attempt to increase the accuracy of the model through Tuning Hyperparameters
    # hyperparam_tuning(X_train, y_train)
    #Best Parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}







if __name__ == "__main__":
    main()