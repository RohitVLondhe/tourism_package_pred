# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism_prediction")

api = HfApi()


Xtrain_path = "hf://datasets/Pvt-Pixel/tourism-package-pred/Xtrain.csv"
Xtest_path = "hf://datasets/Pvt-Pixel/tourism-package-pred/Xtest.csv"
ytrain_path = "hf://datasets/Pvt-Pixel/tourism-package-pred/ytrain.csv"
ytest_path = "hf://datasets/Pvt-Pixel/tourism-package-pred/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# All features are numerical after prep.py's LabelEncoding
numeric_features = Xtrain.columns.tolist()


# Set the class weight to handle class imbalance
# Ensure ytrain is treated as a Series for value_counts
ytrain_series = ytrain.iloc[:, 0] if isinstance(ytrain, pd.DataFrame) and ytrain.shape[1] == 1 else ytrain

class_counts = ytrain_series.value_counts()

# Safely get counts for class 0 and 1, defaulting to 0 if not present
count_0 = class_counts.get(0, 0)
count_1 = class_counts.get(1, 0)

if count_1 > 0:
    class_weight = count_0 / count_1
else:
    # If there are no positive examples, set class_weight to 1.0 (no scaling)
    # This implies the model will train only on negative examples, which is expected
    # if class 1 is genuinely absent in the training set.
    class_weight = 1.0
    print("Warning: No positive examples (class 1) found in the training data. `scale_pos_weight` will be set to 1.0.")

print(f"Calculated class weight: {class_weight}")

# Define the preprocessing steps: only StandardScaler as all features are numerical
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__colsample_bytree': [0.5],
    'xgbclassifier__colsample_bylevel': [0.5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='recall')
    grid_search.fit(Xtrain, ytrain.values.ravel())

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_recall", mean_score)
            mlflow.log_metric("std_test_recall", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45 # Using 0.45 for better recall as suggested for imbalanced datasets

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Pvt-Pixel/tourism_package_model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",
        path_in_repo="best_tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
