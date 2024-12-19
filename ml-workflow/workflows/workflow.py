import joblib
from flytekit import task, workflow
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
from mlflow.models import infer_signature
import os
from typing import Tuple


@task
def load_data(data_selector:str) -> Tuple[np.ndarray,np.ndarray]:
    print("Loading data...")
    X, y = datasets.load_iris(return_X_y=True, as_frame=False)
    return (X,y)

@task
def preprocess(X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    print(f"Preprocessing the data...")
    return X,y

  
@task
def train(X:np.ndarray, y:np.ndarray)-> Tuple[np.ndarray,np.ndarray,np.ndarray,dict,FlyteFile]:
    print("Training the model...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
    }

    # Train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    model_path="model.joblib"
    joblib.dump(value=model, filename=model_path)

    return X_test, y_test, X_train, params, model_path


@task
def validate(X_test:np.ndarray, 
             y_test:np.ndarray, 
             X_train:np.ndarray, 
             params:dict, 
             model_path:FlyteFile)->bool:
    print("Validating the model...")
  
    model=joblib.load(model_path)
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test,y_pred)
    f1_score_= f1_score(y_test, y_pred, average='macro')

    # Set our tracking server uri for logging
    mlflow_host=os.getenv(key="MLFLOW_HOST", default = "localhost:5000")
    mlflow.set_tracking_uri(uri=f"http://{mlflow_host}")

    # Create a new MLflow Experiment
    mlflow.set_experiment("rakuten-pcat")

    # Start an MLflow clrun
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1-score", f1_score_)

        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,
            registered_model_name="rakuten-pcat",
        )
    return True

@task
def release(validation_status:bool)->str:
    # registring the model
    print("Releasing the model...")
    return "947b7b20acb24d6397b40f324a60a2ce"

@workflow
def ml_workflow(data_selector:str) -> str:
    X,y=load_data(data_selector)
    X,y=preprocess(X,y)
    X_test, y_test, X_train, params, model_path=train(X,y)
    validate_res=validate(X_test, y_test, X_train, params, model_path)
    release_res=release(validate_res)
    return release_res

if __name__ == "__main__":
    res=ml_workflow(data_selector="")
    print(f"Running ML worfkow {res}")