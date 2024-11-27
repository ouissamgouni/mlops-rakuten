import flytekit
from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import pickle
import os


@task
def load_data() -> FlyteDirectory:
    print("Loading data...")
    X, y = datasets.load_iris(return_X_y=True)

    working_dir = flytekit.current_context().working_directory
    raw_dir = os.path.join(working_dir, "raw")

    if not os.path.exists(raw_dir): 
        os.makedirs(raw_dir)

    x_path = os.path.join(raw_dir, "X.pkl")
    with open(x_path, mode="wb") as output_file:
        pickle.dump(X, output_file) 

    y_path = os.path.join(raw_dir, "y.pkl")
    with open(y_path, mode="wb") as output_file:
        pickle.dump(y, output_file) 

    return FlyteDirectory(path=raw_dir)

@task
def preprocess(raw_data_dir: FlyteDirectory)->FlyteDirectory:
    print(f"Preprocessing the data from{raw_data_dir}...")
    return raw_data_dir

  
@task
def train(preproc_dir: FlyteDirectory)-> FlyteDirectory:
    print("Training the model...")
    working_dir=flytekit.current_context().working_directory

    x_path = os.path.join(preproc_dir, "X.pkl")
    with open(x_path, mode="rb") as f:
        X=pickle.load(f) 

    y_path = os.path.join(preproc_dir, "y.pkl")
    with open(y_path, mode="rb") as f:
        y=pickle.load(f) 


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dir = os.path.join(working_dir, "train")
    if not os.path.exists(train_dir): 
        os.makedirs(train_dir)

    x_test_path = os.path.join(train_dir, "X_test.pkl")
    with open(x_test_path, mode="wb") as f:
        pickle.dump(X_test, f) 

    y_test_path = os.path.join(train_dir, "y_test.pkl")
    with open(y_test_path, mode="wb") as f:
        pickle.dump(y_test, f) 

    # Define the model hyperparameters
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 8888,
    }

    model_params_path = os.path.join(train_dir, "model_params.pkl")
    with open(model_params_path, mode="wb") as f:
        pickle.dump(params, f)

    # Train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    save_to=os.path.join(train_dir, "model.pkl")

    with open(save_to, 'wb') as f:
        pickle.dump(model, f)

    return FlyteDirectory(path=train_dir)


@task
def validate(train_dir: FlyteDirectory, mlflow_host:str)->bool:
    print("Validating the model...")

    model_path=os.path.join(train_dir, "model.pkl")
    with open(model_path, 'rb') as f:
        model=pickle.load(f)

    model_params_path=os.path.join(train_dir, "model_params.pkl")
    with open(model_params_path, 'rb') as f:
        params=pickle.load(f)

    x_test_path = os.path.join(train_dir, "X_test.pkl")
    with open(x_test_path, mode="rb") as f:
        X_test=pickle.load(f) 

    y_test_path = os.path.join(train_dir, "y_test.pkl")
    with open(y_test_path, mode="rb") as f:
        y_test=pickle.load(f) 

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test,y_pred)
    f1_score_= f1_score(y_test, y_pred, average='macro')

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=f"http://{mlflow_host}")

    # Create a new MLflow Experiment
    mlflow.set_experiment("rakuten-pcat")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1-score", f1_score_)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        #signature = infer_signature(X_train, lr.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            #signature=signature,
            #input_example=X_train,
            registered_model_name="rakuten-pcat",
        )
    return True

@task
def release(status:bool)->str:
    # pushing the model to mlflow
    print("Releasing the model...")
    return "Done"

@workflow
def ml_worfkow(mlflow_host:str) -> str:
    res=load_data()
    res=preprocess(res)
    train_dir=train(res)
    res=validate(train_dir,mlflow_host)
    return release(res)

if __name__ == "__main__":
    print(f"Running ml_worfkow() {ml_worfkow("localhost:5000")}")