from contextlib import asynccontextmanager
from evidently import ColumnMapping
from fastapi import FastAPI, Depends
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import List
from prometheus_client import Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from sklearn import datasets
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler  
from .db import User, create_db_and_tables
from .schemas import UserCreate, UserRead, UserUpdate
from .users import auth_backend, current_active_user, fastapi_users
from .routers import prediction


ml_models = {}

gauge_acc = Gauge('accuracy', 'accuracy')
gauge_precision = Gauge('precision', 'precision')
gauge_recall = Gauge('recall', 'recall')
gauge_f1 = Gauge('f1_score', 'f1-score')

def evaluate():
    print(f"Evaluating the model at {datetime.now()}")
    model =  get_model()

    iris_data = datasets.load_iris(as_frame=True)
    iris_frame = iris_data.frame
    iris_ref = iris_frame.sample(n=150, replace=False)
    iris_cur = iris_frame.sample(n=150, replace=False)
    #Reference and current data for Multiclass classification, option 1
    iris_ref['prediction'] = model.predict(iris_ref[iris_data.feature_names])
    iris_cur['prediction'] = model.predict(iris_cur[iris_data.feature_names])

    data_drift_report = Report(metrics=[
        ClassificationPreset(),
        ])

    column_mapping = ColumnMapping()

    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'
    column_mapping.target_names = ['Setosa', 'Versicolour', 'Virginica']
    data_drift_report.run(current_data=iris_cur, reference_data=iris_ref, column_mapping=column_mapping)
    data_drift_report.save_json('report.json')

    metrics = {}
    report_dict = data_drift_report.as_dict()
    metrics["accuracy"] = report_dict["metrics"][0]["result"]["current"]["accuracy"]
    metrics["precision"] = report_dict["metrics"][0]["result"]["current"]["precision"]
    metrics["recall"] = report_dict["metrics"][0]["result"]["current"]["recall"]
    metrics["f1-score"] = report_dict["metrics"][0]["result"]["current"]["f1"]

    gauge_acc.set(metrics["accuracy"])
    gauge_precision.set(metrics["precision"])
    gauge_recall.set(metrics["recall"])
    gauge_f1.set(metrics["f1-score"])

    print(f"Metrics logged to Prometheus")

    return metrics

# Set up the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(evaluate, 'interval', minutes=15)
scheduler.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    ml_models["model"] = mlflow.sklearn.load_model('models/mlflow/model')
    yield
    ml_models.clear()
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)



class Iris(BaseModel):
    data: List[List[float]]

@app.post('/predict-sample', tags=["predictions"])
def get_prediction_sample(iris: Iris, user: User = Depends(current_active_user)):
    print(f"Hello {user.email}!")
    data = dict(iris)['data']
    model = get_model()
    prediction = model.predict(data).tolist()
    log_proba =  model.predict_proba(data).tolist()
    return {"prediction": prediction,
            "log_proba": log_proba}

def get_model():
    model =  ml_models["model"]
    return model


@app.post('/evaluate')
def call_evaluate():
    return evaluate()

class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1score: float

@app.post('/push-metrics')
def push_metrics(metrics:Metrics):
    gauge_acc.set(metrics.accuracy)
    gauge_precision.set(metrics.precision)
    gauge_recall.set(metrics.recall)
    gauge_f1.set(metrics.f1score)


app.include_router(router=prediction.router)

instrumentator = Instrumentator(excluded_handlers=[".*admin.*", "/metrics"],)
instrumentator.instrument(app).expose(app)
