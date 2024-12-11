from contextlib import asynccontextmanager
from evidently import ColumnMapping
from fastapi import FastAPI
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import Callable, List
from prometheus_fastapi_instrumentator.metrics import Info
from prometheus_client import Counter, Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from sklearn import datasets
from fastapi import FastAPI
from datetime import datetime
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler  # runs tasks in the background
from apscheduler.triggers.cron import CronTrigger  # allows us to specify a recurring time for execution

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
    # Load the ML model
    ml_models["model"] = mlflow.sklearn.load_model('model')
    yield
    ml_models.clear()
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

class Iris(BaseModel):
    data: List[List[float]]

@app.get("/")
def home():
    return "Hello World..."


@app.post('/predict', tags=["predictions"])
def get_prediction(iris: Iris):
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

instrumentator = Instrumentator(excluded_handlers=[".*admin.*", "/metrics"],)
instrumentator.instrument(app).expose(app)
