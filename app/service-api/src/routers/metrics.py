import datetime
from typing import Annotated
from fastapi import Depends, FastAPI, APIRouter
import pandas as pd
from prometheus_client import Gauge
from pydantic import BaseModel
from prometheus_client import Gauge
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select
from src.db import get_async_session, Prediction
from sqlalchemy.ext.asyncio import AsyncSession


gauge_acc = Gauge('accuracy', 'accuracy')
gauge_precision = Gauge('precision', 'precision')
gauge_recall = Gauge('recall', 'recall')
gauge_f1 = Gauge('f1_score', 'f1-score')

DBSessionDep = Annotated[AsyncSession, Depends(get_async_session)]

async def evaluate(db_session:DBSessionDep):
    print(f"Evaluating the model at {datetime.now()}")

    labeled_predictions = (await db_session.scalars(select(Prediction)\
                                            .filter(Prediction.ground_truth.is_not(None))))\
                                                .fetchall()
    df=pd.DataFrame([vars(i) for i in labeled_predictions])

    current = df
    reference =current

    data_drift_report = Report(metrics=[
        ClassificationPreset(),
        ])

    column_mapping = ColumnMapping()

    column_mapping.target = 'ground_truth'
    column_mapping.prediction = 'prediction'
    #column_mapping.target_names = ['Setosa', 'Versicolour', 'Virginica']
    data_drift_report.run(current_data=current, reference_data=reference, column_mapping=column_mapping)
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
scheduler.add_job(evaluate, 'interval',args=[DBSessionDep], minutes=5)
scheduler.start()

router = APIRouter()

@router.get('/evaluate', tags=["metrics"])
async def call_evaluate(db_session:DBSessionDep):
    return await evaluate(db_session)

class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1score: float

@router.post('/dummy-metrics', tags=["metrics"])
def push_metrics(metrics:Metrics):
    gauge_acc.set(metrics.accuracy)
    gauge_precision.set(metrics.precision)
    gauge_recall.set(metrics.recall)
    gauge_f1.set(metrics.f1score)