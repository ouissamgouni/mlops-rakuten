from contextlib import asynccontextmanager
import datetime
import os
from typing import Annotated
from fastapi import Depends, APIRouter, FastAPI, HTTPException, Request
import pandas as pd
from prometheus_client import Counter, Gauge, Info
from pydantic import BaseModel
from prometheus_client import Gauge
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import  func, select
from ..db import get_async_session, Prediction, async_session_maker
from pytz import utc
from sqlalchemy.ext.asyncio import AsyncSession


gauge_acc = Gauge('accuracy', 'accuracy')
gauge_precision = Gauge('precision', 'precision')
gauge_recall = Gauge('recall', 'recall')
gauge_f1 = Gauge('f1_score', 'f1-score')
gauge_pred = Gauge('nb_preds', 'number of predictions')
gauge_labeled_pred = Gauge('nb_labeled_preds', 'number of labeled predictions')

i = Info('app_version', 'Rakuten product category predictor version')
i.info({'version': os.environ['APP_VERSION']})

DBSessionDep = Annotated[AsyncSession, Depends(get_async_session)]


async def evaluate_prediction_batch(prediction_batch_id:int, db_session:DBSessionDep):
    labeled_predictions = (await db_session.scalars(select(Prediction)\
                                        .filter(Prediction.prediction_batch_id==prediction_batch_id)))\
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
    data_drift_report.run(current_data=current, reference_data=reference, column_mapping=column_mapping)
    metrics = {}
    report_dict = data_drift_report.as_dict()
    metrics["accuracy"] = report_dict["metrics"][0]["result"]["current"]["accuracy"]
    metrics["precision"] = report_dict["metrics"][0]["result"]["current"]["precision"]
    metrics["recall"] = report_dict["metrics"][0]["result"]["current"]["recall"]
    metrics["f1-score"] = report_dict["metrics"][0]["result"]["current"]["f1"]

    return metrics

async def evaluate(app: FastAPI, x_last_pred:int| None =os.environ['EVAL_ON_X_LAST_PRED']):
    #limit = os.environ['EVAL_ON_X_LAST_PRED']
    print(f"Evaluating the model at {datetime.now()} on last {x_last_pred} labeled predictions")
    async with async_session_maker() as db_session:
        labeled_predictions = (await db_session.scalars(select(Prediction)\
                                            .filter(Prediction.ground_truth.is_not(None),Prediction.app_version==app.version )\
                                            .order_by(Prediction.ground_truth_at.desc())\
                                                .limit(limit=x_last_pred)))\
                                                .fetchall()
    df=pd.DataFrame([vars(i) for i in labeled_predictions])

    if df.size==0:
        raise HTTPException(status_code=404, detail=f"No predictions founds made by system version '{ app.version }'")

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
    metrics = {}
    report_dict = data_drift_report.as_dict()
    metrics["accuracy"] = report_dict["metrics"][0]["result"]["current"]["accuracy"]
    metrics["precision"] = report_dict["metrics"][0]["result"]["current"]["precision"]
    metrics["recall"] = report_dict["metrics"][0]["result"]["current"]["recall"]
    metrics["f1-score"] = report_dict["metrics"][0]["result"]["current"]["f1"]

    

    async with async_session_maker() as db_session:
        metrics["nb_predictions"] = (await db_session.scalar(select(func.count(Prediction.created_at))\
                                            .where(Prediction.app_version==app.version )))
        

        metrics["nb_labeled_predictions"] = (await db_session.scalar(select(func.count(Prediction.created_at))\
                                            .filter(Prediction.ground_truth.is_not(None),Prediction.app_version==app.version )))

       

    gauge_acc.set(metrics["accuracy"])
    gauge_precision.set(metrics["precision"])
    gauge_recall.set(metrics["recall"])
    gauge_f1.set(metrics["f1-score"])

    gauge_pred.set( metrics["nb_predictions"])
    gauge_labeled_pred.set(metrics["nb_labeled_predictions"])

    print(f"Metrics logged to Prometheus", metrics)

    return metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler(timezone=utc)
    scheduler.start()
    scheduler.add_job(func=evaluate, trigger='interval', args=[app], minutes=1)
    yield
    scheduler.shutdown()
    

router = APIRouter(lifespan=lifespan)

@router.get('/evaluate', tags=["metrics"])
async def call_evaluate(request:Request, x_last_pred:int| None=None):
    return await evaluate(request.app, x_last_pred)

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