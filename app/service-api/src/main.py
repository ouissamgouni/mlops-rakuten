from contextlib import asynccontextmanager
from fastapi import FastAPI
import mlflow
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import List

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["model"] = mlflow.sklearn.load_model('model')
    #print(type(model))
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class Iris(BaseModel):
    data: List[List[float]] #Annotated[list[float], annotated_types.Len(min_length=4)]

@app.get("/")
def home():
    return "Hello World"


@app.post('/predict', tags=["predictions"])
def get_prediction(iris: Iris):
    data = dict(iris)['data']
    model =  ml_models["model"]
    prediction = model.predict(data).tolist()
    log_proba =  model.predict_proba(data).tolist()
    return {"prediction": prediction,
            "log_proba": log_proba}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Instrumentator().instrument(app).expose(app)

Instrumentor.new_merit(){

    # connect to bd

    # calculer f1

    # return f1-score
    
}
