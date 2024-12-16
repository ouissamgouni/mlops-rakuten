from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import List
from routers import metrics  
from .db import User, create_db_and_tables
from .schemas import UserCreate, UserRead, UserUpdate
from .users import auth_backend, current_active_user, fastapi_users
from .routers import prediction


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    ml_models["model"] = mlflow.sklearn.load_model('models/mlflow/model')
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),prefix="/auth",tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),prefix="/auth",tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),prefix="/auth",tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),prefix="/users",tags=["users"],
)


class Iris(BaseModel):
    data: List[List[float]]

@app.post('/dummy', tags=["predictions"])
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



app.include_router(router=prediction.router)
app.include_router(router=metrics.router)

instrumentator = Instrumentator(excluded_handlers=[".*admin.*", "/metrics"],)
instrumentator.instrument(app).expose(app)
