from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Depends
from prometheus_fastapi_instrumentator import Instrumentator
from .routers import metrics, prediction
from .db import User, create_db_and_tables
from .schemas import UserCreate, UserRead, UserUpdate
from .users import auth_backend, current_active_user, fastapi_users

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan, version=os.environ['APP_VERSION'])

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

@app.get('/info')
def info():
    return {'version':app.version}

app.include_router(router=prediction.router)
app.include_router(router=metrics.router)

instrumentator = Instrumentator(excluded_handlers=[".*admin.*", "/metrics"],)
instrumentator.instrument(app).expose(app)
