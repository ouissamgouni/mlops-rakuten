from collections.abc import AsyncGenerator
import os

from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy import TIMESTAMP, Column, Integer, String, text, BigInteger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL =  os.environ.get('DATABASE_URL', "postgresql+asyncpg://postgres:ODFLjD59Jz@localhost:5432/rakutencp")


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    pass


engine = create_async_engine(DATABASE_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)

async def get_predictions_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)

class Prediction(Base):

    __tablename__ = "predictions"    

    prediction_batch_id = Column(String,primary_key=True,nullable=False)
    provided_index = Column(BigInteger,primary_key=True,nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    designation = Column(String,nullable=False)
    description = Column(String,nullable=False)
    product_id = Column(BigInteger,nullable=False)
    image_id = Column(BigInteger,nullable=False)
    txt_model_input = Column(String,nullable=False)
    prediction = Column(Integer,nullable=False)
    ground_truth = Column(Integer,nullable=True)
    ground_truth_at = Column(TIMESTAMP(timezone=True),nullable=True)