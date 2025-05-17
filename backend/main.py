from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
#from data_to_db import data_to_db
from models import Base
from database import engine
from contextlib import asynccontextmanager
from data_processing import run_data_process
from data_learning import train_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    run_data_process()
    train_models()
    #data_to_db()
    yield

app = FastAPI(lifespan=lifespan)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


@app.get("/")
def root():
    return {"message": "Backend API running!"}

