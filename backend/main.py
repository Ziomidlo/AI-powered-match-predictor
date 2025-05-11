from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from models import Base
from database import engine

app = FastAPI()

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