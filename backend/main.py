from typing import List
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
#from data_to_db import data_to_db
from models import Base, Team as DBTeam
from database import engine, get_db
from contextlib import asynccontextmanager
#from data_processing import run_data_process
from data_learning import train_models
from schemas import Team
from sqlalchemy.orm import Session
from crud import get_team, get_teams


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    #run_data_process()
    train_models()
    #data_to_db()
    yield

app = FastAPI(lifespan=lifespan)

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

@app.get("/teams/", response_model=List[Team])
def read_teams(skip: int=0, db: Session = Depends(get_db)):
    teams = get_teams(db, skip)
    return teams

@app.get("/teams/{team_id}", response_model=Team)
def read_team(team_id: int, db: Session = Depends(get_db)):
    db_team = get_team(db, team_id=team_id)
    if db_team is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
