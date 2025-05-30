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
from schemas import Season, League, LearningFeature, PredictedMatch, PredictedMatchOut, SeasonStats, Team, Match, PredictedMatchCreate, PredictedMatchPredictionResult
from sqlalchemy.orm import Session
from crud import get_seasons, get_season, get_learning_features, get_learning_features_for_team, get_predicted_match, get_predicted_matches, get_season_stats_for_team, get_team, get_teams, get_matches, get_match, get_league_table, get_season_stats, create_empty_prediction_match, delete_predicted_match, get_matches_by_season
from prediction_service import generate_and_store_match_prediction
from models import PredictedMatch as PredictedMatchDB


@asynccontextmanager
async def lifespan(app: FastAPI):
    #run_data_process()
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

@app.get("/seasons/", response_model=List[Season])
def read_seasons(skip: int=0, db: Session= Depends(get_db)):
    seasons = get_seasons(db, skip=skip)
    return seasons

@app.get("/seasons/{season_id}", response_model=Season)
def read_season(season_id: int, db:Session = Depends(get_db)):
    season = get_season(db, season_id=season_id)
    if season is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Season not found")
    return season

@app.get("/teams/", response_model=List[Team])
def read_teams(skip: int=0, db: Session = Depends(get_db)):
    teams = get_teams(db, skip=skip)
    return teams

@app.get("/teams/{team_id}", response_model=Team)
def read_team(team_id: int, db: Session = Depends(get_db)):
    db_team = get_team(db, team_id=team_id)
    if db_team is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    return db_team

@app.get("/matches/", response_model=List[Match])
def read_matches(skip: int=0, limit: int=50, db: Session = Depends(get_db)):
    matches = get_matches(db, skip=skip, limit=limit)
    return matches

@app.get("/matches/{match_id}", response_model=Match)
def read_match(match_id: int, db: Session = Depends(get_db)):
    db_match = get_match(db, match_id=match_id)
    if db_match is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match not found")
    return db_match

@app.get("/matches/season/{season_id}", response_model=List[Match])
def read_matches_by_season( season_id: int, skip: int=0, limit: int=50, db:Session = Depends(get_db)):
    matches = get_matches_by_season(db, season_id=season_id, skip=skip, limit=limit)
    return matches

@app.get("/leagues/{season_id}", response_model=List[League])
def read_league(season_id: int, db: Session = Depends(get_db)):
    db_league = get_league_table(db, season_id=season_id)
    if not db_league:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="League not found")
    return db_league

@app.get("/season_stats/season/{season_id}", response_model=List[SeasonStats])
def read_season_stats(season_id: int, db: Session = Depends(get_db)):
    db_season_stats = get_season_stats(db, season_id=season_id)
    if not db_season_stats:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Season stats not found")
    return db_season_stats
    
@app.get("/season_stats/team/{team_id}", response_model=List[SeasonStats])
def read_season_stats_for_team(team_id: int, db: Session = Depends(get_db)):
    db_season_stats_for_team = get_season_stats_for_team(db, team_id=team_id)
    if not db_season_stats_for_team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Season stats for team on id {team_id} not found")
    return db_season_stats_for_team

@app.get("/learning_features/season/{season_id}", response_model=List[LearningFeature])
def read_learning_features(season_id: int, db: Session = Depends(get_db)):
    db_learning_features = get_learning_features(db, season_id=season_id)
    if not db_learning_features:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning Features not found")
    return db_learning_features

@app.get("/learning_features/team/{team_id}", response_model=List[LearningFeature])
def read_learning_features_for_team(team_id: int, db: Session = Depends(get_db)):
    db_learning_features_for_team = get_learning_features_for_team(db, team_id=team_id)
    if not db_learning_features_for_team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning Features not found")
    return db_learning_features_for_team

@app.get("/predicted_matches/", response_model=List[PredictedMatch])
def read_predicted_matches(skip:int=0, db: Session = Depends(get_db)):
    db_predicted_matches = get_predicted_matches(db, skip=skip)
    if not db_predicted_matches:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Predicted Matches not found")
    return db_predicted_matches

@app.get("/predicted_matches/{match_id}", response_model=PredictedMatch)
def read_predicted_match(match_id: int, db: Session = Depends(get_db)):
    db_predicted_match = get_predicted_match(db, match_id=match_id)
    if not db_predicted_match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Predicted Match not found")
    return db_predicted_match

@app.post("/predicted_matches/create", response_model=PredictedMatchOut)
def create_empty_match(data: PredictedMatchCreate, db: Session = Depends(get_db)):
    if data.home_team_id == data.away_team_id:
        raise HTTPException(status_code=400, detail="Teams must be different")
    return create_empty_prediction_match(db, data.home_team_id, data.away_team_id)


@app.post("/predicted_matches/predict/{prediction_id}", response_model=PredictedMatchPredictionResult)
def run_prediction(prediction_id:int, db: Session = Depends(get_db)):
    match = db.query(PredictedMatchDB).filter(PredictedMatchDB.id == prediction_id).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match to predict not found")
    if match.is_predicted:
        raise HTTPException(status_code=400, detail="Match already predicted")
    return generate_and_store_match_prediction(db, prediction_id)

@app.delete("/predicted_matches/{prediction_id}")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    delete_predicted_match(db, prediction_id)
    return {"message": "Prediction deleted."}
                               

        
    


