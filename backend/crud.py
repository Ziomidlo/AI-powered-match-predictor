from sqlalchemy.orm import Session
from models import Team

def get_team(db: Session, team_id: int):
    return db.query(Team).filter(Team.id == team_id).first()

def get_teams(db: Session, skip: int = 0):
    return db.query(Team).offset(skip).all()