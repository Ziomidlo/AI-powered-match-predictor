from sqlalchemy import Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Match(Base):
    __tablename__ = "matches"
    
    id= Column(Integer, primary_key=True, index=True)
    home_team = Column(String)
    away_team = Column(String)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    date = Column(Date)