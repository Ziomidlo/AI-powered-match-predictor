from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Season(Base):
    __tablename__ = "seasons"

    id = Column(Integer, primary_key=True)
    season = Column(String)


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True)
    team_name = Column(String, unique=True)
    venue = Column(String)


class Match(Base):
    __tablename__ = "matches"
    
    id= Column(Integer, primary_key=True)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)
    match_date = Column(String)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    result_numeric = Column(Integer)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    home_xG = Column(Float)
    away_xG = Column(Float)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    season = relationship("Season", foreign_keys=[season_id])

class League(Base):
    __tablename__ = "leagues"

    id = Column(Integer, primary_key=True)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    position = Column(Integer)
    games_played = Column(Integer)
    wins = Column(Integer)
    draws = Column(Integer)
    losses = Column(Integer)
    goals_for = Column(Integer)
    goals_against = Column(Integer)
    goals_difference = Column(Integer)
    points = Column(Integer)

    team = relationship("Team", foreign_keys=[team_id])
    season = relationship("Season", foreign_keys=[season_id])

class SeasonStats(Base):
    __tablename__ = "season_stats"

    id = Column(Integer, primary_key=True)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    shots = Column(Float)
    shots_on_target = Column(Float)
    free_kicks = Column(Float)
    penalty_goals = Column(Float)
    passes_completed = Column(Float)
    passes_attempted = Column(Float)
    pass_completion = Column(Float)
    corners = Column(Float)
    yellow_cards = Column(Float)
    red_cards = Column(Float)
    fouls_conceded = Column(Float)
    penalties_conceded = Column(Float)
    own_goals = Column(Float)

    team = relationship("Team", foreign_keys=[team_id])
    season = relationship("Season", foreign_keys=[season_id])


class LearningFeature(Base):
    __tablename__ = "learning_features"
    
    id = Column(Integer, primary_key=True)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    goals_per_match = Column(Float)
    offensive_strength = Column(Float)
    goals_conceded_per_match = Column(Float)
    defense_strength = Column(Float)
    effective_possession = Column(Float)
    passing_and_control_strength = Column(Float)
    penalties_converted_rate = Column(Float)
    set_piece_strength = Column(Float)
    team_power_index = Column(Float)
    avg_position = Column(Float)
    avg_points = Column(Float)
    
    team = relationship("Team", foreign_keys=[team_id])
    season = relationship("Season", foreign_keys=[season_id])

class PredictedMatch(Base):
    __tablename__ = "predicted_matches"

    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    is_predicted = Column(Boolean)
    home_win_probability_lr = Column(Float)
    draw_probability_lr = Column(Float)
    away_win_probability_lr = Column(Float)
    home_win_probability_rfc = Column(Float)
    draw_probability_rfc = Column(Float)
    away_win_probability_rfc = Column(Float)
    home_win_probability_xgb = Column(Float)
    draw_probability_xgb = Column(Float)
    away_win_probability_xgb = Column(Float)
    home_win_probability_svc = Column(Float)
    draw_probability_svc = Column(Float)
    away_win_probability_svc = Column(Float)
    home_expected_goals_lr = Column(Float)
    away_expected_goals_lr = Column(Float)
    home_expected_goals_rfr = Column(Float)
    away_expected_goals_rfr = Column(Float)
    home_expected_goals_xgb = Column(Float)
    away_expected_goals_xgb = Column(Float)
    home_expected_goals_svr = Column(Float)
    away_expected_goals_svr = Column(Float)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])









    
    


