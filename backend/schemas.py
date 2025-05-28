from typing import Optional
from pydantic import BaseModel, Field

class SeasonBase(BaseModel):
    season: str = Field(examples=['2024/2025', '2023/2024'], description="Season naming")
    model_config = {"from_attributes": True}

class TeamBase(BaseModel):
    team_name: str = Field(examples=["Manchester United FC", "Liverpool FC"], description="Name of the Team")
    venue: str = Field(examples=["Old Trafford", "Anfield"], description="Name of the stadium")
    model_config = {"from_attributes": True}

class MatchBase(BaseModel):
    match_date: str = Field(examples=['2025-11-11', '2024-07-15'], description="Date of the match")
    home_goals: int
    away_goals: int
    home_xG: float = Field(examples=[2.24, 0.49], description="home expected goals in match")
    away_xG: float = Field(examples=[1.4, 3.55], description="away expected goals in match")
    result_numeric: int = Field(examples=[1, 0 , -1], description=("Classificator of result 1 for home win, 0 for draw and -1 for away win"))
    model_config = {"from_attributes" : True }

class LeagueBase(BaseModel):
    position: int = Field(description="League position of the team")
    games_played: int = Field(description="Number of matches played by team")
    wins: int
    draws: int
    losses: int
    goals_for: int = Field(description="Number of scored goals by team")
    goals_against: int = Field(description="Number of conceded goals by team")
    goals_difference: int = Field(examples=[13, 0, -5], description="Difference of scored and conceded goals by team")
    points:int = Field(description="Points gained by team")
    model_config = {"from_attributes" : True }

class SeasonStatsBase(BaseModel):
    shots: float = Field(description="Number of shots attempted")
    shots_on_target: float = Field(description="Number of shots on target")
    free_kicks: float = Field(description="Number of free kicks gained by team")
    penalty_goals: float = Field(description="Goals scored by penalty kicks")
    passes_completed: float = Field(description="Passes that has been attempted and not lost by team")
    passes_attempted: float
    pass_completion: float = Field(description="Percent of pass attempted that has been also completed")
    corners: float = Field(description="Number of corner kicks gained")
    yellow_cards: float = Field(description="Number of yellow cards gained")
    red_cards: float = Field(description="Number of red cards gained")
    fouls_conceded: float
    penalties_conceded: float
    own_goals: float = Field(description="Number of own goals conceded")
    model_config = {"from_attributes" : True }


class LearningFeatureBase(BaseModel):
    goals_per_match: float
    offensive_strength: float = Field(description="Statistics of team offensive strength")
    goals_conceded_per_match: float
    defense_strength: float = Field(description="Statistics of team defense strength")
    effective_possession: float
    passing_and_control_strength: float = Field(description="Statistics of team passing and control strength")
    penalties_converted_rate: float = Field(description="Field of penalties scored")
    set_piece_strength: float = Field(description="Statistics of team set piece strength")
    team_power_index: float = Field(description="Statistics of overall team strength")
    avg_position: float = Field(description="Average position of team in league")
    avg_points: float = Field(description="Average points gained by team in league")
    model_config = {"from_attributes" : True }

class PredictedMatchBase(BaseModel):
    is_predicted: bool = Field(description="Declaration if match has been already predicted")
    home_win_probability_lr: Optional[float] = Field(description="Home win probability by logistic regresion")
    draw_probability_lr: Optional[float] = Field(description="Draw probability by logistic regresion")
    away_win_probability_lr: Optional[float] = Field(description="Away win probability by logistic regresion")
    home_win_probability_rfc: Optional[float] = Field(description="Home win probability by random forrest classifier")
    draw_probability_rfc: Optional[float] = Field(description="Draw probability by random forrest classifier")
    away_win_probability_rfc: Optional[float] = Field(description="Away win probability by random forrest classifier")
    home_win_probability_xgb: Optional[float] = Field(description="Home win probability by XGBoost classifier")
    draw_probability_xgb: Optional[float] = Field(description="Draw probability by XGBoost classifier")
    away_win_probability_xgb: Optional[float] = Field(description="Away win probability by XGBoost classifier")
    home_win_probability_svc: Optional[float] = Field(description="Home win probability by Support Vector Classifier")
    draw_probability_svc: Optional[float] = Field(description="Draw probability by Support Vector Classifier")
    away_win_probability_svc: Optional[float] = Field(description="Away win probability by Support Vector Classifier")
    home_expected_goals_lr: Optional[float] = Field(description="Home expected goals by linear regresion")
    away_expected_goals_lr: Optional[float] = Field(description="Away expected goals by linear regresion")
    home_expected_goals_rfr: Optional[float] = Field(description="Home expected goals by random forrest regresion")
    away_expected_goals_rfr: Optional[float] = Field(description="Away expected goals by random forrest regresion")
    home_expected_goals_xgb: Optional[float] = Field(description="Home expected goals by XGBoost regressor")
    away_expected_goals_xgb: Optional[float] = Field(description="Away expected goals by XGBoost regressor")
    home_expected_goals_svr: Optional[float] = Field(description="Home expected goals by Support Vector regressor")
    away_expected_goals_svr: Optional[float] = Field(description="Away expected goals by Support Vector regressor")
    model_config = {"from_attributes" : True }


class Season(SeasonBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the season")
    model_config = {"from_attributes" : True }
    
class Team(TeamBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the team")

    model_config = {"from_attributes" : True }

class Match(MatchBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the match")
    home_team_id: int
    away_team_id: int
    season_id: int

    home_team: Team
    away_team: Team
    season: Season

class League(LeagueBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the league")
    season_id: int
    team_id: int

    team: Team
    season: Season

class SeasonStats(SeasonStatsBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the season stats")
    season_id: int
    team_id: int

    team: Team
    season: Season

class LearningFeature(LearningFeatureBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the learning feature")
    season_id: int
    team_id: int

    team: Team
    season: Season

class PredictedMatch(PredictedMatchBase):
    id: int = Field(examples=[1, 2], description="Unique indetifier for the predicted match")
    home_team_id : int
    away_team_id: int

    home_team: Team
    away_team: Team

class MatchPredictionRequest(BaseModel):
    home_team_id: int
    away_team_id: int

class PredictedMatchCreate(BaseModel): 
    home_team_id: int
    away_team_id: int



class PredictedMatchOut(BaseModel):
    id: int
    home_team_id: int
    away_team_id: int
    is_predicted: bool

    home_win_probability_lr: Optional[float]
    draw_probability_lr: Optional[float]
    away_win_probability_lr: Optional[float]

    home_win_probability_rfc: Optional[float]
    draw_probability_rfc: Optional[float]
    away_win_probability_rfc: Optional[float]

    home_win_probability_xgb: Optional[float]
    draw_probability_xgb: Optional[float]
    away_win_probability_xgb: Optional[float]

    home_win_probability_svc: Optional[float]
    draw_probability_svc: Optional[float]
    away_win_probability_svc: Optional[float]


    home_expected_goals_lr: Optional[float]
    away_expected_goals_lr: Optional[float]
    home_expected_goals_rfr: Optional[float]
    away_expected_goals_rfr: Optional[float]
    home_expected_goals_xgb: Optional[float]
    away_expected_goals_xgb: Optional[float]
    home_expected_goals_svr: Optional[float]
    away_expected_goals_svr: Optional[float]

    model_config = {"from_attributes" : True }

class PredictedMatchPredictionResult(PredictedMatchCreate):
    is_predicted: bool

    home_win_probability_lr: float
    draw_probability_lr: float
    away_win_probability_lr: float
    home_win_probability_rfc: float
    draw_probability_rfc: float
    away_win_probability_rfc: float
    home_win_probability_xgb: float
    draw_probability_xgb: float
    away_win_probability_xgb: float
    home_win_probability_svc: float
    draw_probability_svc: float
    away_win_probability_svc: float

    home_expected_goals_lr: float
    away_expected_goals_lr: float
    home_expected_goals_rfr: float
    away_expected_goals_rfr: float
    home_expected_goals_xgb: float
    away_expected_goals_xgb: float
    home_expected_goals_svr: float
    away_expected_goals_svr: float

    model_config = {"from_attributes" : True }
