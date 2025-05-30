from typing import Any, Dict, Optional
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session
from models import League, LearningFeature, Match, PredictedMatch, Season, SeasonStats, Team
from schemas import PredictedMatchCreate, PredictedMatchPredictionResult

def get_season(db: Session, season_id: int):
    return db.query(Season).filter(Season.id == season_id).first()

def get_seasons(db: Session, skip: int = 0):
    return db.query(Season).offset(skip).all()

def get_team(db: Session, team_id: int):
    return db.query(Team).filter(Team.id == team_id).first()

def get_teams(db: Session, skip: int = 0):
    return db.query(Team).offset(skip).all()

def get_match(db: Session, match_id: int):
    return db.query(Match).filter(Match.id == match_id).first()

def get_matches(db: Session, skip: int = 0, limit: int = 50):
    return db.query(Match).offset(skip).limit(limit=limit).all()

def get_matches_by_season(db: Session, season_id:int, skip: int =0, limit: int = 50):
    return db.query(Match).filter(Match.season_id == season_id).offset(skip).limit(limit=limit).all()

def get_league_table(db: Session, season_id: int):
    return db.query(League).filter(League.season_id == season_id)\
        .order_by(League.position).all()

def get_season_stats(db: Session, season_id: int):
    return db.query(SeasonStats).filter(SeasonStats.season_id == season_id)\
        .order_by(SeasonStats.team_id).all()

def get_season_stats_for_team(db: Session, team_id: int):
    return db.query(SeasonStats).filter(SeasonStats.team_id == team_id)\
        .order_by(SeasonStats.season_id).all()

def get_learning_features(db: Session, season_id: int):
    return db.query(LearningFeature).filter(LearningFeature.season_id == season_id)\
        .order_by(LearningFeature.team_id).all()

def get_learning_features_for_team(db: Session, team_id: int):
    return db.query(LearningFeature).filter(LearningFeature.team_id == team_id)\
        .order_by(LearningFeature.season_id).all()

def get_average_learning_features_for_team(db: Session, team_id: int) -> Optional[Dict[str, Any]]:
    avg_query_result = db.query(
        func.avg(LearningFeature.goals_per_match).label("avg_goals_per_match"),
        func.avg(LearningFeature.offensive_strength).label("avg_offensive_strength"),
        func.avg(LearningFeature.goals_conceded_per_match).label("avg_goals_conceded_per_match"),
        func.avg(LearningFeature.defense_strength).label("avg_defense_strength"),
        func.avg(LearningFeature.effective_possession).label("avg_effective_possession"),
        func.avg(LearningFeature.passing_and_control_strength).label("avg_passing_and_control_strength"),
        func.avg(LearningFeature.penalties_converted_rate).label("avg_penalties_converted_rate"),
        func.avg(LearningFeature.set_piece_strength).label("avg_set_piece_strength"),
        func.avg(LearningFeature.team_power_index).label("avg_team_power_index"),
        func.avg(LearningFeature.avg_position).label("avg_avg_position"),
        func.avg(LearningFeature.avg_points).label("avg_avg_points")
    ).filter(LearningFeature.team_id == team_id).first()

    if avg_query_result and any(value is not None for value in avg_query_result):
        avg_features_dict = {
            "goals_per_match": avg_query_result.avg_goals_per_match,
            "offensive_strength": avg_query_result.avg_offensive_strength,
            "goals_conceded_per_match": avg_query_result.avg_goals_conceded_per_match,
            "defense_strength": avg_query_result.avg_defense_strength,
            "effective_possession": avg_query_result.avg_effective_possession,
            "passing_and_control_strength": avg_query_result.avg_passing_and_control_strength,
            "penalties_converted_rate": avg_query_result.avg_penalties_converted_rate,
            "set_piece_strength": avg_query_result.avg_set_piece_strength,
            "team_power_index": avg_query_result.avg_team_power_index,
            "avg_position": avg_query_result.avg_avg_position,
            "avg_points": avg_query_result.avg_avg_points,
        }
        return avg_features_dict
    return None

def get_predicted_match(db: Session, match_id: int):
    return db.query(PredictedMatch).filter(PredictedMatch.id == match_id).first()

def get_predicted_matches(db: Session, skip: int = 0):
    return db.query(PredictedMatch).offset(skip).all()

def get_recent_matches(db: Session, team_id: int, limit: int):
    return db.query(Match).filter(
        or_(
            Match.home_team_id == team_id,
            Match.away_team_id == team_id
        )
    ).order_by(Match.match_date.desc()).limit(limit).all()

def get_h2h_matches(db: Session, team1_id: int, team2_id: int):
    return db.query(Match).filter(
        or_(
            and_(Match.home_team_id == team1_id, Match.away_team_id == team2_id),
            and_(Match.home_team_id == team2_id, Match.away_team_id == team1_id)
        )
    ).all()  

def get_home_matches(db: Session, home_team_id: int):
    return db.query(Match).filter(Match.home_team_id == home_team_id).all()

def get_home_matches_at_season(db: Session, home_team_id: int, season_id: int):
    return db.query(Match).filter(Match.home_team_id == home_team_id, Match.season_id == season_id).all()

def get_average_home_xg_for_team(
    db: Session, 
    team_id: int, 
) -> Optional[float]:

    query = db.query(func.avg(Match.home_xG))\
              .filter(Match.home_team_id == team_id)
        
    average_xg = query.scalar()
    return average_xg

def get_average_away_xg_for_team(
    db: Session, 
    team_id: int, 
) -> Optional[float]:
    query = db.query(func.avg(Match.away_xG))\
              .filter(Match.away_team_id == team_id)

    average_xg = query.scalar()
    return average_xg

def create_empty_prediction_match(
    db: Session, 
    home_id: int, 
    away_id:int
) -> PredictedMatch:
    prediction = PredictedMatch(
        home_team_id = home_id,
        away_team_id = away_id,
        is_predicted = False
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction

def save_predicted_match_result(db: Session, prediction: PredictedMatchPredictionResult) -> PredictedMatch:
    db_predicted_match = PredictedMatch(
        home_team_id=prediction.home_team_id,
        away_team_id=prediction.away_team_id,
        is_predicted=prediction.is_predicted,

        home_win_probability_lr=prediction.home_win_lr,
        draw_probability_lr=prediction.draw_lr,
        away_win_probability_lr=prediction.away_win_lr,
        home_win_probability_rfc=prediction.home_win_rfc,
        draw_probability_rfc=prediction.draw_rfc,
        away_win_probability_rfc=prediction.away_win_rfc,
        home_win_probability_xgb=prediction.home_win_xgb,
        draw_probability_xgb=prediction.draw_xgb,
        away_win_probability_xgb=prediction.away_win_xgb,
        home_win_probability_svc=prediction.home_win_svc,
        draw_probability_svc=prediction.draw_svc,
        away_win_probability_svc=prediction.away_win_svc,

        home_expected_goals_lr=prediction.home_xg_lr,
        away_expected_goals_lr=prediction.away_xg_lr,
        home_expected_goals_rfr=prediction.home_xg_rfr,
        away_expected_goals_rfr=prediction.away_xg_rfr,
        home_expected_goals_xgb=prediction.home_xg_xgb,
        away_expected_goals_xgb=prediction.away_xg_xgb,
        home_expected_goals_svr=prediction.home_xg_svr,
        away_expected_goals_svr=prediction.away_xg_svr

    )
    db.add(db_predicted_match)
    db.commit()
    db.refresh(db_predicted_match)
    return db_predicted_match

def delete_predicted_match(db: Session, prediction_id: int):
    prediction = db.query(PredictedMatch).filter(PredictedMatch.id == prediction_id).first()
    if prediction:
        db.delete(prediction)
        db.commit()

