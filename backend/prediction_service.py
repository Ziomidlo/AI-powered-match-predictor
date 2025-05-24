from typing import Tuple
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session
import pandas as pd
from . import crud
from . import schemas
from . import models
from data_learning import trained_ml_models, TRAINING_FEATURE_COLUMNS

def calculate_form_db(db: Session, team_id: int) -> float:
    past_matches_orms = crud.get_recent_matches(
        db, team_id=team_id, limit=5
    )

    points = 0
    possible_points = 0 

    if not past_matches_orms:
        return 0.0

    for match_orm in past_matches_orms:
        possible_points += 3
        if match_orm.home_team_id == team_id:
            if match_orm.result_numeric == 1: 
                points += 3
            elif match_orm.result_numeric == 0:
                points += 1
        elif match_orm.away_team_id == team_id:
            if match_orm.result_numeric == -1:
                points += 3
            elif match_orm.result_numeric == 0:
                points += 1
      
    percentage_of_points = (points / possible_points) * 100 if possible_points > 0 else 0
    return round(percentage_of_points, 2)

def calculate_goal_difference_db(db: Session, team_id: int) -> int:
    past_matches_orms = crud.get_recent_matches(
        db, team_id=team_id, limit=5
    )

    goal_difference = 0
    if not past_matches_orms:
        return 0

    for match_orm in past_matches_orms:
        if match_orm.home_team_id == team_id:
            goal_difference += (match_orm.home_goals - match_orm.away_goals)
        elif match_orm.away_team_id == team_id:
            goal_difference += (match_orm.away_goals - match_orm.home_goals)
            
    return goal_difference

def head_to_head_results_db(db: Session, home_team_id: int, away_team_id: int) -> Tuple[int, int, int, int, int]:
    h2h_matches_orms = crud.get_h2h_matches(db, team1_id=home_team_id, team2_id=away_team_id)

    home_team_h2h_wins = 0
    away_team_h2h_wins = 0
    h2h_draws = 0
    home_team_h2h_goals = 0
    away_team_h2h_goals = 0

    if not h2h_matches_orms:
        return 0, 0, 0, 0, 0

    for match_orm in h2h_matches_orms:
        if match_orm.home_team_id == home_team_id:
            home_team_h2h_goals += match_orm.home_goals
            away_team_h2h_goals += match_orm.away_goals
        elif match_orm.home_team_id == away_team_id:
            home_team_h2h_goals += match_orm.away_goals
            away_team_h2h_goals += match_orm.home_goals
        
        if match_orm.result_numeric == 0:
            h2h_draws += 1
        elif (match_orm.home_team_id == home_team_id and match_orm.result_numeric == 1) or \
             (match_orm.away_team_id == home_team_id and match_orm.result_numeric == -1):
            home_team_h2h_wins += 1
        elif (match_orm.home_team_id == away_team_id and match_orm.result_numeric == 1) or \
             (match_orm.away_team_id == away_team_id and match_orm.result_numeric == -1):
            away_team_h2h_wins +=1
            
    return home_team_h2h_wins, away_team_h2h_wins, h2h_draws, home_team_h2h_goals, away_team_h2h_goals

def calculate_venue_impact_db(
        db: Session,
        home_team_id: int,
) -> Tuple[float, float, float, float, float]:
    home_venue_matches_orms = crud.get_home_matches(
        db, home_team_id=home_team_id
    )

    home_wins_venue = 0
    away_wins_venue = 0
    draws_main_venue = 0
    home_goals_venue_sum = 0
    away_goals_venue_sum = 0

    for match in home_venue_matches_orms:
        if match.result_numeric == 1: home_wins_venue += 1
        elif match.result_numeric == -1: away_wins_venue +=1
        else: draws_main_venue +=1
        home_goals_venue_sum += match.home_goals
        away_goals_venue_sum += match.away_goals

    total_venue = len(home_venue_matches_orms)
    hw_perc = (home_wins_venue / total_venue * 100) if total_venue > 0 else 0
    aw_perc = (away_wins_venue / total_venue * 100) if total_venue > 0 else 0
    dr_perc = (draws_main_venue / total_venue * 100) if total_venue > 0 else 0
    hg_mean = (home_goals_venue_sum / total_venue) if total_venue > 0 else 0
    ag_mean = (away_goals_venue_sum / total_venue) if total_venue > 0 else 0

    return (
        round(hw_perc, 2),
        round(aw_perc, 2),
        round(dr_perc, 2),
        round(hg_mean, 2),
        round(ag_mean, 2)
    )

def calculate_avg_home_xg_db(
    db: Session, 
    team_id: int, 
) -> float:
    avg_xg = crud.get_average_home_xg_for_team(db, team_id=team_id)
    return round(avg_xg, 2) if avg_xg is not None else 0.0

def calculate_avg_away_xg_db(
    db: Session, 
    team_id: int, 
) -> float:
    avg_xg = crud.get_average_away_xg_for_team(db, team_id=team_id)
    return round(avg_xg, 2) if avg_xg is not None else 0.0


def generate_features_for_hypothetical_match(
        db: Session,
        home_team: models.Team,
        away_team: models.Team
) -> dict:
    features = {}

    home_lf_record = crud.get_average_learning_features_for_team(db, team_id=home_team.id)
    away_lf_record = crud.get_average_learning_features_for_team(db, team_id=away_team.id)

    features['home_xG_avg'] = calculate_avg_home_xg_db(db, team_id=home_team.id)
    features['away_xG_avg'] = calculate_avg_away_xg_db(db, team_id=away_team.id)
    features['home_form_percentage'] = calculate_form_db(db, home_team.id)
    features['away_form_percentage'] = calculate_form_db(db, away_team.id)
    features['home_goal_difference'] = calculate_goal_difference_db(db, home_team.id)
    features['away_goal_difference'] = calculate_goal_difference_db(db, away_team.id)
    features['goal_diff_delta'] = features.get('home_goal_difference', 0) - features.get('away_goal_difference', 0)
    features['home_team_strength'] = features.get('home_form_percentage') + features.get('home_goal_difference')
    features['away_team_strength'] = features.get('away_form_percentage') + features.get('away_goal_difference')

    h2h_hw, h2h_aw, h2h_dr, h2h_hg, h2h_ag = head_to_head_results_db(db, home_team_id=home_team.id, away_team_id=away_team.id)
    features['h2h_home_wins'] = h2h_hw
    features['h2h_away_wins'] = h2h_aw
    features['h2h_draws'] = h2h_dr
    features['h2h_home_goals'] = h2h_hg
    features['h2h_away_goals'] = h2h_ag

    venue_stats = calculate_venue_impact_db(db, home_team.id)
    features['venue_home_wins'] = venue_stats[0]
    features['venue_away_wins'] = venue_stats[1]
    features['venue_draw_perc'] = venue_stats[2]
    features['venue_avg_home_goals'] = venue_stats[3]
    features['venue_avg_away_goals'] = venue_stats[4]


    default_lf_value = 0.0 

    features['home_goals_per_match'] = round(home_lf_record.get("goals_per_match", default_lf_value), 2) if home_lf_record else default_lf_value
    features['home_offensive_strength'] = round(home_lf_record.get("offensive_strength", default_lf_value), 2) if home_lf_record else default_lf_value
    features['home_goals_conceded_per_match'] = round(home_lf_record.get("goals_conceded_per_match", default_lf_value),2) if home_lf_record else default_lf_value
    features['home_defense_strength'] = round(home_lf_record.get("defense_strength", default_lf_value),2) if home_lf_record else default_lf_value
    features['home_effective_possession'] = round(home_lf_record.get("effective_possession", default_lf_value), 2) if home_lf_record else default_lf_value
    features['home_passing_and_control_strength'] = round(home_lf_record.get("passing_and_control_strength", default_lf_value), 2) if home_lf_record else default_lf_value
    features['home_penalties_converted_rate'] = round(home_lf_record.get('penalties_converted_rate', default_lf_value), 2) if home_lf_record else default_lf_value
    features['home_set_piece_strength'] = round(home_lf_record.get("set_piece_strength", default_lf_value), 2) if home_lf_record else default_lf_value
    features['home_team_power_index'] = round(home_lf_record.get("team_power_index", default_lf_value), 2) if home_lf_record else default_lf_value
    features['home_avg_position_ovr'] = round(home_lf_record.get("avg_position", 20.0), 2) if home_lf_record else 20.0
    features['home_avg_points_per_game'] = round(home_lf_record.get('avg_points', default_lf_value), 2) if home_lf_record else default_lf_value

    features['away_goals_per_match'] = round(away_lf_record.get("goals_per_match", default_lf_value), 2) if away_lf_record else default_lf_value
    features['away_offensive_strength'] = round(away_lf_record.get("offensive_strength", default_lf_value), 2) if away_lf_record else default_lf_value
    features['away_goals_conceded_per_match'] = round(away_lf_record.get("goals_conceded_per_match", default_lf_value),2) if away_lf_record else default_lf_value
    features['away_defense_strength'] = round(away_lf_record.get("defense_strength", default_lf_value),2) if away_lf_record else default_lf_value
    features['away_effective_possession'] = round(away_lf_record.get("effective_possession", default_lf_value), 2) if away_lf_record else default_lf_value
    features['away_passing_and_control_strength'] = round(away_lf_record.get("passing_and_control_strength", default_lf_value), 2) if away_lf_record else default_lf_value
    features['away_penalties_converted_rate'] = round(away_lf_record.get('penalties_converted_rate', default_lf_value), 2) if away_lf_record else default_lf_value
    features['away_set_piece_strength'] = round(away_lf_record.get("set_piece_strength", default_lf_value), 2) if away_lf_record else default_lf_value
    features['away_team_power_index'] = round(away_lf_record.get("team_power_index", default_lf_value), 2) if away_lf_record else default_lf_value
    features['away_avg_position_ovr'] = round(away_lf_record.get("avg_position", 20.0), 2) if away_lf_record else 20.0
    features['away_avg_points_per_game'] = round(away_lf_record.get('avg_points', default_lf_value), 2) if away_lf_record else default_lf_value

    return features

def generate_and_store_match_prediction(
    db: Session,
    home_team_id: int,
    away_team_id: int
) -> models.PredictedMatch:

    home_team_orm = crud.get_team(db, team_id=home_team_id)
    away_team_orm = crud.get_team(db, team_id=away_team_id)

    if not home_team_orm:
        raise ValueError(f"Home team with ID {home_team_id} has been not found.")
    if not away_team_orm:
        raise ValueError(f"Away team with ID {away_team_id} has been not found.")

    feature_dict = generate_features_for_hypothetical_match(db, home_team_orm, away_team_orm)

    if not TRAINING_FEATURE_COLUMNS:
        raise ValueError("List TRAINING_FEATURE_COLUMNS is empty. Check configuration in data_learning.py.")
    if not feature_dict:
        raise ValueError("Not possible to generate a match feature.")

    try:
        data_for_df = {}
        for col in TRAINING_FEATURE_COLUMNS:
            if col not in feature_dict:
                print(f"WARNING: Missing feature '{col}' in feature_dict. Using default value 0.")
                data_for_df[col] = 0.0
            else:
                data_for_df[col] = feature_dict[col]
        
        X_to_predict_df = pd.DataFrame([data_for_df], columns=TRAINING_FEATURE_COLUMNS)
    except Exception as e:
        print(f"Critical error during creating DataFrame to prediction : {e}")
        raise ValueError(f"Couldn't set entering data for models: {e}")


    if not trained_ml_models:
        print("Critical Error: Dict `trained_ml_models` is empty.")
        raise RuntimeError("Models ML are not loaded. Check a starting process of application.")
    try:

    
        sample_classifier_key = 'support_vector_classifier' 
        if sample_classifier_key not in trained_ml_models:
             raise RuntimeError(f"Lack of crucial model '{sample_classifier_key}' to set an order of classes.")

        model_classes = trained_ml_models[sample_classifier_key].named_steps['model'].classes_ if isinstance(trained_ml_models[sample_classifier_key], Pipeline) else trained_ml_models[sample_classifier_key].classes_
        
        class_to_idx = {cls_val: idx for idx, cls_val in enumerate(model_classes)}
        idx_away_win = class_to_idx.get(-1) 
        idx_draw = class_to_idx.get(0)
        idx_home_win = class_to_idx.get(1)

        if any(idx is None for idx in [idx_away_win, idx_draw, idx_home_win]):
            raise RuntimeError(f"Couldn't map results of classes (-1,0,1) with `model.classes_`: {model_classes}")

        probs_lr_raw = trained_ml_models['logistic_regression_classifier'].predict_proba(X_to_predict_df)[0]
        probs_rfc_raw = trained_ml_models['random_forest_classifier'].predict_proba(X_to_predict_df)[0]
        probs_xgb_raw = trained_ml_models['xgboost_classifier'].predict_proba(X_to_predict_df)[0]
        probs_svc_raw = trained_ml_models['support_vector_classifier'].predict_proba(X_to_predict_df)[0]
        
        home_xg_lr_raw = trained_ml_models['linear_regressor_home'].predict(X_to_predict_df)[0]
        away_xg_lr_raw = trained_ml_models['linear_regressor_away'].predict(X_to_predict_df)[0]
        home_xg_rfr_raw = trained_ml_models['random_forest_regressor_home'].predict(X_to_predict_df)[0]
        away_xg_rfr_raw = trained_ml_models['random_forest_regressor_away'].predict(X_to_predict_df)[0]
        home_xg_xgb_raw = trained_ml_models['xgboost_regressor_home'].predict(X_to_predict_df)[0]
        away_xg_xgb_raw = trained_ml_models['xgboost_regressor_away'].predict(X_to_predict_df)[0]
        home_xg_svr_raw = trained_ml_models['svr_regressor_home'].predict(X_to_predict_df)[0]
        away_xg_svr_raw = trained_ml_models['svr_regressor_away'].predict(X_to_predict_df)[0]

    except KeyError as e:
        print(f"Critical Error: Missing model w `trained_ml_models`: {e}")
        raise RuntimeError(f"Missing model '{e}'. Make sure, every models are prepared well.")
    except Exception as e:
        print(f"Critical Error during model prediction: {e}")
        raise RuntimeError(f"Critical Error during model prediction: {e}")

    prediction_create_data = schemas.PredictedMatchCreate(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        is_predicted=True,

        home_win_lr=round(probs_lr_raw[idx_home_win] * 100, 2),
        draw_lr=round(probs_lr_raw[idx_draw] * 100, 2),
        away_win_lr=round(probs_lr_raw[idx_away_win] * 100, 2),
        
        home_win_rfc=round(probs_rfc_raw[idx_home_win] * 100, 2),
        draw_rfc=round(probs_rfc_raw[idx_draw] * 100, 2),
        away_win_rfc=round(probs_rfc_raw[idx_away_win] * 100, 2),

        home_win_xgb=round(probs_xgb_raw[idx_home_win] * 100, 2),
        draw_xgb=round(probs_xgb_raw[idx_draw] * 100, 2),
        away_win_xgb=round(probs_xgb_raw[idx_away_win] * 100, 2),

        home_win_svc=round(probs_svc_raw[idx_home_win] * 100, 2),
        draw_svc=round(probs_svc_raw[idx_draw] * 100, 2),
        away_win_svc=round(probs_svc_raw[idx_away_win] * 100, 2),

        home_xg_lr=round(float(home_xg_lr_raw), 2),
        away_xg_lr=round(float(away_xg_lr_raw), 2),
        home_xg_rfr=round(float(home_xg_rfr_raw), 2),
        away_xg_rfr=round(float(away_xg_rfr_raw), 2),
        home_xg_xgb=round(float(home_xg_xgb_raw), 2),
        away_xg_xgb=round(float(away_xg_xgb_raw), 2),
        home_xg_svr=round(float(home_xg_svr_raw), 2),
        away_xg_svr=round(float(away_xg_svr_raw), 2)
    )

    db_prediction = crud.create_predicted_match(db, prediction=prediction_create_data)
    return db_prediction