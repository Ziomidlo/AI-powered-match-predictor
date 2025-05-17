 #Database table inserting
from sqlite3 import IntegrityError
import pandas as pd
from sqlalchemy.orm import Session
from models import League, LearningFeature, Match, PredictedMatch, Season, SeasonStats, Team
from database import SessionLocal

db: Session = SessionLocal()

cleaned_data_folder = "cleaned_data/"
seasons = pd.read_csv(cleaned_data_folder + 'seasons.csv')
teams = pd.read_csv(cleaned_data_folder + 'teams.csv') 
matches = pd.read_csv(cleaned_data_folder + 'FullMergedMatchesInfo.csv')
team_stats = pd.read_csv(cleaned_data_folder + 'FullMergedSeasonStats.csv')
predicted_matches = pd.read_csv(cleaned_data_folder + 'PredictedUpcomingMatches.csv')


def data_to_db():
    with SessionLocal() as session:
        all_teams_in_db = session.query(Team).all()
        all_seasons_in_db = session.query(Season).all()
        all_matches_in_db = session.query(Match).all()
        all_leagues_in_db = session.query(League).all()
        all_season_stats_in_db = session.query(SeasonStats).all()
        all_learning_features_in_db = session.query(LearningFeature).all()
        all_predicted_matches_in_db = session.query(PredictedMatch).all()
        
        print(f"Numbers of team in base: {len(all_teams_in_db)}, seasons: {len(all_seasons_in_db)}, \
              matches: {len(all_matches_in_db)}, leagues: {len(all_leagues_in_db)}, \
              season stats: {len(all_season_stats_in_db)}, learning features: {len(all_learning_features_in_db)}, \
                predicted matches: {len(all_predicted_matches_in_db)}")
    season_data_to_db()
    team_data_to_db()
    match_data_to_db()
    league_data_to_db()
    season_stats_to_db()
    learning_feature_to_db()
    predicted_match_to_db()

def season_data_to_db():
    added_seasons = 0
    for _, row in seasons.iterrows():
        season_name = row['Season']
        existing_season = db.query(Season).filter(Season.season == season_name).first()
        if existing_season is None:
            new_season = Season(
            season = season_name)
            db.add(new_season)
            added_seasons += 1
        else:
            pass
    try:
        db.commit()
        print(f"Processing of season data has been finished. Added {added_seasons} new seasons")
    except IntegrityError as e:
        print(f"Data integrity error during commit: {e}. Withdraw the changes.")
        db.rollback()
    except Exception as e:
        print(f"Another error occurred during the commit: {e}. Withdraw the changes.")
        db.rollback() 

def team_data_to_db():
    added_teams = 0
    for _, row in teams.iterrows():
        team = row['Team']
        venue = row['Venue']
        existing_team = db.query(Team).filter(Team.team_name == team).first()
        if existing_team is None:
            new_team = Team(
            team_name = team,
            venue = venue)
            db.add(new_team)
            added_teams += 1
        else:
            pass
    try:
        db.commit()
        print(f"Processing of team data has been finished. Added {added_teams} new teams")
    except IntegrityError as e:
        print(f"Data integrity error during commit: {e}. Withdraw the changes.")
        db.rollback()
    except Exception as e:
        print(f"Another error occurred during the commit: {e}. Withdraw the changes.")
        db.rollback()

def match_data_to_db():
    added_matches = 0
    for _, row in matches.iterrows(): 
        season_name = row['Season']
        season = db.query(Season).filter(Season.season==season_name).first()

        home_team_id = row['Home Id']
        away_team_id = row['Away Id']
        match_date = row['Date']

        existing_matches = db.query(Match).filter(
            Match.match_date == match_date,
            Match.home_team_id == home_team_id,
            Match.away_team_id == away_team_id).first()
        if existing_matches is None:
            new_match = Match(
            season_id=season.id,
            match_date=row['Date'],
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=row['Home Goals'],
            away_goals=row['Away Goals'],
            home_xG=row.get('home_xG', None), 
            away_xG=row.get('away_xG', None)
        )
            db.add(new_match)
            added_matches += 1
        else:
            pass
    try:
        db.commit()
        print(f"Processing of match data has been finished. Added {added_matches} matches")
    except IntegrityError as e:
        print(f"Data integrity error during commit: {e}. Withdraw the changes.")
        db.rollback()
    except Exception as e:
        print(f"Another error occurred during the commit: {e}. Withdraw the changes.")
        db.rollback()
    
        
def league_data_to_db():
    added_leagues = 0
    for _, row in team_stats.iterrows():
        season_name = row['Season']
        team_id = row['Team Id']
        season = db.query(Season).filter(Season.season == season_name).first()
        existing_leagues = db.query(League).filter(
            League.season_id == season.id,
            League.team_id == team_id
        ).first()
        
        if existing_leagues is None:
            new_league = League(
                season_id = season.id,
                team_id = team_id,
                position = row['Position'],
                games_played = row['GP'],
                wins = row['W'],
                draws = row['D'],
                losses = row['L'],
                goals_for = row['GF'],
                goals_against = row['GA'],
                goals_difference = row['GD'],
                points = row['Pts']
            )
            db.add(new_league)
            added_leagues += 1
        else:
            pass
    try:
        db.commit()
        print(f"Processing of league data has been finished. Added {added_leagues} leagues")
    except IntegrityError as e:
        print(f"Data integrity error during commit: {e}. Withdraw the changes.")
        db.rollback()
    except Exception as e:
        print(f"Another error occurred during the commit: {e}. Withdraw the changes.")
        db.rollback()

def season_stats_to_db():
    added_season_stats = 0
    for _, row in team_stats.iterrows():
        season_name = row['Season']
        team_id = row['Team Id']
        season = db.query(Season).filter(Season.season == season_name).first()
        existing_stats = db.query(SeasonStats).filter(
            SeasonStats.season_id == season.id,
            SeasonStats.team_id == team_id
        ).first()

        if existing_stats is None:
            new_season_stats = SeasonStats(
                season_id = season.id,
                team_id = team_id,
                shots = row['Sh'],
                shots_on_target = row['SoT'],
                free_kicks = row['FK'],
                penalty_goals = row['PK'],
                passes_completed = row['Cmp'],
                passes_attempted = row['Att'],
                pass_completion = row['Cmp%'],
                corners = row['CK'],
                yellow_cards = row['CrdY'],
                red_cards = row['CrdR'],
                fouls_conceded = row['Fls'],
                penalites_conceded = row['PKcon'],
                own_goals = row['OG']
            )
            db.add(new_season_stats)
            added_season_stats += 1
        else:
            pass
    try:
        db.commit()
        print(f"Processing of season stats data has been finished. Added {added_season_stats} seasons stats")
    except IntegrityError as e:
        print(f"Data integrity error during commit: {e}. Withdraw the changes.")
        db.rollback()
    except Exception as e:
        print(f"Another error occurred during the commit: {e}. Withdraw the changes.")
        db.rollback()

def learning_feature_to_db():
    added_learning_feature = 0
    for _, row in team_stats.iterrows():
        season_name = row['Season']
        team_id = row['Team Id']
        season = db.query(Season).filter(Season.season == season_name).first()
        existing_features = db.query(LearningFeature).filter(
            LearningFeature.season_id == season.id,
            LearningFeature.team_id == team_id
        ).first()

        if existing_features is None:
            new_learning_feature = LearningFeature(
                season_id = season.id,
                team_id = team_id,
                goals_per_match = row['goals_per_match'],
                offensive_strength = row['offensive_strength'],
                goals_conceded_per_match = row['goals_conceded_per_match'],
                defense_strength = row['defense_strength'],
                effective_possession= row['effective_possesion'],
                passing_and_control_strength = row['passing_and_control_strength'],
                penalties_converted_rate= row['penalties_converted_rate'],
                set_piece_strength = row['set_piece_strength'],
                team_power_index = row['team_power_index'],
                avg_position = row['avg_position_ovr'],
                avg_points = row['avg_points_per_game']
            )
            db.add(new_learning_feature)
            added_learning_feature += 1
        else:
            pass
    try:
        db.commit()
        print(f"Processing of learning feature  data has been finished. Added {added_learning_feature} Learning Features")
    except IntegrityError as e:
        print(f"Data integrity error during commit: {e}. Withdraw the changes.")
        db.rollback()
    except Exception as e:
        print(f"Another error occurred during the commit: {e}. Withdraw the changes.")
        db.rollback()

def predicted_match_to_db():
    added_predicted_matches = 0
    for _, row in predicted_matches.iterrows():
        is_predicted = db.query(PredictedMatch.is_predicted == True).first()
        if is_predicted is None:
            new_predicted_match = PredictedMatch(
                home_team_id = row['Home Id'],
                away_team_id = row['Away Id'],
                is_predicted = True,
                home_win_probability_lr = row['home_win_lr'],
                draw_probability_lr = row['draw_lr'],
                away_win_probability_lr = row['away_win_lr'],
                home_win_probability_rfc = row['home_win_rfc'],
                draw_probability_rfc = row['draw_rfc'],
                away_win_probability_rfc = row['away_win_rfc'],
                home_expected_goals_lr = row['home_xG_lr'],
                away_expected_goals_lr = row['away_xG_lr'],
                home_expected_goals_rfr = row['home_xG_rfr'],
                away_expected_goals_rfr = row['away_xG_rfr']
            )
            db.add(new_predicted_match)
            added_predicted_matches += 1
        else:
            pass
    try:
        db.commit()
        print(f"Processing of predicted matches data has been finished. Added {added_predicted_matches} Predicted Matches")
    except IntegrityError as e:
        print(f"Data integrity error during commit: {e}. Withdraw the changes.")
        db.rollback()
    except Exception as e:
        print(f"Another error occurred during the commit: {e}. Withdraw the changes.")
        db.rollback()

data_to_db()