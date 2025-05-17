from datetime import datetime
import json
import os
from sqlite3 import IntegrityError
from dotenv import load_dotenv
import pandas as pd
import requests
from sqlalchemy.orm import Session
from models import Season, Team
from database import SessionLocal


def run_data_process():
   load_dotenv()
   API_KEY = os.getenv("API_KEY")

   uri = "https://api.football-data.org/v4/competitions/PL/"
   matches = "/matches"
   finished_matches =  matches + "?status=FINISHED"
   upcoming_matches = matches + "?status=SCHEDULED"
   standings = "standings"
   cleaned_data_folder = "cleaned_data/"
   json_data_folder = "json_data/"

   #Get Data From API
   headers = {'X-Auth-Token': API_KEY} 
   response = requests.get(uri + finished_matches , headers=headers)
   data_finished = response.json()


   with open(json_data_folder + 'dataPLMatches2024Finished.json', 'w') as file:
      json.dump(data_finished, file)

   with open(json_data_folder + 'dataPLMatches2024Finished.json', 'r') as file:
      data_finished = json.load(file)

   response_upcoming = requests.get(uri + upcoming_matches, headers=headers)
   data_upcoming = response_upcoming.json()

   with open(json_data_folder +'dataPLMatches2024Upcoming.json', 'w') as file:
      json.dump(data_upcoming, file)

   with open(json_data_folder + 'dataPLMatches2024Upcoming.json', 'r') as file:
      data_upcoming = json.load(file)
   
   response_team = requests.get(uri + standings, headers=headers)
   data_team = response_team.json()

   with open(json_data_folder + 'dataPLStandings.json', 'w') as file:
      json.dump(data_team, file)
   with open(json_data_folder + 'dataPLStandings.json', 'r') as file:
      data_team = json.load(file)

      #Data definition

   finished_matches = data_finished['matches']
   upcoming_matches = data_upcoming['matches']
   team_standings = data_team['standings'][0]['table']
   past_matches = pd.read_csv(cleaned_data_folder + 'past-matches.csv')
   past_seasons_stats = pd.read_csv(cleaned_data_folder + 'season-stats.csv')
   teams = pd.read_csv(cleaned_data_folder + 'teams.csv') 

   #Data Structuring & cleaning for missing values from API

   structured_current_season_matches = []
   structured_last_season_matches = []
   structured_scheduled_season_matches = []
   structured_team_standings = {}
   structured_past_seasons = {}


   team_mapping = dict(zip(teams['Team'], teams['Id']))
   team_venues = dict(zip(teams['Team'], teams['Venue']))

   for team in team_standings:
      team_name = team["team"]["name"]
      team_id = team_mapping.get(team_name, -1)
      structured_team_standings[team_name] = {
         "Season" : "2024/2025",
         "Team Id" : team_id,
         "Position": team["position"],
         "GP": team["playedGames"],
         "W": team["won"],
         "D": team["draw"],
         "L": team["lost"],
         "GF": team["goalsFor"],
         "GA": team["goalsAgainst"],
         "GD": team["goalDifference"],
         "Pts": team["points"]
      }

   for match in finished_matches:

      home_score = match["score"]["fullTime"].get("home", None)
      away_score = match["score"]["fullTime"].get("away", None)
      home_team = match["homeTeam"]["name"]
      away_team = match["awayTeam"]["name"]
      venue = team_venues.get(home_team, -1)
      home_team_id = team_mapping.get(home_team, -1)
      away_team_id = team_mapping.get(away_team, -1)
      match_date = datetime.strptime(match["utcDate"], '%Y-%m-%dT%H:%M:%SZ')
      goal_diff = None if home_score is None or away_score is None else (home_score - away_score)
      season = "2024/2025"
      
      if home_score is None or away_score is None:
         result = "Unknown"
      elif home_score > away_score:
         result = "Home win"
      elif home_score < away_score:
         result = "Away win"
      else:
         result = "Draw"

      structured_current_season_matches.append({
         "Season": season,
         "Date": match_date,
         "Home Id" : home_team_id,
         "Home": home_team,
         "Away Id" : away_team_id,
         "Away": away_team,
         "Home Goals": home_score,
         "Away Goals": away_score,
         "Result": result,
         "Venue" : venue,
         "Match": match["status"],
         "GD": goal_diff
      })

   for match in upcoming_matches:
      home_team = match['homeTeam']['name']
      away_team = match['awayTeam']['name']
      home_team_id = team_mapping.get(home_team, -1)
      away_team_id = team_mapping.get(away_team, -1)
      match_date = datetime.strptime(match["utcDate"], '%Y-%m-%dT%H:%M:%SZ') 
      venue = team_venues.get(home_team, -1)
      
      structured_scheduled_season_matches.append({
         "Season":"2024/2025",
         "Date": match_date,
         "Home Id": home_team_id,
         "Home": home_team,
         "Away Id": away_team_id,
         "Away": away_team,
         "Venue" : venue
      })

   #Creating Data Frames & Analysis

   missing_columns = ['Sh', 'SoT', 'FK', 'PK', 'Cmp', 'Att', 'Cmp%', 'CK', 'CrdY', 'CrdR', 'Fls', 'PKcon', 'OG']
   team_average_stats = past_seasons_stats.groupby("Team").mean(numeric_only=True).__round__(2)
   season_2023 = past_seasons_stats[past_seasons_stats['Season'] == "2023/2024"]
   bottom_five_teams = season_2023.sort_values("Position", ascending=False).head(5)
   bottom_five_avg_stats = bottom_five_teams[missing_columns].mean(numeric_only=True) 

   upcoming_matches = pd.DataFrame(structured_scheduled_season_matches)
   upcoming_matches['Date'] = upcoming_matches['Date'].astype(object)
   team_standings_df = pd.DataFrame.from_dict(structured_team_standings, orient="index").reset_index()
   team_standings_df = team_standings_df.rename(columns={'index': 'Team'})

   for col in missing_columns:
      team_standings_df[col] = team_standings_df.apply(
         lambda row: team_average_stats.loc[row['Team'], col] 
         if row['Team'] in team_average_stats.index else bottom_five_avg_stats[col],
         axis=1
      )
   #print(df.head())


   current_matches_df = pd.DataFrame(structured_current_season_matches)
   current_matches_df = current_matches_df.sort_values(by='Date', ascending=False)
   #print(df2.head())
   current_matches_df.info()
   current_matches_df.isnull().sum()
   current_matches_df.nunique()
   current_matches_df.describe()
   current_matches_df['xG'] = current_matches_df.groupby('Home')['Home Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
   current_matches_df['xG.1'] = current_matches_df.groupby('Away')['Away Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())


   print(past_matches.head())
   past_matches.info()
   past_matches.isnull().sum()
   past_matches.nunique()
   past_matches.describe()

   merged_season_stats = pd.concat([team_standings_df, past_seasons_stats], ignore_index=True)

   merged_season_stats = merged_season_stats.sort_values(by='Season', ascending=False)
   print(merged_season_stats.head())
   merged_season_stats.info()
   merged_season_stats.isnull().sum()
   merged_season_stats.nunique()
   merged_season_stats.describe()

   venue_mapping = past_matches[['Home', 'Venue']].drop_duplicates()
   print(venue_mapping)

   merged_matches = pd.concat([current_matches_df, past_matches], ignore_index=True)
   merged_matches['Match'] = merged_matches['Match'].fillna('FINISHED')
   merged_matches['GD'] = merged_matches['GD'].fillna(merged_matches['Home Goals'] - merged_matches['Away Goals'])
   merged_matches['result_numeric'] = merged_matches['Result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})

   merged_matches['Date'] = pd.to_datetime(merged_matches['Date']).dt.date
   merged_matches.info()
   merged_matches.isnull().sum()
   merged_matches.nunique()
   merged_matches.describe()


   team_standings_df[missing_columns] = team_standings_df[missing_columns].round(2)
   team_standings_df.info()
   team_standings_df.isnull().sum()
   team_standings_df.info()
   team_standings_df.describe()

   merged_season_stats['GP'] = merged_season_stats['GP'].fillna(38.0)
   merged_season_stats.info()
   merged_season_stats.isnull().sum()
   merged_season_stats.nunique()
   merged_season_stats.describe()

   upcoming_matches.info()
   upcoming_matches.isnull().sum()
   upcoming_matches.nunique()
   upcoming_matches.describe()

   #Feauture Engineering

   season_weights = {
      '2020/2021': 0.5,
      '2021/2022': 0.75,
      '2022/2023': 1.0,
      '2023/2024': 1.25,
      '2024/2025': 1.5
   }

   merged_season_stats['season_weight'] = merged_season_stats['Season'].map(season_weights)


   def calculate_form(team, match_date, df):
      past_matches = df[
      ((df['Home'] == team) | (df['Away'] == team))
      & (df['Date'] < match_date)
      ].sort_values(by='Date', ascending=False).head(5)

      points = 0
      total_points = 15
      for _, row in past_matches.iterrows():
         if row['Home'] == team:
            if row['Result'] == 'Home win':
               points += 3
            elif row['Result'] == 'Draw':
               points += 1
         elif row['Away'] == team:
            if row['Result'] == 'Away win':
               points += 3
            elif row['Result'] == 'Draw':
               points += 1
      percentage_of_points = (points / total_points) * 100
      return percentage_of_points.__round__(2)

   def calculate_goal_difference(team, match_date, df):
      past_matches = df[
         ((df['Home'] == team) | (df['Away'] == team)) & 
         (df['Date'] < match_date)
      ].sort_values(by='Date', ascending=False).head(5)

      home_matches = past_matches[past_matches['Home'] == team]
      away_matches = past_matches[past_matches['Away'] == team]

      goal_difference = (home_matches['Home Goals'] - home_matches['Away Goals']).sum() + \
                        (away_matches['Away Goals'] - away_matches['Home Goals']).sum()
      return goal_difference

   def head_to_head_results(home_team, away_team, df):
      h2h_matches = df[((df['Home'] == home_team) & (df['Away'] == away_team)) |
                  ((df['Home'] == away_team) & (df['Away'] == home_team))]
      
      home_wins = ((h2h_matches["Home"] == home_team) & (h2h_matches["result_numeric"] == 1)).sum() + \
               ((h2h_matches["Away"] == home_team) & (h2h_matches["result_numeric"] == -1)).sum()
      away_wins = ((h2h_matches["Away"] == away_team) & (h2h_matches["result_numeric"] == -1)).sum() + \
               ((h2h_matches["Home"] == away_team) & (h2h_matches["result_numeric"] == 1)).sum()

      draws = (h2h_matches['result_numeric'] == 0).sum()

      home_goals = h2h_matches.loc[h2h_matches["Home"] == home_team, "Home Goals"].sum() + \
                  h2h_matches.loc[h2h_matches["Away"] == home_team, "Away Goals"].sum()
      away_goals = h2h_matches.loc[h2h_matches["Home"] == away_team, "Home Goals"].sum() + \
                  h2h_matches.loc[h2h_matches["Away"] == away_team, "Away Goals"].sum()

      return home_wins, away_wins, draws, home_goals, away_goals

   def venue_impact(row, df):
      home_team = row['Home']
      venue = row['Venue']
      season = row['Season']

      home_venue_matches = df[
         (((df['Home'] == home_team)) &
         (df['Venue'] == venue) &
         (df['Season'] == season))
      ]

      other_venue_matches = df[
         (((df['Home'] == home_team)) &
         (df['Venue'] != venue) &
         (df['Season'] == season)) 
      ]

      home_wins = (home_venue_matches['result_numeric'] == 1).sum()
      away_wins = (home_venue_matches['result_numeric'] == -1).sum()
      draws = (home_venue_matches['result_numeric'] == 0).sum()

      home_wins_venues = (other_venue_matches['result_numeric'] == 1).sum()

      total_home = len(home_venue_matches)
      total_home_venues = len(other_venue_matches)

      home_win_percentage = (home_wins / total_home * 100) if total_home > 0 else 0
      away_win_percentage = (away_wins / total_home * 100) if total_home > 0 else 0
      draws_percentage = (draws/total_home * 100) if total_home > 0 else 0

      home_win_venues_percentage = (home_wins_venues / total_home_venues * 100) if total_home_venues > 0 else 0

      venue_impact_diff = home_win_percentage - home_win_venues_percentage

      home_goals_mean = home_venue_matches['Home Goals'].mean().__round__(2)
      away_goals_mean = home_venue_matches['Away Goals'].mean().__round__(2)

      return pd.Series([
         home_win_percentage.__round__(2),
         away_win_percentage.__round__(2),
         draws_percentage.__round__(2),
         home_goals_mean,
         away_goals_mean,
         venue_impact_diff.__round__(2),
      ])


   def offensive_strength_per_stats(row, df):
      team = row['Team']
      season = row['Season']

      offensive_stats_filter = df[(df['Team'] == team) & (df['Season'] == season)]

      stats = offensive_stats_filter.iloc[0]

      shots_on_target_per_goal = (stats["SoT"] / stats['GF']) if stats['GF'] != 0 else 0
      shots_per_goal = (stats["Sh"] / stats['GF'])  if stats['GF'] != 0 else 0
      sot_ratio = (stats["SoT"] / stats['Sh'])if stats['Sh'] != 0 else 0
      goals_per_match = (stats["GF"] / stats['GP']) if stats['GP'] != 0 else 0

      offensive_strength = (shots_on_target_per_goal  + shots_per_goal + sot_ratio + goals_per_match)

      return pd.Series([
         shots_on_target_per_goal,
         shots_per_goal,
         sot_ratio,
         goals_per_match,
         offensive_strength
      ])

   def defense_strength_per_stats(row, df):
      team = row['Team']
      season = row['Season']

      defense_stats_filter = df[(df['Team'] == team) & (df['Season'] == season)]

      stats = defense_stats_filter.iloc[0]

      goals_conceded_per_match = (stats['GA'] / stats['GP'])  if stats['GP'] != 0 else 0
      fouls_per_match = (stats['Fls'] / stats['GP'])  if stats['GP'] != 0 else 0
      cards_ratio = ((stats['CrdY'] + stats['CrdR']) / stats['GP'])  if stats['GP'] != 0 else 0
      own_goals_ratio = (stats['OG'] / stats['GP'])  if stats['GP'] != 0 else 0 

      defense_strength = 1 / (goals_conceded_per_match + fouls_per_match + cards_ratio + own_goals_ratio + 1e-5)

      return pd.Series([
         goals_conceded_per_match,
         fouls_per_match,
         cards_ratio,
         own_goals_ratio,
         defense_strength
      ])

   def passing_and_control_stats(row, df):
      team = row['Team']
      season = row['Season']
   
      stats_filter = df[(df['Team'] == team) & (df['Season'] == season)]

      stats = stats_filter.iloc[0]

      games_played = stats['GP'] 
      pass_attempted = stats['Att'] 
      pass_completed = stats['Cmp'] 
      pass_completion = stats['Cmp%'] 

      pass_accuracy = pass_completion
      passes_per_game = pass_attempted / games_played if games_played != 0 else 0
      effective_possesion = pass_completed / games_played if games_played !=0 else 0

      passing_and_control_strength = (pass_accuracy + effective_possesion) / passes_per_game
      
      return pd.Series([
         pass_accuracy,
         passes_per_game,
         effective_possesion,
         passing_and_control_strength
      ])

   def set_piece_strength(row, df):
      team = row['Team']
      season = row['Season']

      set_piece_filter = df[(df['Team'] == team) & (df['Season'] == season)]

      stats = set_piece_filter.iloc[0]
      
      corners_per_match = stats['CK'] / stats['GP'] if stats['GP'] != 0 else 0
      free_kicks_per_match = stats['FK'] / stats['GP'] if stats['GP'] != 0 else 0
      penalties_converted_rate = stats['PK'] / stats['PKcon'] if stats['PKcon'] != 0 else 0

      set_piece_strength = (penalties_converted_rate + free_kicks_per_match + corners_per_match)

      return pd.Series ([
         corners_per_match,
         free_kicks_per_match,
         penalties_converted_rate,
         set_piece_strength
      ])

   def team_power_index(row, df):
      team = row['Team']

      team_power_index_filter = df[df['Team'] == team]

      team_points = team_power_index_filter['Pts'].sum()
      games_played_by_team = team_power_index_filter['GP'].sum()
      season_counter = team_power_index_filter['Season'].count() 
      season_weight = team_power_index_filter['season_weight'].sum()
      attack_stats = team_power_index_filter['offensive_strength'].mean()
      defense_stats = team_power_index_filter['defense_strength'].mean()
      passing_and_position_stats = team_power_index_filter['passing_and_control_strength'].mean()
      set_piece_stats = team_power_index_filter['set_piece_strength'].mean()
      avg_position_overall = team_power_index_filter['Position'].mean()
      avg_points_per_game = team_points / games_played_by_team
      weight_points = (team_points / games_played_by_team) * season_weight
      
      tpi = (attack_stats + set_piece_stats + passing_and_position_stats - defense_stats - avg_position_overall + avg_points_per_game + weight_points) * season_weight / season_counter

      return pd.Series ([tpi,
                        avg_position_overall,
                        avg_points_per_game,
                        weight_points
                        ])

   def upcoming_matches_features(match_df, context_df, season_stats_df):
      match_df = match_df.copy()

      match_df['Date'] = pd.to_datetime(match_df['Date'])
      context_df['Date'] = pd.to_datetime(context_df['Date'])

      home_xg_avg = context_df.groupby('Home')['xG'].mean().to_dict()
      away_xg_avg = context_df.groupby('Away')['xG.1'].mean().to_dict()

      match_df['home_team_form'] = match_df.apply(lambda row: calculate_form(row['Home'], row['Date'], context_df), axis=1).__round__(2)
      match_df['away_team_form'] = match_df.apply(lambda row: calculate_form(row['Away'], row['Date'], context_df), axis=1).__round__(2)
      match_df["home_team_goal_difference"] = match_df.apply(lambda row: calculate_goal_difference(row["Home"], row["Date"], context_df), axis=1).__round__(2)
      match_df["away_team_goal_difference"] = match_df.apply(lambda row: calculate_goal_difference(row["Away"], row["Date"], context_df), axis=1).__round__(2)
      match_df['home_team_strength'] = match_df['home_team_form'] + match_df['home_team_goal_difference']
      match_df['away_team_strength'] = match_df['away_team_form'] + match_df['away_team_goal_difference']
      match_df['goal_diff_delta'] = (match_df['home_team_goal_difference'] - match_df['away_team_goal_difference']).__round__(2)
      match_df['home_xG_avg'] = match_df['Home'].map(home_xg_avg).round(2)
      match_df['away_xG_avg'] = match_df['Away'].map(away_xg_avg).round(2)
      match_df[['h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals']]  = \
      match_df.apply(lambda row: pd.Series(head_to_head_results(row['Home'], row['Away'], context_df)), axis=1)
      match_df[['venue_home_wins', 'venue_away_wins', 'venue_draws', 'venue_avg_home_goals', 'venue_avg_away_goals', 'venue_impact_diff']] = \
      match_df.apply(lambda row: pd.Series(venue_impact(row, context_df)), axis=1)

      homeStats = season_stats_df.add_prefix("home_").rename(columns={'home_Team': 'Home', 'home_Season': 'Season'})
      awayStats = season_stats_df.add_prefix("away_").rename(columns={'away_Team': 'Away', 'away_Season': 'Season'})

      match_df = match_df.merge(homeStats, on=['Season', 'Home'], how='left')
      match_df = match_df.merge(awayStats, on=['Season', 'Away'], how='left')

      return match_df
      

   
   #Data for model traning

   merged_matches['result_numeric'] = merged_matches['Result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
   merged_matches.to_csv(cleaned_data_folder + 'mergedMatches.csv')
   merged_matches["home_team_form"] = merged_matches.apply(lambda row: calculate_form(row["Home"], row["Date"], merged_matches), axis=1).__round__(2)
   merged_matches["away_team_form"] = merged_matches.apply(lambda row: calculate_form(row["Away"], row["Date"], merged_matches), axis=1).__round__(2)
   merged_matches["home_team_goal_difference"] = merged_matches.apply(lambda row: calculate_goal_difference(row["Home"], row["Date"], merged_matches), axis=1).__round__(2)
   merged_matches["away_team_goal_difference"] = merged_matches.apply(lambda row: calculate_goal_difference(row["Away"], row["Date"], merged_matches), axis=1).__round__(2)
   merged_matches["home_team_strength"] = merged_matches["home_team_form"] + merged_matches["home_team_goal_difference"].__round__(2)
   merged_matches["away_team_strength"] = merged_matches["away_team_form"] + merged_matches["away_team_goal_difference"].__round__(2)
   merged_matches["goal_diff_delta"] = merged_matches["home_team_goal_difference"] - merged_matches["away_team_goal_difference"].__round__(2)
   merged_matches["home_xG_avg"] = merged_matches.groupby('Home')['xG'].transform('mean').__round__(2)
   merged_matches["away_xG_avg"] = merged_matches.groupby('Away')['xG.1'].transform('mean').__round__(2)
   merged_matches[['h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals']] = \
      merged_matches.apply(lambda row: pd.Series(head_to_head_results(row['Home'], row['Away'], merged_matches)), axis=1)
   merged_matches[['venue_home_wins', 'venue_away_wins', 'venue_draws', 'venue_avg_home_goals', 'venue_avg_away_goals', 'venue_impact_diff']] = \
      merged_matches.apply(lambda row: pd.Series(venue_impact(row, merged_matches)), axis=1)


   merged_season_stats[['shots_on_target_per_goal', 'shots_per_goal', 'sot_ratio', 'goals_per_match', 'offensive_strength']] = \
      merged_season_stats.apply(lambda row: pd.Series(offensive_strength_per_stats(row, merged_season_stats)), axis=1)
   merged_season_stats[['goals_conceded_per_match', 'fouls_per_match', 'cards_ratio', 'own_goals_ratio', 'defense_strength']] = \
      merged_season_stats.apply(lambda row: pd.Series(defense_strength_per_stats(row, merged_season_stats)), axis=1)
   merged_season_stats[['pass_accuracy', 'passes_per_game', 'effective_possesion', 'passing_and_control_strength']] = \
      merged_season_stats.apply(lambda row: pd.Series(passing_and_control_stats(row, merged_season_stats)), axis=1)
   merged_season_stats[['corners_per_match', 'free_kicks_per_match', 'penalties_converted_rate', 'set_piece_strength']] = \
      merged_season_stats.apply(lambda row: pd.Series(set_piece_strength(row, merged_season_stats)), axis=1)
   merged_season_stats[["team_power_index", 'avg_position_ovr', 'avg_points_per_game', 'weight_points']] = merged_season_stats.apply(lambda row: pd.Series(team_power_index(row, merged_season_stats)), axis=1)

   merged_matches.to_csv(cleaned_data_folder + "FullMergedMatchesInfo.csv")
   merged_season_stats.to_csv(cleaned_data_folder + "FullMergedSeasonStats.csv")
   upcoming_matches = upcoming_matches_features(upcoming_matches, merged_matches, merged_season_stats)
   upcoming_matches.to_csv(cleaned_data_folder + "LearningUpcomingMatches.csv")

   home_season_stats = merged_season_stats.copy()
   home_season_stats = home_season_stats.add_prefix("home_")
   home_season_stats = home_season_stats.rename(columns={'home_Team' : 'Home'})
   home_season_stats = home_season_stats.rename(columns={'home_Season' : 'Season'})

   away_season_stats = merged_season_stats.copy()
   away_season_stats = away_season_stats.add_prefix("away_")
   away_season_stats = away_season_stats.rename(columns={'away_Team' : 'Away'})
   away_season_stats = away_season_stats.rename(columns={'away_Season' : 'Season'})

   merged_matches = pd.merge(merged_matches, home_season_stats, on=['Season', 'Home'], how='left')
   merged_matches = pd.merge(merged_matches, away_season_stats, on=['Season', 'Away'], how='left')

   merged_matches.to_csv(cleaned_data_folder + 'LearningMatchesData.csv')
   merged_season_stats.to_csv(cleaned_data_folder + 'LearningSeasonStatsData.csv')

run_data_process()


