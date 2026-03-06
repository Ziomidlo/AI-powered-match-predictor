from datetime import datetime
import pandas as pd

class ApiAdapter:

   MISSING_KAGLE_COLUMNS = ['Sh', 'SoT', 'FK', 'PK', 'Cmp', 'Att', 'Cmp%', 'CK', 'CrdY', 'CrdR', 'Fls', 'PKcon', 'OG']

   NAME_MAPPING = {
        "AFC Bournemouth": "Bournemouth",
        "Arsenal FC": "Arsenal",
        "Aston Villa FC": "Aston Villa",
        "Brentford FC": "Brentford",
        "Brighton & Hove Albion FC": "Brighton",
        "Chelsea FC": "Chelsea",
        "Crystal Palace FC": "Crystal Palace",
        "Everton FC": "Everton",
        "Fulham FC": "Fulham",
        "Ipswich Town FC": "Ipswich Town",
        "Leicester City FC": "Leicester City",
        "Liverpool FC": "Liverpool",
        "Manchester City FC": "Manchester City",
        "Manchester United FC": "Manchester Utd",
        "Newcastle United FC": "Newcastle Utd",
        "Nottingham Forest FC": "Nott'ham Forest",
        "Southampton FC": "Southampton",
        "Tottenham Hotspur FC": "Tottenham",
        "West Ham United FC": "West Ham",
        "Wolverhampton Wanderers FC": "Wolves",
        "Burnley FC": "Burnley",
        "Leeds United FC": "Leeds United",
        "Sheffield United FC": "Sheffield Utd",
        "Sunderland AFC": "Sunderland",
        "Luton Town FC": "Luton Town",
        "Norwich City FC": "Norwich City",
        "West Bromwich Albion FC": "West Bromwich Albion"
    }

   def __init__(self, api_json_matches, api_json_upcoming_matches ,api_json_standings, teams_csv_path):
      self.raw_data = api_json_matches
      self.raw_team_data = api_json_standings
      self.raw_data_upcoming = api_json_upcoming_matches
      teams_df = pd.read_csv(teams_csv_path)
      teams_df['Team'] = teams_df['Team'].replace(self.NAME_MAPPING)
      teams_df.to_csv(teams_csv_path, index=False)
      self.team_mapping = dict(zip(teams_df["Team"], teams_df["Id"]))
      self.venue_mapping = dict(zip(teams_df["Team"], teams_df["Venue"]))
      self.current_season = self.get_current_season_string()


   def get_current_season_string(self):
      now = datetime.now()
      start_year = now.year if now.month >= 8 else now.year - 1
      return f"{start_year}/{start_year + 1}"
   


   def get_match_data(self):
      matches = []

      for match in self.raw_data:
         if match.get("status") not in ["FINISHED"]:
            continue
         home_team = match["homeTeam"]["name"]
         away_team = match["awayTeam"]["name"]
         score_ft = match.get("score", {}).get("fullTime", {})
         score_ht = match.get("score", {}).get("halfTime" or {})
         home_score = score_ft.get("home")
         away_score = score_ft.get("away")
         halftime_home = score_ht.get("home")
         halftime_away = score_ht.get("away")

         if home_team:
            home_team = self.NAME_MAPPING.get(home_team, home_team)
         if away_team:
            away_team = self.NAME_MAPPING.get(away_team, away_team)



         if home_score is None or away_score is None:
            result = "Unknown"
         elif home_score > away_score:
            result = "Home win"
         elif home_score < away_score:
            result = "Away win"
         else:
            result = "Draw"

         venue = self.venue_mapping.get(home_team)
         goal_diff = None if home_score is None or away_score is None else (home_score - away_score)

         matches.append({
            "Season" : self.current_season,
            "Date" : datetime.strptime(match["utcDate"], "%Y-%m-%dT%H:%M:%SZ"),
            "Home Id" : self.team_mapping.get(home_team, -1),
            "Home" : home_team,
            "Away Id" : self.team_mapping.get(away_team, -1),
            "Away" : away_team,
            "Home Goals HT" : halftime_home,
            "Away Goals HT" : halftime_away,
            "Home Goals" : home_score,
            "Away Goals" : away_score,
            "Result" : result,
            "Venue" : venue,
            "Status" : match["status"],
            "GD" : goal_diff
         })

      return pd.DataFrame(matches)
   
   def get_upcoming_match_data(self):
      matches = []
      for match in self.raw_data_upcoming:
         if match.get("status") not in ["SCHEDULED" , "TIMED"]:
            continue
         home_team = match["homeTeam"]["name"]
         away_team = match["awayTeam"]["name"]

         if home_team:
            home_team = self.NAME_MAPPING.get(home_team, home_team)
         if away_team:
            away_team = self.NAME_MAPPING.get(away_team, away_team)

         home_team_id = self.team_mapping.get(home_team, -1)
         away_team_id = self.team_mapping.get(away_team, -1)
         match_date = datetime.strptime(match["utcDate"], "%Y-%m-%dT%H:%M:%SZ")

         venue = self.venue_mapping.get(home_team, -1)

         matches.append({
            "Season" : self.current_season,
            "Date" : match_date,
            "Home Id" : home_team_id,
            "Home" : home_team,
            "Away Id" : away_team_id,
            "Away" : away_team,
            "Venue" : venue
         })

      return pd.DataFrame(matches).reset_index().rename(columns={"index" : "Team"})

   
   def get_standings(self):
      structured_team_standings = {}
      standings_data = self.raw_team_data

      if not standings_data:
         return pd.DataFrame()
      
      for team in standings_data:
         team_name = team["team"]["name"]
         if team_name:
            team_name = self.NAME_MAPPING.get(team_name, team_name)
         team_id = self.team_mapping.get(team_name, -1)
         team_stats = {
            "Season" : self.current_season,
            "Team Id" : team_id,
            "Position" : team["position"],
            "GP" : team["playedGames"],
            "W" : team["won"],
            "D" : team["draw"],
            "L" : team["lost"],
            "GF" : team["goalsFor"],
            "GA" : team["goalsAgainst"],
            "GD" : team["goalDifference"],
            "Pts" : team["points"]
         }

         for col in self.MISSING_KAGLE_COLUMNS:
            team_stats[col] = None
         
         structured_team_standings[team_name] = team_stats


      return pd.DataFrame.from_dict(structured_team_standings, orient="index").reset_index().rename(columns={'index': 'Team'})