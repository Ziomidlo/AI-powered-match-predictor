import requests
import pandas as pd
import json
import csv 
import seaborn as seaborn
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, make_scorer
from dotenv import load_dotenv
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import scipy.stats as stats

load_dotenv()
API_KEY = os.getenv("API_KEY")

uri = "https://api.football-data.org/v4/competitions/PL/"
matches = "/matches"
finishedMatches =  matches + "?status=FINISHED"
upcomingMaches = matches + "?status=TIMED"
standings = "standings"
cleanedDataFolder = "cleaned_data/"
visualizationFolder = "visualizations/"

#Get Data From API
headers = {'X-Auth-Token': API_KEY} 
response = requests.get(uri + finishedMatches , headers=headers)
dataFinished = response.json()


with open('dataPLMatches2024Finished.json', 'w') as file:
   json.dump(dataFinished, file)

with open('dataPLMatches2024Finished.json', 'r') as file:
   dataFinished = json.load(file)

responseUpcoming = requests.get(uri + upcomingMaches, headers=headers)
dataUpcoming = responseUpcoming.json()

with open('dataPLMatches2024Upcoming.json', 'w') as file:
   json.dump(dataUpcoming, file)

with open('dataPLMatches2024Upcoming.json', 'r') as file:
   dataUpcoming = json.load(file)
  
responseTeam = requests.get(uri + standings, headers=headers)
dataTeam = responseTeam.json()

with open('dataPLStandings.json', 'w') as file:
   json.dump(dataTeam, file)
with open('dataPLStandings.json', 'r') as file:
   dataTeam = json.load(file)


#Data definition

finishedMatches = dataFinished['matches']
upcomingMatches = dataUpcoming['matches']
teamStandings = dataTeam['standings'][0]['table']
seasons = pd.read_csv(cleanedDataFolder + 'seasons.csv')
pastMatches = pd.read_csv(cleanedDataFolder + 'past-matches.csv')
pastSeasonsStats = pd.read_csv(cleanedDataFolder + 'season-stats.csv')
teams = pd.read_csv(cleanedDataFolder + 'teams.csv') 

#Data Structuring & cleaning for missing values from API

structuredCurrentSeasonMatches = []
structuredLastSeasonsMatches = []
structuredTeamStandings = {}
structuredPastSeasons = {}


teamMapping = dict(zip(teams['Team'], teams['Id']))

for team in teamStandings:
   teamName = team["team"]["name"]
   teamId = teamMapping.get(teamName, -1)
   structuredTeamStandings[teamName] = {
      "Season" : "2024/2025",
      "Team Id" : teamId,
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


for match in finishedMatches:

   homeScore = match["score"]["fullTime"].get("home", None)
   awayScore = match["score"]["fullTime"].get("away", None)
   homeTeam = match["homeTeam"]["name"]
   awayTeam = match["awayTeam"]["name"]
   homeTeamId = teamMapping.get(homeTeam, -1)
   awayTeamId = teamMapping.get(awayTeam, -1)
   currentHomePossition = structuredTeamStandings[homeTeam]["Position"]
   currentAwayPossition = structuredTeamStandings[awayTeam]["Position"]
   matchDate = datetime.strptime(match["utcDate"], '%Y-%m-%dT%H:%M:%SZ')
   goalDiff = None if homeScore is None or awayScore is None else (homeScore - awayScore)
   season = "2024/2025"
   
   if homeScore is None or awayScore is None:
      result = "Unknown"
   elif homeScore > awayScore:
      result = "Home win"
   elif homeScore < awayScore:
      result = "Away win"
   else:
      result = "Draw"

   structuredCurrentSeasonMatches.append({
      "Season": season,
      "Date": matchDate,
      "Home Id" : homeTeamId,
      "Home": homeTeam,
      "Away Id" : awayTeamId,
      "Away": awayTeam,
      "Home Goals": homeScore,
      "Away Goals": awayScore,
      "Result": result,
      "Match": match["status"],
      "GD": goalDiff
   })

#Creating Data Frames & Analysis

teamStandingsDf = pd.DataFrame.from_dict(structuredTeamStandings, orient="index").reset_index()
teamStandingsDf = teamStandingsDf.rename(columns={'index': 'Team'})
#print(df.head())

currentMatchesDf = pd.DataFrame(structuredCurrentSeasonMatches)
currentMatchesDf = currentMatchesDf.sort_values(by='Date', ascending=False)
#print(df2.head())
currentMatchesDf.info()
currentMatchesDf.isnull().sum()
currentMatchesDf.nunique()
currentMatchesDf.describe()
currentMatchesDf['xG'] = currentMatchesDf.groupby('Home')['Home Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
currentMatchesDf['xG.1'] = currentMatchesDf.groupby('Away')['Away Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())


print(pastMatches.head())
pastMatches.info()
pastMatches.isnull().sum()
pastMatches.nunique()
pastMatches.describe()

pastSeasonsStats = pastSeasonsStats.sort_values(by='Season', ascending=False)
print(pastSeasonsStats.head())
pastSeasonsStats.info()
pastSeasonsStats.isnull().sum()
pastSeasonsStats.nunique()
pastSeasonsStats.describe()

homeStats = pastSeasonsStats.copy()
homeStats = homeStats.add_prefix("home_")
homeStats = homeStats.rename(columns={"home_Season": "Season"})
homeStats = homeStats.rename(columns={"home_Team" : "Home"})
homeStats.info()

awayStats = pastSeasonsStats.copy()
awayStats = awayStats.add_prefix("away_")
awayStats = awayStats.rename(columns={"away_Season" : "Season"})
awayStats = awayStats.rename(columns={"away_Team" : "Away"})
awayStats.info()


mergedMatches = pd.concat([currentMatchesDf, pastMatches], ignore_index=True)
venueMapping = pastMatches[['Home', 'Venue']].drop_duplicates()
team_venue_dict = venueMapping.set_index('Home')['Venue'].to_dict()
mergedMatches['Venue'] = mergedMatches['Venue'].fillna(mergedMatches['Home'].map(team_venue_dict))
missingVenues = {"Ipswich Town FC" : "Portman Road"}
mergedMatches['Venue'] = mergedMatches["Venue"].fillna(mergedMatches['Home'].map(missingVenues))
mergedMatches['Match'] = mergedMatches['Match'].fillna('FINISHED')
mergedMatches['GD'] = mergedMatches['GD'].fillna(mergedMatches['Home Goals'] - mergedMatches['Away Goals'])

mergedMatches['Date'] = pd.to_datetime(mergedMatches['Date']).dt.date
mergedMatches = pd.merge(mergedMatches, homeStats, on=['Season', 'Home'], how='left')
mergedMatches = pd.merge(mergedMatches, awayStats, on=['Season', 'Away'], how='left')
mergedMatches.info()
mergedMatches.isnull().sum()
mergedMatches.nunique()
mergedMatches.describe()

missingColumns = ['Sh', 'SoT', 'FK', 'PK', 'Cmp', 'Att', 'Cmp%', 'CK', 'CrdY', 'CrdR', 'Fls', 'PKcon', 'OG']
teamAverageStats = pastSeasonsStats.groupby("Team").mean(numeric_only=True).__round__(2)
season2023 = pastSeasonsStats[pastSeasonsStats['Season'] == "2023/2024"]
bottomFiveTeams = season2023.sort_values("Position", ascending=False).head(5)
bottomFiveAvgStats = bottomFiveTeams[missingColumns].mean(numeric_only=True) 

for col in missingColumns:
    teamStandingsDf[col] = teamStandingsDf.apply(
        lambda row: teamAverageStats.loc[row['Team'], col] 
        if row['Team'] in teamAverageStats.index else bottomFiveAvgStats[col],
        axis=1
    )
teamStandingsDf[missingColumns] = teamStandingsDf[missingColumns].round(2)
mergedSeasonStats = pd.concat([teamStandingsDf, pastSeasonsStats], ignore_index=True)
mergedSeasonStats['GP'] = mergedSeasonStats['GP'].fillna(38.0)
mergedSeasonStats.info()
mergedSeasonStats.isnull().sum()
mergedSeasonStats.nunique()
mergedSeasonStats.describe()

#Feauture Engineering

seasonWeights = {
    '2020/2021': 0.5,
    '2021/2022': 0.75,
    '2022/2023': 1.0,
    '2023/2024': 1.25,
    '2024/2025': 1.5
}

mergedSeasonStats['season_weight'] = mergedSeasonStats['Season'].map(seasonWeights)


def calculateForm(team, matchDate, df):
   pastMatches = df[
   ((df['Home'] == team) | (df['Away'] == team))
   & (df['Date'] < matchDate)
   ].sort_values(by='Date', ascending=False).head(5)

   points = 0
   totalPoints = 15
   for _, row in pastMatches.iterrows():
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
   percentageOfPoints = (points / totalPoints) * 100
   return percentageOfPoints.__round__(2)

def calculateGoalDifference(team, matchDate, df):
    pastMatches = df[
        ((df['Home'] == team) | (df['Away'] == team)) & 
        (df['Date'] < matchDate)
    ].sort_values(by='Date', ascending=False).head(5)

    homeMatches = pastMatches[pastMatches['Home'] == team]
    awayMatches = pastMatches[pastMatches['Away'] == team]

    goalDifference = (homeMatches['Home Goals'] - homeMatches['Away Goals']).sum() + \
                      (awayMatches['Away Goals'] - awayMatches['Home Goals']).sum()
    return goalDifference

def headToHeadResults(homeTeam, awayTeam, df):
   h2hMatches = df[((df['Home'] == homeTeam) & (df['Away'] == awayTeam)) |
               ((df['Home'] == awayTeam) & (df['Away'] == homeTeam))]
   
   homeWins = ((h2hMatches["Home"] == homeTeam) & (h2hMatches["result_numeric"] == 1)).sum() + \
            ((h2hMatches["Away"] == homeTeam) & (h2hMatches["result_numeric"] == -1)).sum()
   awayWins = ((h2hMatches["Away"] == awayTeam) & (h2hMatches["result_numeric"] == -1)).sum() + \
            ((h2hMatches["Home"] == awayTeam) & (h2hMatches["result_numeric"] == 1)).sum()

   draws = (h2hMatches['result_numeric'] == 0).sum()

   homeGoals = h2hMatches.loc[h2hMatches["Home"] == homeTeam, "Home Goals"].sum() + \
               h2hMatches.loc[h2hMatches["Away"] == homeTeam, "Away Goals"].sum()
   awayGoals = h2hMatches.loc[h2hMatches["Home"] == awayTeam, "Home Goals"].sum() + \
               h2hMatches.loc[h2hMatches["Away"] == awayTeam, "Away Goals"].sum()

   return homeWins, awayWins, draws, homeGoals, awayGoals

def venueImpact(row, df):
   homeTeam = row['Home']
   venue = row['Venue']
   season = row['Season']

   homeVenueMatches = df[
      (((df['Home'] == homeTeam)) &
      (df['Venue'] == venue) &
      (df['Season'] == season))
   ]

   otherVenueMatches = df[
      (((df['Home'] == homeTeam)) &
      (df['Venue'] != venue) &
      (df['Season'] == season)) 
   ]

   homeWins = (homeVenueMatches['result_numeric'] == 1).sum()
   awayWins = (homeVenueMatches['result_numeric'] == -1).sum()
   draws = (homeVenueMatches['result_numeric'] == 0).sum()

   homeWinsVenues = (otherVenueMatches['result_numeric'] == 1).sum()

   totalHome = len(homeVenueMatches)
   totalHomeVenues = len(otherVenueMatches)

   homeWinPercentage = (homeWins / totalHome * 100) if totalHome > 0 else 0
   awayWinPercentage = (awayWins / totalHome * 100) if totalHome > 0 else 0
   drawsPercentage = (draws/totalHome * 100) if totalHome > 0 else 0

   homeWinVenuesPercentage = (homeWinsVenues / totalHomeVenues * 100) if totalHomeVenues > 0 else 0

   venueImpactDiff = homeWinPercentage - homeWinVenuesPercentage

   homeGoalsMean = homeVenueMatches['Home Goals'].mean().__round__(2)
   awayGoalsMean = homeVenueMatches['Away Goals'].mean().__round__(2)

   return pd.Series([
      homeWinPercentage.__round__(2),
      awayWinPercentage.__round__(2),
      drawsPercentage.__round__(2),
      homeGoalsMean,
      awayGoalsMean,
      venueImpactDiff.__round__(2),
   ])


def offensiveStrengthPerStats(row, df):
   team = row['Team']
   season = row['Season']

   offensiveStatsFilter = df[(df['Team'] == team) & (df['Season'] == season)]

   stats = offensiveStatsFilter.iloc[0]

   shotsOnTargetPerGoal = (stats["SoT"] / stats['GF']) if stats['GF'] != 0 else 0
   shotsPerGoal = (stats["Sh"] / stats['GF'])  if stats['GF'] != 0 else 0
   sotRatio = (stats["SoT"] / stats['Sh'])if stats['Sh'] != 0 else 0
   goalsPerMatch = (stats["GF"] / stats['GP']) if stats['GP'] != 0 else 0

   offensiveStrength = ((shotsOnTargetPerGoal * 2) + (shotsPerGoal * 1.5) + (sotRatio * 1.2) + (goalsPerMatch * 2))

   return pd.Series([
      shotsOnTargetPerGoal,
      shotsPerGoal,
      sotRatio,
      goalsPerMatch,
      offensiveStrength
   ])

def defenseStrengthPerStats(row, df):
   team = row['Team']
   season = row['Season']

   defenseStatsFilter = df[(df['Team'] == team) & (df['Season'] == season)]

   stats = defenseStatsFilter.iloc[0]

   goalsConcededPerMatch = (stats['GA'] / stats['GP'])  if stats['GP'] != 0 else 0
   foulsPerMatch = (stats['Fls'] / stats['GP'])  if stats['GP'] != 0 else 0
   cardsRatio = ((stats['CrdY'] + stats['CrdR']) / stats['GP'])  if stats['GP'] != 0 else 0
   ownGoalsRatio = (stats['OG'] / stats['GP'])  if stats['GP'] != 0 else 0 

   defenseStrength = 1 / ((goalsConcededPerMatch * 1.5) + foulsPerMatch + cardsRatio + ownGoalsRatio + 1e-5)

   return pd.Series([
      goalsConcededPerMatch,
      foulsPerMatch,
      cardsRatio,
      ownGoalsRatio,
      defenseStrength
   ])

def passingAndControlStats(row, df):
   team = row['Team']
   season = row['Season']
 
   statsFilter = df[(df['Team'] == team) & (df['Season'] == season)]

   stats = statsFilter.iloc[0]

   gamesPlayed = stats['GP'] 
   passAttempted = stats['Att'] 
   passCompleted = stats['Cmp'] 
   passCompletion = stats['Cmp%'] 

   passAccuracy = passCompletion
   passesPerGame = passAttempted / gamesPlayed if gamesPlayed != 0 else 0
   effectivePossesion = passCompleted / gamesPlayed if gamesPlayed !=0 else 0

   passingAndControlStrength = (passAccuracy + effectivePossesion) / passesPerGame
   
   return pd.Series([
      passAccuracy,
      passesPerGame,
      effectivePossesion,
      passingAndControlStrength
   ])

def setPieceStrength(row, df):
   team = row['Team']
   season = row['Season']

   setPieceFilter = df[(df['Team'] == team) & (df['Season'] == season)]

   stats = setPieceFilter.iloc[0]
   
   cornersPerMatch = stats['CK'] / stats['GP'] if stats['GP'] != 0 else 0
   freeKicksPerMatch = stats['FK'] / stats['GP'] if stats['GP'] != 0 else 0
   penaltiesConvertedRate = stats['PK'] / stats['PKcon'] if stats['PKcon'] != 0 else 0

   setPieceStrength = ((penaltiesConvertedRate * 2) + (freeKicksPerMatch * 1.5) + cornersPerMatch * 1.3)

   return pd.Series ([
      cornersPerMatch,
      freeKicksPerMatch,
      penaltiesConvertedRate,
      setPieceStrength
   ])

def teamPowerIndex(row, df):
   team = row['Team']

   teamPowerIndexFilter = df[df['Team'] == team]

   teamPoints = teamPowerIndexFilter['Pts'].sum()
   gamesPlayedByTeam = teamPowerIndexFilter['GP'].sum()
   seasonCounter = teamPowerIndexFilter['Season'].count() 
   seasonWeight = teamPowerIndexFilter['season_weight'].sum()
   attackStats = teamPowerIndexFilter['offensive_strength'].mean()
   defenseStats = teamPowerIndexFilter['defense_strength'].mean()
   passingAndPositionStats = teamPowerIndexFilter['passing_and_control_strength'].mean()
   setPieceStats = teamPowerIndexFilter['set_piece_strength'].mean()
   avgPositionOverall = teamPowerIndexFilter['Position'].mean()
   avgPointsPerGame = teamPoints / gamesPlayedByTeam
   weightPoints = (teamPoints / gamesPlayedByTeam) * seasonWeight
   
   tpi = ((attackStats * 2) + (setPieceStats) + passingAndPositionStats - (defenseStats * 2) - (avgPositionOverall * 2) + (avgPointsPerGame * 2) + (weightPoints * 2)) * seasonWeight / seasonCounter

   return pd.Series ([tpi,
                     avgPositionOverall,
                     avgPointsPerGame,
                     weightPoints
                     ])


#Data for model traning

mergedMatches['result_numeric'] = mergedMatches['Result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
mergedMatches.to_csv(cleanedDataFolder + 'mergedMatches.csv')
mergedMatches["home_team_form"] = mergedMatches.apply(lambda row: calculateForm(row["Home"], row["Date"], mergedMatches), axis=1).__round__(2)
mergedMatches["away_team_form"] = mergedMatches.apply(lambda row: calculateForm(row["Away"], row["Date"], mergedMatches), axis=1).__round__(2)
mergedMatches["home_team_goal_difference"] = mergedMatches.apply(lambda row: calculateGoalDifference(row["Home"], row["Date"], mergedMatches), axis=1).__round__(2)
mergedMatches["away_team_goal_difference"] = mergedMatches.apply(lambda row: calculateGoalDifference(row["Away"], row["Date"], mergedMatches), axis=1).__round__(2)
mergedMatches["home_team_strength"] = mergedMatches["home_team_form"] + mergedMatches["home_team_goal_difference"].__round__(2)
mergedMatches["away_team_strength"] = mergedMatches["away_team_form"] + mergedMatches["away_team_goal_difference"].__round__(2)
mergedMatches["goal_diff_delta"] = mergedMatches["home_team_goal_difference"] - mergedMatches["away_team_goal_difference"].__round__(2)
mergedMatches["home_xG_avg"] = mergedMatches.groupby('Home')['xG'].transform('mean').__round__(2)
mergedMatches["away_xG_avg"] = mergedMatches.groupby('Away')['xG.1'].transform('mean').__round__(2)
mergedMatches[['h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals']] = \
   mergedMatches.apply(lambda row: pd.Series(headToHeadResults(row['Home'], row['Away'], mergedMatches)), axis=1)
mergedMatches[['venue_home_wins', 'venue_away_wins', 'venue_draws', 'venue_avg_home_goals', 'venue_avg_away_goals', 'venue_impact_diff']] = \
   mergedMatches.apply(lambda row: pd.Series(venueImpact(row, mergedMatches)), axis=1)

mergedSeasonStats[['shots_on_target_per_goal', 'shots_per_goal', 'sot_ratio', 'goals_per_match', 'offensive_strength']] = \
   mergedSeasonStats.apply(lambda row: pd.Series(offensiveStrengthPerStats(row, mergedSeasonStats)), axis=1)
mergedSeasonStats[['goals_conceded_per_match', 'fouls_per_match', 'cards_ratio', 'own_goals_ratio', 'defense_strength']] = \
   mergedSeasonStats.apply(lambda row: pd.Series(defenseStrengthPerStats(row, mergedSeasonStats)), axis=1)
mergedSeasonStats[['pass_accuracy', 'passes_per_game', 'effective_possesion', 'passing_and_control_strength']] = \
   mergedSeasonStats.apply(lambda row: pd.Series(passingAndControlStats(row, mergedSeasonStats)), axis=1)
mergedSeasonStats[['corners_per_match', 'free_kicks_per_match', 'penalties_converted_rate', 'set_piece_strength']] = \
   mergedSeasonStats.apply(lambda row: pd.Series(setPieceStrength(row, mergedSeasonStats)), axis=1)
mergedSeasonStats[["team_power_index", 'avg_position_ovr', 'avg_points_per_game', 'weight_points']] = mergedSeasonStats.apply(lambda row: pd.Series(teamPowerIndex(row, mergedSeasonStats)), axis=1)

homeSeasonStats = mergedSeasonStats.copy()
homeSeasonStats = homeSeasonStats.add_prefix("home_")
homeSeasonStats = homeSeasonStats.rename(columns={'home_Team' : 'Home'})
homeSeasonStats = homeSeasonStats.rename(columns={'home_Season' : 'Season'})

awaySeasonStats = mergedSeasonStats.copy()
awaySeasonStats = awaySeasonStats.add_prefix("away_")
awaySeasonStats = awaySeasonStats.rename(columns={'away_Team' : 'Away'})
awaySeasonStats = awaySeasonStats.rename(columns={'away_Season' : 'Season'})

mergedMatches = pd.merge(mergedMatches, homeSeasonStats, on=['Season', 'Home'], how='left')
mergedMatches = pd.merge(mergedMatches, awaySeasonStats, on=['Season', 'Away'], how='left')

mergedSeasonStats.to_csv(cleanedDataFolder + 'FullMergedSeasonStats.csv')
mergedMatches.to_csv(cleanedDataFolder + 'FullMergedMatchesInfo.csv')

#Correlation Analysis
corr, p_value = spearmanr(mergedMatches["h2h_home_wins"], mergedMatches["result_numeric"])
print(f"Spearsman's correlation: {corr:.3f}, p_value: {p_value:.3f}")
pearson_corr, p_value = pearsonr(mergedMatches["h2h_home_wins"], mergedMatches["result_numeric"])
print(f"Pearson's correlation: {pearson_corr:.3f}, p_value: {p_value:.3f}")

#Training models

X = mergedMatches[['home_xG_avg', 'away_xG_avg','goal_diff_delta', 'home_team_strength', 'away_team_strength', 
                   'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals', 
                   'venue_home_wins','venue_away_wins', 'venue_draws', 'venue_avg_home_goals', 'venue_avg_away_goals', 'venue_impact_diff',
                   'home_defense_strength', 'away_defense_strength', 'home_offensive_strength', 'away_offensive_strength', 'home_passing_and_control_strength', 
                   'away_passing_and_control_strength', 'home_team_power_index', 'away_team_power_index', 'home_avg_points_per_game', 'away_avg_points_per_game',
                   'home_avg_position_ovr', 'away_avg_position_ovr', 'home_weight_points', 'away_weight_points']]

y = mergedMatches["result_numeric"]

tscv_tuning = TimeSeriesSplit(n_splits= 5)

#Logistic Regression tuning
def logisticRegressionTuning():
   print("--- Tuning Logistic Regression ---")

   lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter= 5000))

   param_distributions_lr = {
      'logisticregression__C': stats.loguniform(0.001, 100),
      'logisticregression__penalty' : ['l2', 'l1'],
      'logisticregression__solver' : ['lbfgs', 'saga']
   }

   random_search_lr = RandomizedSearchCV(lr_pipeline, param_distributions_lr, n_iter=100, cv=tscv_tuning, scoring='accuracy', random_state=42,  n_jobs=-1)

   print("Starting Tuning Logistic Regression...")
   random_search_lr.fit(X, y)

   print("\n--- Results of Tuning  Logistic Regression ---")
   print("Best parameters:", random_search_lr.best_params_)
   print("Best average Result CV (Accuracy): ", random_search_lr.best_score_)

   best_lr_model = random_search_lr.best_estimator_
   print("Best Model:", best_lr_model)

#Logistic Regression 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model =  make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs', max_iter=5000, random_state=42, C=0.008))
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))


#Random Forest Classifier Tuning
def randomForestClassifierTuning():
   print("\n--- Tuning Random Forest Classifier ---")

   rf_clf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, class_weight='balanced'))

   param_distributions_rf_clf = {
      'randomforestclassifier__n_estimators' : stats.randint(100, 1000),
      'randomforestclassifier__max_depth' : [None] + list(stats.randint(5,30).rvs(10)),
      'randomforestclassifier__min_samples_split': stats.randint(2, 50),
      'randomforestclassifier__min_samples_leaf': stats.randint(1, 30),
      'randomforestclassifier__max_features': ['sqrt', 'log2', None], 
      'randomforestclassifier__criterion': ['gini', 'entropy'],
   }

   print("Startning tuning Random Forest Classifier...")
   random_search_rf_clf = RandomizedSearchCV(rf_clf_pipeline, param_distributions_rf_clf, n_iter=100, scoring="accuracy", random_state=42, n_jobs=-1)
   random_search_rf_clf.fit(X, y)

   print("\n--- Results of Tuning Random Forest Classifier ---")
   print("Best parameters:", random_search_rf_clf.best_params_)
   print("Best average Result CV (Accuracy):", random_search_rf_clf.best_score_)

   best_rf_clf_model = random_search_rf_clf.best_estimator_
   print("Best model:", best_rf_clf_model)

#Random Forest Classifier scores
rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', n_estimators=445, random_state=42, class_weight='balanced', max_depth=22, min_samples_leaf=4, min_samples_split=20))
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classifier - Accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#Linear Regression

#Target for linear reggresion
y_home_goals = mergedMatches["Home Goals"]
y_away_goals = mergedMatches["Away Goals"]

# Split for Home Goals
X_train_home, X_test_home, y_train_home, y_test_home = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)

# Split for Away Goals
X_train_away, X_test_away, y_train_away, y_test_away = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)


home_goals_model = make_pipeline(StandardScaler(), LinearRegression())
away_goals_model = make_pipeline(StandardScaler(), LinearRegression())
home_goals_model.fit(X_train_home, y_train_home)
away_goals_model.fit(X_train_away, y_train_away)

y_pred_home = home_goals_model.predict(X_test_home)
y_pred_away = away_goals_model.predict(X_test_away)

print("Linear Regression - Home Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_home, y_pred_home))
print("MSE:", mean_squared_error(y_test_home, y_pred_home))

print("\nLinear Regression - Away Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_away, y_pred_away))
print("MSE:", mean_squared_error(y_test_away, y_pred_away))


#Random Forest Regressor Tuning


def randomForestRegressorTuning():
   print("--- Tuning for Random Forest Regressor ---")
   rf_reg_pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42))

   param_distributions_rf_reg = {
      'randomforestregressor__n_estimators': stats.randint(100, 1000),
      'randomforestregressor__max_depth': [None] + list(stats.randint(5, 30).rvs(10)),
      'randomforestregressor__min_samples_split': stats.randint(2, 50),
      'randomforestregressor__min_samples_leaf': stats.randint(1, 30),
      'randomforestregressor__max_features': ['sqrt', 'log2', None],
      'randomforestregressor__criterion': ['squared_error', 'absolute_error'] 
   }

   mae_scorer = make_scorer(mean_absolute_error)

   print("Starting Tuning for Random Forest Regressor...")
   random_search_rf_reg = RandomizedSearchCV(rf_reg_pipeline, param_distributions_rf_reg, n_iter=100, cv=tscv_tuning, scoring=mae_scorer, random_state=42, n_jobs=-1 )

   print("Tuning for Home Goals...")
   random_search_rf_reg_home = random_search_rf_reg.fit(X, y_home_goals)

   print("\n--- Results of Tuning Random Forest Regressor (Home Goals) ---")
   print("Best parameters:", random_search_rf_reg_home.best_params_)
   print("Best average result CV (MAE):", random_search_rf_reg_home.best_score_)

   best_rf_reg_home_model = random_search_rf_reg_home.best_estimator_
   print("Best model:", best_rf_reg_home_model)

   print("Tuning for Away Goals...")
   random_search_rf_reg_away = random_search_rf_reg.fit(X, y_away_goals)

   print("\n--- Results of Tuning Random Forest Regressor (Away Goals) ---")
   print("Best parameters:", random_search_rf_reg_away.best_params_)
   print("Best average result CV (MAE):", random_search_rf_reg_away.best_score_)

   best_rf_reg_away_model = random_search_rf_reg_away.best_estimator_
   print("Best model:", best_rf_reg_away_model)


#Random Forest Regressor Scores

home_goals_rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=838, random_state=42, max_depth=23, max_features='log2', min_samples_leaf=27, min_samples_split=36))
away_goals_rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=956, random_state=42, max_depth=11, max_features='log2', min_samples_leaf=29, min_samples_split=45))

home_goals_rf.fit(X_train_home, y_train_home)
away_goals_rf.fit(X_train_away, y_train_away)

y_pred_home_rf = home_goals_rf.predict(X_test_home)
y_pred_away_rf = away_goals_rf.predict(X_test_away)

print("Random Forest Regressor - Home Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_home, y_pred_home_rf))
print("MSE:", mean_squared_error(y_test_home, y_pred_home_rf))

print("\nRandom Forest Regressor - Away Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_away, y_pred_away_rf))
print("MSE:", mean_squared_error(y_test_away, y_pred_away_rf))

probs = model.predict_proba(X_test)

importances = model.named_steps['logisticregression'].coef_[0]
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))


rf_model_step = rf_model.named_steps['randomforestclassifier']
importancesRF = rf_model_step.feature_importances_
feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance RF': importancesRF})
print(feature_importance_rf.sort_values(by='Importance RF', ascending=False))

logisticRegressionTuning()
randomForestRegressorTuning()
randomForestClassifierTuning()


#Visualization

h2hHomeWins = mergedMatches['h2h_home_wins'].sum()
h2hAwayWins = mergedMatches['h2h_away_wins'].sum()
h2hDraws = mergedMatches['h2h_draws'].sum()
h2hValues = [h2hHomeWins, h2hAwayWins, h2hDraws]
labels = ['Home Wins', 'Away Wins', 'Draws']
resultCounts = mergedMatches['result_numeric'].value_counts()

values = [resultCounts[1], resultCounts[0], resultCounts[-1]]

# Bar Chart
plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['blue', 'red', 'gray'])
plt.xlabel("Match Result")
plt.ylabel("Count")
plt.title("Distribution of Match Results")
plt.tight_layout()
plt.savefig(visualizationFolder + "Bar chart of Match Result Distribution")


plt.figure(figsize=(12, 6))
plt.hist(mergedMatches["goal_diff_delta"], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel("Goal Difference (Home - Away)")
plt.ylabel("Frequency")
plt.title("Distribution of Goal Difference Delta")
plt.tight_layout()
plt.savefig(visualizationFolder + "Histogram of Goal Difference")

plt.figure(figsize=(12, 6))
seaborn.boxplot(x=mergedMatches["h2h_home_wins"], y=mergedMatches['result_numeric'])
plt.title("H2H home wins vs. match result")
plt.xlabel("H2H home win matches")
plt.ylabel("Match Result")
plt.tight_layout()
plt.savefig(visualizationFolder + "Boxplot of H2H Home Wins and Match Result")

plt.figure(figsize=(12, 6))
seaborn.boxplot(x=mergedMatches["h2h_away_wins"], y=mergedMatches['result_numeric'])
plt.title("H2H aways wins vs. match result")
plt.xlabel("H2H away win matches")
plt.ylabel("Match Result")
plt.tight_layout()
plt.savefig(visualizationFolder + "Boxplot of H2H Away Wins and Match Result")

plt.figure(figsize=(12, 6))
seaborn.boxplot(x=mergedMatches["h2h_draws"], y=mergedMatches['result_numeric'])
plt.title("H2H draws win vs. match result")
plt.xlabel("H2H draw matches")
plt.ylabel("Match Result")
plt.tight_layout()
plt.savefig(visualizationFolder + "Boxplot of H2H draws and Match Result")

plt.figure(figsize=(12, 6))
seaborn.boxplot(x=mergedMatches["h2h_home_goals"], y=mergedMatches['result_numeric'])
plt.title("H2H home goals  vs. match result")
plt.xlabel("H2H home goals")
plt.ylabel("Match Result")
plt.tight_layout()
plt.savefig(visualizationFolder + "Boxplot of H2H Home Goals and Match Result")

plt.figure(figsize=(12, 6))
seaborn.boxplot(x=mergedMatches["h2h_away_goals"], y=mergedMatches['result_numeric'])
plt.title("H2H away goals vs. match result")
plt.xlabel("H2H away goals")
plt.ylabel("Match Result")
plt.tight_layout()
plt.savefig(visualizationFolder + "Boxplot of H2H Away Goals and Match Result")


plt.figure(figsize=(6,6))
plt.pie(h2hValues, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'red', 'gray'])
plt.title('Head-to-Head Results')
plt.tight_layout()
plt.savefig(visualizationFolder + "Pie chart of H2H Results")


plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="goals_per_match", hue="Season")
plt.title("Goals Per Match per Team by Season")
plt.ylabel("Goals Per Match")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Bar plot of Goals Scored per Match by Season")

plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="shots_on_target_per_goal", hue="Season")
plt.title("Shots On Target Per Goal Across Seasons")
plt.ylabel("Shots On Target Per goal")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Bar plot of Shots on Target per Goal Across Seasons")


plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="goals_conceded_per_match", hue="Season")
plt.title("Goals Conceded Per Match by Team")
plt.ylabel("Goals Conceded")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Bar plot of Goals Conceded per Match by Season")


plt.figure(figsize=(10, 8))
corr_features = mergedSeasonStats[[
    "shots_on_target_per_goal", "shots_per_goal", "sot_ratio", "goals_per_match",
    "goals_conceded_per_match", "fouls_per_match", "cards_ratio",
    "own_goals_ratio", 'offensive_strength', 'defense_strength', 'corners_per_match', 'free_kicks_per_match',
    'penalties_converted_rate','set_piece_strength', 'passing_and_control_strength', 'avg_position_ovr', 'avg_points_per_game', 'weight_points', 'team_power_index'
]]
sns.heatmap(corr_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Season stats")
plt.tight_layout()
plt.savefig(visualizationFolder + "Heatmap of correlation between Season stats")

plt.figure(figsize=(10, 8))
offensive_features = mergedSeasonStats[["shots_on_target_per_goal", "shots_per_goal", "sot_ratio", "goals_per_match", 'offensive_strength', 'team_power_index']]
sns.heatmap(offensive_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Offensive Metrics")
plt.tight_layout()
plt.savefig(visualizationFolder + "Heatmap of correlation between offensive metrics And Team Power Index")

plt.figure(figsize=(10, 8))
defensive_feature = mergedSeasonStats[["goals_conceded_per_match", "fouls_per_match", "cards_ratio", "own_goals_ratio", "defense_strength", 'team_power_index']]
sns.heatmap(defensive_feature.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Defensive Metrics")
plt.tight_layout()
plt.savefig(visualizationFolder + "Heatmap of correlation between Defensive metrics And Team Power Index")

plt.figure(figsize=(10, 8))
passing_control_feature = mergedSeasonStats[['pass_accuracy', 'passes_per_game', 'effective_possesion', 'passing_and_control_strength', 'team_power_index']]
sns.heatmap(passing_control_feature.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Passing & Control Metrics")
plt.tight_layout()
plt.savefig(visualizationFolder + "Heatmap of correlation between Passing & Control metrics And Team Power Index")

plt.figure(figsize=(10, 8))
set_piece_feature = mergedSeasonStats[['corners_per_match', 'free_kicks_per_match', 'penalties_converted_rate', 'set_piece_strength', 'team_power_index']]
sns.heatmap(set_piece_feature.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Set Piece Metrics")
plt.tight_layout()
plt.savefig(visualizationFolder + "Heatmap of correlation between Set Piece metrics And Team Power Index")

plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="weight_points")
plt.title("Weight points By Team")
plt.ylabel("Weight Points")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Barplot of Weight Points by Team")


plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="team_power_index")
plt.title("Team Power Index Barplot")
plt.ylabel("Team Power Index (TPI)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Barplot of Team Power Index")

plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="avg_points_per_game")
plt.title("Average Points Per Game By Team")
plt.ylabel("Average Points Per Game")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Barplot of Average Points Per Game by Team")

plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="avg_position_ovr")
plt.title("Average Position Overall By Team")
plt.ylabel("Average Position Overall")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Barplot of Average Position Overall by Team")

plt.figure(figsize=(13, 8))
sns.barplot(data=feature_importance_df.sort_values(by='Importance', ascending=False).head(20),
            x='Importance', y='Feature', palette='viridis')
plt.title("Top 20 Feature Importances (Logistic Regression)")
plt.savefig(visualizationFolder + "Barplot of top 20 importance features")


'''
plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="scaled_points")
plt.title("Scaled points By Team")
plt.ylabel("Scaled Points")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Barplot of Scaled Points by Team")

plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="normalized_position")
plt.title("Normalized Position By Team")
plt.ylabel("Normalized Position)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Barplot of Normalized postion by Team")

plt.figure(figsize=(12, 6))
sns.barplot(data=mergedSeasonStats, x="Team", y="weight_position")
plt.title("Weight Position By Team")
plt.ylabel("Weight Position)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(visualizationFolder + "Barplot of Weight postion by Team")
'''



