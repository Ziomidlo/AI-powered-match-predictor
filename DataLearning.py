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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from dotenv import load_dotenv

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


#Variables

finishedMatches = dataFinished['matches']
upcomingMatches = dataUpcoming['matches']
teamStandings = dataTeam['standings'][0]['table']
seasons = pd.read_csv(cleanedDataFolder + 'seasons.csv')
pastMatches = pd.read_csv(cleanedDataFolder + 'past-matches.csv')
pastSeasonsStats = pd.read_csv(cleanedDataFolder + 'season-stats.csv')
teams = pd.read_csv(cleanedDataFolder + 'teams.csv') 

#Data Structuring

structuredCurrentSeasonMatches = []
structuredLastSeasonsMatches = []
structuredTeamStandings = {}
structuredPastSeasons = {}


teamMapping = dict(zip(teams['Team'], teams['Id']))

for team in teamStandings:
   teamName = team["team"]["name"]
   teamId = teamMapping.get(teamName, -1)
   structuredTeamStandings[teamName] = {
      "Team Id" : teamId,
      "Position": team["position"],
      "GP": team["playedGames"],
      "W": team["won"],
      "D": team["draw"],
      "L": team["lost"],
      "GF": team["goalsFor"],
      "GA": team["goalsAgainst"],
      "GD": team["goalDifference"],
      "Points": team["points"]
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
#print(df.head())
teamStandingsDf.info()
teamStandingsDf.isnull().sum()
teamStandingsDf.nunique()
teamStandingsDf.describe()

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

print(pastSeasonsStats.head())
pastSeasonsStats.info()
pastSeasonsStats.isnull().sum()
pastSeasonsStats.nunique()
pastSeasonsStats.describe()

mergedMatches = pd.concat([currentMatchesDf, pastMatches], ignore_index=True)
venueMapping = pastMatches[['Home', 'Venue']].drop_duplicates()
team_venue_dict = venueMapping.set_index('Home')['Venue'].to_dict()
mergedMatches['Venue'] = mergedMatches['Venue'].fillna(mergedMatches['Home'].map(team_venue_dict))
missingVenues = {"Ipswich Town FC" : "Portman Road"}
mergedMatches['Venue'] = mergedMatches["Venue"].fillna(mergedMatches['Home'].map(missingVenues))
mergedMatches['Match'] = mergedMatches['Match'].fillna('FINISHED')
mergedMatches['GD'] = mergedMatches['GD'].fillna(mergedMatches['Home Goals'] - mergedMatches['Away Goals'])

mergedMatches['Date'] = pd.to_datetime(mergedMatches['Date']).dt.date
mergedMatches.info()
mergedMatches.isnull().sum()
mergedMatches.nunique()
mergedMatches.describe()

#Feauture Engineering
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

   return pd.Series([homeWinPercentage.__round__(2), awayWinPercentage.__round__(2), drawsPercentage.__round__(2), homeGoalsMean, awayGoalsMean, venueImpactDiff.__round__(2)])



#Data for model traning
'''
currentMatchesDf['result_numeric'] = currentMatchesDf['Result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
currentMatchesDf["home_team_form"] = currentMatchesDf.apply(lambda row: calculate_form(row["Home"], row["Date"], currentMatchesDf), axis=1)
currentMatchesDf["away_team_form"] = currentMatchesDf.apply(lambda row: calculate_form(row["Away"], row["Date"], currentMatchesDf), axis=1)
currentMatchesDf["home_team_goal_difference"] = currentMatchesDf.apply(lambda row: calculate_goal_difference(row["Home"], row["Date"], currentMatchesDf), axis=1)
currentMatchesDf["away_team_goal_difference"] = currentMatchesDf.apply(lambda row: calculate_goal_difference(row["Away"], row["Date"], currentMatchesDf), axis=1)
currentMatchesDf["home_team_strength"] = currentMatchesDf["home_team_form"] + currentMatchesDf["home_team_goal_difference"]
currentMatchesDf["away_team_strength"] = currentMatchesDf["away_team_form"] + currentMatchesDf["away_team_goal_difference"]
currentMatchesDf["goal_diff_delta"] = currentMatchesDf["home_team_goal_difference"] - currentMatchesDf["away_team_goal_difference"]
pastMatches["home_xG_avg"] = pastMatches.groupby('Home')['xG'].transform('mean')
pastMatches["away_xG_avg"] = pastMatches.groupby('Away')['xG.1'].transform('mean')
'''

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
mergedMatches[['venue_home_wins', 'venue_away_wins', 'venue_draws', 'venue_home_goals', 'venue_away_goals', 'venue_impact_diff']] = \
   mergedMatches.apply(lambda row: pd.Series(venueImpact(row, mergedMatches)), axis=1)

mergedMatches.to_csv(cleanedDataFolder + 'FullMergedMatchesInfo.csv')

#Correlation Analysis
corr, p_value = spearmanr(mergedMatches["h2h_home_wins"], mergedMatches["result_numeric"])
print(f"Spearsman's correlation: {corr:.3f}, p_value: {p_value:.3f}")
pearson_corr, p_value = pearsonr(mergedMatches["h2h_home_wins"], mergedMatches["result_numeric"])
print(f"Pearson's correlation: {pearson_corr:.3f}, p_value: {p_value:.3f}")

#Model Training Logistic and Linear Regression
#Logistic Regression

X = mergedMatches[[ "home_team_strength", "away_team_strength", 'goal_diff_delta', 'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 'h2h_away_goals', 'venue_home_wins', 'venue_away_wins', 'venue_draws', 'venue_home_goals', 'venue_away_goals', 'venue_impact_diff']]
y = mergedMatches["result_numeric"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model =  LogisticRegression(class_weight='balanced',solver='saga', max_iter=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

#Linear Regression

#Target for linear reggresion
y_home_goals = mergedMatches["Home Goals"]
y_away_goals = mergedMatches["Away Goals"]

X_train, X_test, y_train_home, y_test_home = train_test_split(X, y_home_goals, test_size=0.2, random_state=42)
X_train, X_test, y_train_away, y_test_away = train_test_split(X, y_away_goals, test_size=0.2, random_state=42)


home_goals_model = LinearRegression()
away_goals_model = LinearRegression()
home_goals_model.fit(X_train, y_train_home)
away_goals_model.fit(X_train, y_train_away)

y_pred_home = home_goals_model.predict(X_test)
y_pred_away = away_goals_model.predict(X_test)

print("Home Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_home, y_pred_home))
print("MSE:", mean_squared_error(y_test_home, y_pred_home))

print("\nAway Goals Prediction:")
print("MAE:", mean_absolute_error(y_test_away, y_pred_away))
print("MSE:", mean_squared_error(y_test_away, y_pred_away))


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
plt.savefig(visualizationFolder + "Bar chart of Match Result Distribution")


plt.figure(figsize=(8,5))
plt.hist(mergedMatches["goal_diff_delta"], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.xlabel("Goal Difference (Home - Away)")
plt.ylabel("Frequency")
plt.title("Distribution of Goal Difference Delta")
plt.savefig(visualizationFolder + "Histogram of Goal Difference")

plt.figure(figsize=(8,5))
plt.scatter(mergedMatches["home_xG_avg"], mergedMatches["away_xG_avg"], alpha=0.6, color='green')
plt.xlabel("Home Team xG Average")
plt.ylabel("Away Team xG Average")
plt.title("Home vs Away xG Comparison")
plt.grid(True)


seaborn.boxplot(x=mergedMatches["h2h_home_wins"], y=mergedMatches['result_numeric'])
plt.title("H2H home wins vs. match result")
plt.xlabel("H2H home win matches")
plt.ylabel("Match Result")
plt.savefig(visualizationFolder + "Boxplot of H2H Home Wins and Match Result")

seaborn.boxplot(x=mergedMatches["h2h_away_wins"], y=mergedMatches['result_numeric'])
plt.title("H2H aways wins vs. match result")
plt.xlabel("H2H away win matches")
plt.ylabel("Match Result")
plt.savefig(visualizationFolder + "Boxplot of H2H Away Wins and Match Result")


seaborn.boxplot(x=mergedMatches["h2h_draws"], y=mergedMatches['result_numeric'])
plt.title("H2H draws win vs. match result")
plt.xlabel("H2H draw matches")
plt.ylabel("Match Result")
plt.savefig(visualizationFolder + "Boxplot of H2H draws and Match Result")

seaborn.boxplot(x=mergedMatches["h2h_home_goals"], y=mergedMatches['result_numeric'])
plt.title("H2H home goals  vs. match result")
plt.xlabel("H2H home goals")
plt.ylabel("Match Result")
plt.savefig(visualizationFolder + "Boxplot of H2H Home Goals and Match Result")

seaborn.boxplot(x=mergedMatches["h2h_away_goals"], y=mergedMatches['result_numeric'])
plt.title("H2H away goals vs. match result")
plt.xlabel("H2H away goals")
plt.ylabel("Match Result")
plt.savefig(visualizationFolder + "Boxplot of H2H Away Goals and Match Result")


plt.figure(figsize=(6,6))
plt.pie(h2hValues, labels=labels, autopct='%1.1f%%', startangle=90, colors=['blue', 'red', 'gray'])
plt.title('Head-to-Head Results')
plt.savefig(visualizationFolder + "Pie chart of H2H Results")


