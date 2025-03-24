import requests
import pandas as pd
import json
import csv 
import seaborn as seaborn
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


uri = "https://api.football-data.org/v4/competitions/PL/"
matches = "/matches"
finishedMatches =  matches + "?status=FINISHED"
upcomingMaches = matches + "?status=TIMED"
standings = "standings"
cleanedDataFolder = "cleaned_data/"

#Get Data From API
headers = {'X-Auth-Token': ''}
response = requests.get(uri + finishedMatches , headers=headers)
dataFinished = response.json()

with open('dataPLMatches2024Finished.json', 'w') as file:
   json.dump(dataFinished, file)

with open('dataPLMatches2024Finished.json') as file:
   dataFinished = json.load(file)


responseUpcoming = requests.get(uri + upcomingMaches, headers=headers)
dataUpcoming = responseUpcoming.json()

with open('dataPLMatches2024Upcoming.json', 'w') as file:
   json.dump(dataUpcoming, file)

with open('dataPLMatches2024Upcoming.json') as file:
   dataUpcoming = json.load(file)
  
responseTeam = requests.get(uri + standings, headers=headers)
dataTeam = responseTeam.json()

with open('dataPLStandings.json', 'w') as file:
   json.dump(dataTeam, file)

with open('dataPLStandings.json') as file:
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
   CurrentAwayPossition = structuredTeamStandings[awayTeam]["Position"] 
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



#Feauture Engineering
def calculate_form(team, match_date):
   pastMatches = currentMatchesDf[
   ((currentMatchesDf['Home'] == team) | (currentMatchesDf['Away'] == team))
   & (currentMatchesDf['Date'] < match_date)
   ].sort_values(by='Date', ascending=False).head(5)

   points = 0
   totalPoints = 15
   for _, row in pastMatches.iterrows():
      if row['Home'] == team:
         if row['Result'] == 'Home win':
            points += 3
         elif row['Result'] == 'Draw':
            points += 1
      elif row['away_team'] == team:
         if row['Result'] == 'Away win':
            points += 3
         elif row['Result'] == 'Draw':
            points += 1
   percentageOfPoints = (points / totalPoints) * 100
   return percentageOfPoints.__round__(2)

def calculate_goal_difference(team, match_date):
   pastMatches = currentMatchesDf[
   ((currentMatchesDf['Home'] == team) | (currentMatchesDf['Away'] == team))
   & (currentMatchesDf['Date'] < match_date)
   ].sort_values(by='Date', ascending=False).head(5)
   goal_difference = 0

   for _, row in pastMatches.iterrows():
      if row['Home'] == team:
         goal_difference += (row['Home Goals'] - row['Away Goals'])
      elif row['away_team'] == team:
         goal_difference += (row['Away Goals'] - row['Home Goals'])
   return goal_difference



#Data for model traning
currentMatchesDf['result_numeric'] = currentMatchesDf['Result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
pastMatches['result_numeric'] = pastMatches['Result'].map({"Home win" : 1, "Draw" : 0, "Away win": -1})
currentMatchesDf["home_team_form"] = currentMatchesDf.apply(lambda row: calculate_form(row["Home"], row["Date"]), axis=1)
currentMatchesDf["away_team_form"] = currentMatchesDf.apply(lambda row: calculate_form(row["Away"], row["Date"]), axis=1)
currentMatchesDf["home_team_goal_difference"] = currentMatchesDf.apply(lambda row: calculate_goal_difference(row["Home"], row["Date"]), axis=1)
currentMatchesDf["away_team_goal_difference"] = currentMatchesDf.apply(lambda row: calculate_goal_difference(row["Away"], row["Date"]), axis=1)
currentMatchesDf["home_team_strength"] = currentMatchesDf["home_team_form"] + currentMatchesDf["home_team_goal_difference"]
currentMatchesDf["away_team_strength"] = currentMatchesDf["away_team_form"] + currentMatchesDf["away_team_goal_difference"]
currentMatchesDf["goal_diff_delta"] = currentMatchesDf["home_team_goal_difference"] - currentMatchesDf["away_team_goal_difference"]

#Correlation Analysis
corr, p_value = spearmanr(currentMatchesDf["home_team_strength"], currentMatchesDf["result_numeric"])
print(f"Spearsman's correlation: {corr:.3f}, p_value: {p_value:.3f}")
pearson_corr, p_value = pearsonr(currentMatchesDf["home_team_strength"], currentMatchesDf["result_numeric"])
print(f"Pearson's correlation: {pearson_corr:.3f}, p_value: {p_value:.3f}")

#Model Training
X = currentMatchesDf[["goal_diff_delta", "home_team_strength"]]
y = currentMatchesDf["result_numeric"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model =  LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))


#Visualization

#seaborn.boxplot(x=df2["goal_diff_delta"], y=df2["result_numeric"])
#plt.title("Goal Difference Delta vs. Match Result")
#plt.xlabel("Goal Difference Delta (Home - Away)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Goal Difference Delta and Match Result")
#plt.show()


#seaborn.boxplot(x=df2["home_team_strength"], y=df2["result_numeric"])
#plt.title("Correlation between Home Team Strength and Match Result")
#plt.xlabel("Home Team Strength (Form + Goal Difference)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Home Team Strength and Match Result")
#plt.show()

#seaborn.boxplot(x=df2["home_team_goal_difference"], y=df2["result_numeric"])
#plt.title("Correlation between Home Team Goal Difference and Match Result")
#plt.xlabel("Home Team Goal Difference (Last 5 matches)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Home Team Goal Difference and Match Result")
#plt.show()

#seaborn.boxplot(x=df2["away_team_goal_difference"], y=df2["result_numeric"])
#plt.title("Correlation between Away Team Goal Difference and Match Result")
#plt.xlabel("Away Team Goal Difference (Last 5 matches)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Away Team Goal Difference and Match Result")
#plt.show()

#seaborn.boxplot(x=df2["home_team_form"], y=df2["result_numeric"])
#plt.title("Correlation between Home Team Form and Match Result")
#plt.xlabel("Home Team Form(%)")
#plt.ylabel("Match Result")
#plt.savefig("Boxplot of Home Team Form and Match Result")
#plt.show()


# seaborn.histplot(df['result'])
# plt.title('Histogram of Match Results')
# plt.xlabel('Result')
# plt.ylabel('Frequency')
# plt.savefig('Histogram of Match Results')
# plt.show()

# seaborn.boxplot(x=df['result'], y=df['home_score'])
# plt.title('Boxplot of Home Scores by Match Result')
# plt.xlabel('Result')
# plt.ylabel('Home Score')
# plt.savefig('Boxplot of Home Scores')
# plt.show()

# seaborn.boxplot(x=df['result'], y=df['away_score'])
# plt.title('Boxplot of Away Scores by Match Result')
# plt.xlabel('Result')
# plt.ylabel('Away Score')
# plt.savefig('Boxplot of Away Scores')
# plt.show()

