import requests
import pandas as pd
import json
import os

teamNameMapping = {
    "Bournemouth": "AFC Bournemouth",
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Chelsea": "Chelsea FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Ipswich Town": "Ipswich Town FC",
    "Leicester City": "Leicester City FC",
    "Liverpool": "Liverpool FC",
    "Manchester City": "Manchester City FC",
    "Manchester Utd": "Manchester United FC",
    "Newcastle Utd": "Newcastle United FC",
    "Nott'ham Forest": "Nottingham Forest FC",
    "Southampton": "Southampton FC",
    "Tottenham": "Tottenham Hotspur FC",
    "West Ham": "West Ham United FC",
    "Wolves": "Wolverhampton Wanderers FC"
}

os.makedirs("cleaned_data", exist_ok=True)

selectedSeasons = ["2020/2021", "2021/2022", "2022/2023", "2023/2024"]

teamsDf = pd.read_csv("teams.csv")
teamsDf['Team'] = teamsDf['Team'].replace(teamNameMapping)
teamsDf.info()
teamMapping = dict(zip(teamsDf['Team'], teamsDf['Id']))

def getTeamId(teamName):
    return teamMapping.get(teamName, -1)

def getResult(homeGoals, awayGoals):
    if homeGoals > awayGoals:
      result = "Home Win"
    elif homeGoals < awayGoals:
       result = "Away Win"
    else:
       result = "Draw"
    return result



seasonStatsDf = pd.read_csv("seasonstats.csv")
filteredSeasonStatsDf = seasonStatsDf[seasonStatsDf['Season'].isin(selectedSeasons)]
filteredSeasonStatsDf = filteredSeasonStatsDf.drop('Unnamed: 0', axis=1)
filteredSeasonStatsDf = filteredSeasonStatsDf.rename(columns={'Squad' : 'Team'})
filteredSeasonStatsDf['Team'] = filteredSeasonStatsDf['Team'].replace(teamNameMapping)
filteredSeasonStatsDf['GD'] = filteredSeasonStatsDf['GF'] - filteredSeasonStatsDf['GA']
filteredSeasonStatsDf['Team Id'] =  filteredSeasonStatsDf["Team"].map(getTeamId)
filteredSeasonStatsDf = filteredSeasonStatsDf.sort_values(by=['Season', 'Pts', 'GD'], ascending=[True, False, False])
filteredSeasonStatsDf['Position'] = filteredSeasonStatsDf.groupby('Season')['Pts'].rank(
    ascending=False, 
    method='first'
).astype(int)
filteredSeasonStatsDf.info()


kaggleMatchesDf = pd.read_csv('matches.csv')
filteredKaggleMatchesDf = kaggleMatchesDf[kaggleMatchesDf['Season'].isin(selectedSeasons)]
filteredKaggleMatchesDf = filteredKaggleMatchesDf.drop('Unnamed: 0', axis=1)
filteredKaggleMatchesDf = filteredKaggleMatchesDf.drop('Attendance', axis=1)
filteredKaggleMatchesDf = filteredKaggleMatchesDf.dropna(axis=0)
filteredKaggleMatchesDf['Home'] = filteredKaggleMatchesDf['Home'].replace(teamNameMapping)
filteredKaggleMatchesDf['Away'] = filteredKaggleMatchesDf['Away'].replace(teamNameMapping)
filteredKaggleMatchesDf['Home Id'] = filteredKaggleMatchesDf['Home'].map(getTeamId)
filteredKaggleMatchesDf['Away Id'] = filteredKaggleMatchesDf['Away'].map(getTeamId)
filteredKaggleMatchesDf['Result'] = filteredKaggleMatchesDf.apply(lambda row: getResult(row['Home Goals'], row['Away Goals']), axis=1)
filteredKaggleMatchesDf.info()

teamsDf.to_csv("cleaned_data/teams.csv", index=False)
filteredSeasonStatsDf.to_csv("cleaned_data/season-stats.csv", index=False)
filteredKaggleMatchesDf.to_csv('cleaned_data/past-matches.csv', index=False)
