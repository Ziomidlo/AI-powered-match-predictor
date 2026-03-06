import pandas as pd
import os

# 1. The Master Mapping (Force everything to Short Names)
NAME_MAPPING = {
    "Sunderland AFC": "Sunderland",
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
    "Luton Town FC": "Luton Town",
    "Norwich City FC": "Norwich City",
    "West Bromwich Albion FC": "West Bromwich Albion"
}

def clean_matches():
    path = "cleaned_data/FullMergedMatchesInfo.csv" # Check your actual file name
    if not os.path.exists(path): return

    print(f"Cleaning Duplicates in {path}...")
    df = pd.read_csv(path)

    # 1. Standardize Names
    df['Home'] = df['Home'].replace(NAME_MAPPING)
    df['Away'] = df['Away'].replace(NAME_MAPPING)

    # 2. Drop Duplicates (The Nuclear Option)
    # We define a "Match" as unique by Date + Home + Away.
    # keep='last' means we keep the latest version (which likely has the stats).
    before = len(df)
    df = df.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last')
    after = len(df)

    print(f"Removed {before - after} duplicate matches.")
    df.to_csv(path, index=False)

def clean_stats():
    path = "cleaned_data/FullMergedSeasonStats.csv"
    if not os.path.exists(path): return

    print(f"Cleaning Duplicates in {path}...")
    df = pd.read_csv(path)

    # 1. Standardize Names
    df['Team'] = df['Team'].replace(NAME_MAPPING)

    # 2. Drop Duplicates
    # Unique by Season + Team
    before = len(df)
    df = df.drop_duplicates(subset=['Season', 'Team'], keep='last')
    after = len(df)

    print(f"Removed {before - after} duplicate stats rows.")
    df.to_csv(path, index=False)

if __name__ == "__main__":
    clean_matches()
    #clean_stats()
    print("Done. Database is clean.")