import pandas as pd
import os

# 1. The Mapping: "Polluted API Name" -> "Clean Kaggle Name"
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
    "Manchester United FC": "Manchester Utd", # Ensure this matches your teams.csv!
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

# 2. Files to Clean
files_to_clean = [
    "cleaned_data/mergedMatches.csv", 
    "cleaned_data/FullMergedSeasonStats.csv",
    "cleaned_data/upcoming_matches.csv",
    "cleaned_data/season-stats.csv"
]

def clean_file(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (Not found)")
        return

    print(f"Cleaning {file_path}...")
    df = pd.read_csv(file_path)

    # 3. Apply Mapping to ANY column that looks like a Team Name
    # We check for all common variations of column names
    target_columns = ['Team', 'Home', 'Away', 'HomeTeam', 'AwayTeam']
    
    changes_made = False
    for col in target_columns:
        if col in df.columns:
            # Check how many we are about to change
            matches = df[col].isin(NAME_MAPPING.keys()).sum()
            if matches > 0:
                print(f"  - Fixing {matches} names in column '{col}'")
                df[col] = df[col].replace(NAME_MAPPING)
                changes_made = True

    # 4. Save only if we touched it
    if changes_made:
        df.to_csv(file_path, index=False)
        print("  -> Saved updates.")
    else:
        print("  -> No changes needed.")

if __name__ == "__main__":
    for f in files_to_clean:
        clean_file(f)
    print("Done. Storage is now clean.")