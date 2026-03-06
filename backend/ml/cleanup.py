import os

import pandas as pd


files_to_clean = [
    "cleaned_data/LearningMatchesData.csv",
    "cleaned_data/LearningUpcomingMatches.csv",
    "cleaned_data/LearningSeasonStatsData.csv"
]
team_to_remove = "Sunderland AFC"

for file_path in files_to_clean:
    if os.path.exists(file_path):
        print(f'Cleaning {file_path}...')
        df = pd.read_csv(file_path)

        cols_to_drop = df.filter(regex="^Unnamed").columns
        if len(cols_to_drop) > 0:
            df = df.drop(columns=cols_to_drop)
            df.to_csv(file_path, index=False)
            print(f"Removed: {list(cols_to_drop)}")
        else:
            print("File is already clean.")

        initial_count = len(df)
        
        # Filtrowanie: zachowaj tylko te wiersze, które NIE MAJĄ "Sunderland AFC"
        # Przykład dla konkretnej kolumny (najbezpieczniejsza opcja):
        if 'team_name' in df.columns:
            df = df[df['team_name'] != team_to_remove]
        elif 'team' in df.columns:
            df = df[df['team'] != team_to_remove]
        else:
            # Opcja "brute force": usuwa wiersz, jeśli jakakolwiek kolumna zawiera tę nazwę
            df = df[~df.isin([team_to_remove]).any(axis=1)]

        rows_removed = initial_count - len(df)
