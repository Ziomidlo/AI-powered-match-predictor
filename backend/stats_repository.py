import os
import pandas as pd

class TeamStatsRepository:

    RAW_SCHEMA = ['Team', 'Season','Team Id', 'Position', 'GP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Sh', 'SoT', 'FK', 'PK', 'Cmp', 'Att', 'Cmp%', 'CK', 'CrdY', 'CrdR', 'Fls', 'PKcon', 'OG']
    def __init__(self, stats_csv_path):
        self.file_path = stats_csv_path

    def load_stats(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        return pd.DataFrame()
    
    def _filter_to_schema(self,df):

        if df.empty: return df

        valid_cols = [c for c in self.RAW_SCHEMA if c in df.columns]

        cleaned = df[valid_cols]

        for col in self.RAW_SCHEMA:
            if col not in cleaned.columns:
                cleaned[col] = 0
        return cleaned[self.RAW_SCHEMA]
    
    def save_stats(self, df):
        if df is not None and not df.empty:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            clean_df = self._filter_to_schema(df)
            clean_df.to_csv(self.file_path, index=False)
            print("Data saved to path" + self.file_path)
        else:
            print("No data to save.")
    
    def get_promoted_profile(self, target_columns):
        df = self.load_stats()
        if df.empty : return None

        seasons = sorted(df['Season'].unique())

        if len(seasons) < 2:
            return None
        
        teams_in_prev = set(df[df['Season'] == seasons[-2]]['Team'])
        teams_in_curr = set(df[df['Season'] == seasons[-1]]['Team'])

        promoted_teams = teams_in_curr - teams_in_prev

        if not promoted_teams:
            return None

        promoted_stats = df[
            (df['Team'].isin(promoted_teams)) &
            (df['Season'] == seasons[-1])
        ]

        return promoted_stats[target_columns].mean(numeric_only=True)
    
    def get_historical_averages(self, target_columns):
        df = self.load_stats()
        if df.empty : return pd.DataFrame()

        return df.groupby("Team")[target_columns].mean(numeric_only=True)
    

    #merge and update old stats with the newest ones.
    def merge_and_update(self, new_data_df):
        history_df = self.load_stats()

        if history_df.empty:
            self.save_stats(new_data_df)
            return 
        
        current_season = new_data_df['Season'].iloc[0]

        one_season_history = history_df[
            history_df['Season'] != current_season
        ]

        cleaned_history_data = self._filter_to_schema(one_season_history)
        cleaned_new_data = self._filter_to_schema(new_data_df)
        
        updated_df = pd.concat([cleaned_history_data, cleaned_new_data], ignore_index=True)

        print(f"Updated season stats for {current_season}")

        self.save_stats(updated_df)

        return updated_df


    def get_base_stats(self):

        df = self.load_stats()

        return self._filter_to_schema(df)

