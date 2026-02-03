import pandas as pd

from stats_repository import TeamStatsRepository

class FeatureEngineer:

    MISSING_KAGLE_COLUMNS = ['Sh', 'SoT', 'FK', 'PK', 'Cmp', 'Att', 'Cmp%', 'CK', 'CrdY', 'CrdR', 'Fls', 'PKcon', 'OG']

    def __init__(self, stats_repository, decay_rate=1.1):
        self.stats_repo = stats_repository
        self.decay_rate = decay_rate
        pass

    def _get_start_year(self, season_str):
        try:
            return int(season_str.split('/')[0])
        except (ValueError, IndexError, AttributeError):
            return None
    
    def add_season_weights(self, df):
        df = df.copy()
        df['start_year'] = df['Season'].apply(self._get_start_year)
        current_max_year = df['start_year'].max()

        if pd.isna(current_max_year):
            print("Warning: Could not determine current year. Setting weights to 1.0")
            df['season_weight'] = 1.0
            return df
        
        df['data_age'] = current_max_year - df['start_year']
        df['season_weight'] = 1 / (self.decay_rate ** df['data_age'])

        df = df.drop(columns=['start_year', 'data_age'])

        return df

    
    def impute_stats(self, new_data_df):
        new_data_df = new_data_df.copy()
        team_averages = self.stats_repo.get_historical_averages(self.MISSING_KAGLE_COLUMNS)
        promoted_profile = self.stats_repo.get_promoted_profile(self.MISSING_KAGLE_COLUMNS)

        for col in self.MISSING_KAGLE_COLUMNS:
            if col not in new_data_df.columns:
                new_data_df[col] = None
            def get_fill_value(team_name):
                if team_name in team_averages.index:
                    return team_averages.loc[team_name, col]
                elif promoted_profile is not None:
                    return promoted_profile[col]
                else:
                    return 0
                
            mask = new_data_df[col].isna()
            if mask.any():
                new_data_df.loc[mask, col] = new_data_df.loc[mask, 'Team'].apply(get_fill_value)
            
        return new_data_df
    






