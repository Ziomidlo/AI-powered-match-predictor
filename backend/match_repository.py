import os
import pandas as pd
import numpy as np


class MatchRepository:

    def __init__(self, file_path):
        self.file_path = file_path
        pass

    def load_data(self):
        if os.path.exists(self.file_path):         
            df = pd.read_csv(self.file_path, parse_dates=['Date']) 
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return df
        else:
            print(f"File {self.file_path} not found. Returning empty DB.")
            return pd.DataFrame()
    
    def save_data(self, df):
        if df is not None and not df.empty:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            df.to_csv(self.file_path, index=False)
            print(f"Data saved to {self.file_path}")
        else:
            print("No data to save.")

    def merge_and_update(self, new_api_data):
        history_data = self.load_data()
        if not history_data.empty:
            history_data['Date'] = pd.to_datetime(history_data['Date'])
        if not new_api_data.empty:
            new_api_data['Date'] = pd.to_datetime(new_api_data['Date'])
        else:
            return history_data
        
        cutoff_date = new_api_data['Date'].min()

        if pd.isnull(cutoff_date):
            print("Warning: New data has no valid dates")
            return history_data
        
        print(f'Overwritting history from {cutoff_date} onwards...')

        clean_history = history_data[history_data['Date'] < cutoff_date]

        full_data = pd.concat([clean_history, new_api_data], ignore_index=True)
        
        full_data = full_data.sort_values(by='Date', ascending=True)
        full_data = full_data.drop_duplicates(
        subset=['Date', 'Home', 'Away'],
        keep='last'
        )

        full_data['Status'] = full_data['Status'].fillna('FINISHED')
        full_data['GD'] = full_data['GD'].fillna(full_data['Home Goals'] - full_data['Away Goals'])

        conditions = [
            (full_data['Home Goals'].isna()) | (full_data['Away Goals'].isna()),
            (full_data['Home Goals'] > full_data['Away Goals']),
            (full_data['Home Goals'] < full_data['Away Goals'])
        ]

        choices = ['Unknown', 'Home win', 'Away win']

        full_data['Result'] = np.select(conditions, choices, default='Draw')
        
        if full_data['Result'].dtype == 'object':
            full_data['result_numeric'] = full_data['Result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
        else:
            full_data['result_numeric'] = full_data['Result']
        
        full_data['xG'] = full_data.groupby('Home')['Home Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        full_data['xG.1'] = full_data.groupby('Away')['Away Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())

        self.save_data(full_data)
        
        return full_data
    
