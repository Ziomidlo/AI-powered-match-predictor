import os
import pandas as pd


class MatchRepository:

    def __init__(self, file_path):
        self.file_path = file_path
        pass

    def load_data(self):
        if os.path.exists(self.file_path):           
            return pd.read_csv(self.file_path, parse_dates=['Date'])
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
        if history_data.empty:
            full_data = new_api_data
        else:
            full_data = pd.concat([history_data, new_api_data], ignore_index=True)
        full_data = full_data.drop_duplicates(
            subset=['Date', 'Home', 'Away'],
            keep='last'
        )

        full_data['Status'] = full_data['Status'].fillna('FINISHED')
        full_data['GD'] = full_data['GD'].fillna(full_data['Home Goals'] - full_data['Away Goals'])
        if full_data['Result'].dtype == 'object':
            full_data['result_numeric'] = full_data['Result'].map({"Home win" : 1, "Draw" : 0, "Away win" : -1})
        else:
            full_data['result_numeric'] = full_data['Result']
        
        full_data['xG'] = full_data.groupby('Home')['Home Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        full_data['xG.1'] = full_data.groupby('Away')['Away Goals'].transform(lambda x: x.rolling(5, min_periods=1).mean())

        self.save_data(full_data)
        
        return full_data
    
