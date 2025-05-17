import requests
import pandas as pd
from datetime import datetime

def download_clubelo():
    url = "https://www.clubelo.com/2024-05-01/ClubElo.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print("Attempting to download ClubElo data...")
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(pd.StringIO(response.text))
            df['From'] = pd.to_datetime(df['From'])
            df.to_csv('clubelo.csv', index=False)
            print(f"Successfully downloaded ClubElo data with {len(df)} teams")
            print("Sample of downloaded data:")
            print(df.head())
            return True
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading ClubElo data: {e}")
        return False

if __name__ == "__main__":
    download_clubelo() 