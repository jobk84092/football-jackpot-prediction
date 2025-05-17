# clubelo_api.py
# Simple script to pull ELO rankings from api.clubelo.com/Fixtures

import requests
import pandas as pd
import datetime
from io import StringIO

def get_clubelo_fixtures(date=None):
    """
    Pull fixture data with ELO rankings from api.clubelo.com/Fixtures
    
    Args:
        date (str, optional): Date in format 'YYYY-MM-DD'. If None, uses today's date.
    
    Returns:
        pandas.DataFrame: Dataframe containing fixture data with ELO rankings
    """
    # Set up the API endpoint
    base_url = "http://api.clubelo.com/Fixtures"
    
    # If date is not provided, use today's date
    if date is None:
        date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Construct the full URL with the date parameter
    url = f"{base_url}/{date}"
    
    # Make the API request
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the CSV data from the response
        data = StringIO(response.text)
        df = pd.read_csv(data, sep=',')
        
        # Clean up and format the dataframe
        if not df.empty:
            # Convert date columns to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Often, the ELO columns need to be converted to numeric
            numeric_columns = ['HomeElo', 'AwayElo', 'HomeProb', 'DrawProb', 'AwayProb']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Successfully retrieved {len(df)} fixtures for {date}")
        else:
            print(f"No fixtures found for {date}")
            
        return df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Club ELO API: {e}")
        return pd.DataFrame()  # Return empty dataframe on error

def save_to_csv(df, filename="clubelo_fixtures.csv"):
    """
    Save the dataframe to a CSV file
    
    Args:
        df (pandas.DataFrame): Dataframe to save
        filename (str): Output filename
    """
    if not df.empty:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save")

# Example usage
if __name__ == "__main__":
    # Get fixtures for today
    today_fixtures = get_clubelo_fixtures()
    
    # Print the first few rows
    if not today_fixtures.empty:
        print("\nSample of retrieved data:")
        print(today_fixtures.head())
    
    # Save to CSV
    save_to_csv(today_fixtures)
    
    # Example: Get fixtures for a specific date
    # specific_date = "2023-04-15"  # Format: YYYY-MM-DD
    # date_fixtures = get_clubelo_fixtures(specific_date)
    # save_to_csv(date_fixtures, f"clubelo_fixtures_{specific_date}.csv")
