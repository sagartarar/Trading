import requests
import pandas as pd
import io

url = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
headers = {'User-Agent': 'Mozilla/5.0'}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        csv_content = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        print(f"Successfully downloaded Nifty 200 list.")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Number of rows: {len(df)}")
        print("First 5 symbols:")
        print(df.head())
        
        # Save to file for inspection
        df.to_csv("nifty_200_raw.csv", index=False)
    else:
        print(f"Failed to download. Status code: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
