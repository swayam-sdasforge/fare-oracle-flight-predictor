import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    print(f"Loading dataset from {file_path}...")
    # 1. Load the data
    df = pd.read_csv(file_path)
    
    print(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Clean 'Total_Stops'
    # Convert things like 'non-stop' to 0, and '1 stop' to 1
    stop_mapping = {'0': 0, 'non-stop': 0, '1': 1, '1 stop': 1, '2': 2, '2 stops': 2, '3': 3, '3 stops': 3}
    # We convert to string, lower case it, and map. Anything not mapped becomes NaN, then we fill with 0
    df['Total_Stops'] = df['Total_Stops'].astype(str).str.lower().str.strip().map(stop_mapping).fillna(0).astype(int)
    
    # 3. Clean 'Duration'
    # Some are decimals (1.67), some are strings ('3h 11m'). Let's convert everything to total minutes.
    def convert_duration_to_minutes(val):
        try:
            val = str(val).strip().lower()
            if 'h' in val or 'm' in val:
                hours = int(val.split('h')[0].strip()) if 'h' in val else 0
                
                minutes = 0
                if 'm' in val:
                    # extract the part before 'm' and after 'h' (if present)
                    part = val.split('h')[1] if 'h' in val else val
                    minutes = int(part.replace('m','').strip())
                return (hours * 60) + minutes
            else:
                # If it's already a decimal (e.g., 1.67 hours)
                return float(val) * 60
        except Exception as e:
            return np.nan # Return Not a Number if conversion fails
            
    df['Duration_minutes'] = df['Duration'].apply(convert_duration_to_minutes)
    
    # Drop the old Duration column as we have Duration_minutes now
    df = df.drop('Duration', axis=1)

    # 4. Handle Missing Values
    # Clean the Price column to ensure it is strictly numeric (sometimes datasets have commas or currency symbols)
    df['Price'] = df['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Fill missing 'Season' with 'Unknown'
    if 'Season' in df.columns:
        df['Season'] = df['Season'].fillna('Unknown')
    
    # Drop any remaining rows where crucial data (like Price) is completely missing
    df = df.dropna(subset=['Price', 'Duration_minutes'])
    
    print(f"Cleaned shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df

if __name__ == "__main__":
    file_name = 'flight_pricing_dataset.csv'
    if not os.path.exists(file_name):
        print(f"Error: {file_name} not found in the current directory.")
    else:
        # Test our cleaning function
        df_cleaned = load_and_clean_data(file_name)
        
        # Show the first few cleaned rows
        print("\nFirst 5 cleaned rows:")
        print(df_cleaned[['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration_minutes', 'Price']].head())
        
        # Save the cleaned data to a new CSV so we can use it for visualization later
        df_cleaned.to_csv('cleaned_flights.csv', index=False)
        print("\nSuccess! Cleaned data saved as 'cleaned_flights.csv'.")
