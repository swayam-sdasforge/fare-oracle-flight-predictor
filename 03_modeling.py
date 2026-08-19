import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

def build_model():
    print("1. Loading cleaned data...")
    df = pd.read_csv('cleaned_flights.csv')

    print("2. Feature Engineering (Converting text to numbers)...")
    
    # Clean 'Distance_km' if it contains 'km' string
    if 'Distance_km' in df.columns:
        df['Distance_km'] = df['Distance_km'].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True)
    # Clean 'Days_Before_Departure' if it contains 'days' string
    if 'Days_Before_Departure' in df.columns:
        df['Days_Before_Departure'] = df['Days_Before_Departure'].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True)
        df['Days_Before_Departure'] = pd.to_numeric(df['Days_Before_Departure'], errors='coerce')
        df['Days_Before_Departure'] = df['Days_Before_Departure'].fillna(df['Days_Before_Departure'].mean())
        
    categorical_columns = ['Airline', 'Source', 'Destination', 'Travel_Class', 'Season', 'Weekday']
    df = df.drop(['Flight_ID', 'Departure_Date', 'Departure_Time', 'Arrival_Time', 'Aircraft_Type', 'Booking_Channel'], axis=1, errors='ignore')

    # Force all other columns to be numeric
    for col in df.columns:
        if col not in categorical_columns and col != 'Price':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    # Drop any remaining NaNs in df
    df = df.dropna()
        
    # We need to convert categorical text columns into numbers so the ML model can understand them.
    # We use 'One-Hot Encoding' for this.
    categorical_columns = ['Airline', 'Source', 'Destination', 'Travel_Class', 'Season', 'Weekday']
    
    # We drop columns that are too unique or not useful for prediction right now
    df = df.drop(['Flight_ID', 'Departure_Date', 'Departure_Time', 'Arrival_Time', 'Aircraft_Type', 'Booking_Channel'], axis=1, errors='ignore')
    
    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    print("3. Splitting data into Training and Testing sets...")
    # 'X' contains all the features, 'y' contains what we want to predict (Price)
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']
    
    # Split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("4. Training the Random Forest Machine Learning Model (this might take a few seconds)...")
    # We use a Random Forest model. We limit it to 50 trees and max depth of 15 to make it train quickly.
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("5. Evaluating the Model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"--> Mean Absolute Error (MAE): {mae:.2f}")
    print(f"--> Accuracy (R-Squared Score): {r2*100:.2f}%")
    
    print("6. Extracting Feature Importance (Finding what drives the prices)...")
    # Get feature importances from the model
    feature_importances = model.feature_importances_
    features = X.columns
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10) # Top 10 factors
    
    # Plotting Feature Importance
    plt.figure(figsize=(10, 6))
    plt_sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma')
    plt.title('Top 10 Key Features Driving Flight Prices')
    plt.tight_layout()
    
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    plt.savefig('graphs/6_feature_importance.png')
    plt.close()
    print("--> Saved feature importance chart as 'graphs/6_feature_importance.png'")

if __name__ == "__main__":
    build_model()
