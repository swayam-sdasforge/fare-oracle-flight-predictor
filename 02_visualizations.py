import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os

def create_visualizations():
    print("Loading cleaned dataset...")
    df = pd.read_csv('cleaned_flights.csv')

    # Set up the visualization style
    plt_sns.set_theme(style="whitegrid")
    
    # Create a folder to save our graphs
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    print("Generating Graph 1: Price Distribution...")
    # 1. Distribution of Flight Prices
    plt.figure(figsize=(10, 6))
    plt_sns.histplot(df['Price'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Flight Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig('graphs/1_price_distribution.png')
    plt.close()

    print("Generating Graph 2: Average Price by Airline...")
    # 2. Average Price by Airline
    plt.figure(figsize=(12, 6))
    airline_prices = df.groupby('Airline')['Price'].mean().sort_values(ascending=False).reset_index()
    plt_sns.barplot(data=airline_prices, x='Airline', y='Price', palette='viridis')
    plt.title('Average Flight Price by Airline')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('graphs/2_price_by_airline.png')
    plt.close()

    print("Generating Graph 3: Price vs Total Stops...")
    # 3. Price vs Total Stops
    plt.figure(figsize=(8, 6))
    plt_sns.barplot(data=df, x='Total_Stops', y='Price', palette='muted')
    plt.title('Average Price based on Number of Stops')
    plt.xlabel('Total Stops')
    plt.ylabel('Average Price')
    plt.savefig('graphs/3_price_by_stops.png')
    plt.close()

    print("Generating Graph 4: Price by Travel Class...")
    # 4. Price by Travel Class (Economy vs Business)
    if 'Travel_Class' in df.columns:
        plt.figure(figsize=(8, 6))
        plt_sns.boxplot(data=df, x='Travel_Class', y='Price', palette='pastel')
        plt.title('Flight Prices: Economy vs Business Class')
        plt.savefig('graphs/4_price_by_class.png')
        plt.close()

    print("Generating Graph 5: Price vs Days Before Departure...")
    # 5. Price vs Days Before Departure
    if 'Days_Before_Departure' in df.columns:
        plt.figure(figsize=(10, 6))
        # We group by days left and get the average price
        days_price = df.groupby('Days_Before_Departure')['Price'].mean().reset_index()
        plt_sns.lineplot(data=days_price, x='Days_Before_Departure', y='Price', color='red', marker='o')
        plt.title('Average Price vs. Days Left to Book')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Average Price')
        plt.gca().invert_xaxis() # Invert x-axis to show days counting down to 0
        plt.savefig('graphs/5_price_vs_days_left.png')
        plt.close()

    print("All 5 visualizations have been saved in the 'graphs' folder!")

if __name__ == "__main__":
    create_visualizations()
