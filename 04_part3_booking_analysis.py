import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os

def analyze_booking_time():
    print("Running Part 3: Cheapest Booking Time Analysis...\n")
    
    # 1. Load the cleaned data
    df = pd.read_csv('cleaned_flights.csv')
    
    if 'Days_Before_Departure' not in df.columns:
        print("Error: 'Days_Before_Departure' column missing.")
        return

    # 2. Clean Days_Before_Departure if it contains text
    df['Days_Before_Departure'] = df['Days_Before_Departure'].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True)
    df['Days_Before_Departure'] = pd.to_numeric(df['Days_Before_Departure'], errors='coerce')
    df = df.dropna(subset=['Days_Before_Departure'])

    # 3. Calculate the Global Sweet Spot (All Airlines)
    # Group by days left and find the average price for each day
    days_price = df.groupby('Days_Before_Departure')['Price'].mean().reset_index()
    
    # Find the row with the minimum average price
    sweet_spot = days_price.loc[days_price['Price'].idxmin()]
    best_day = int(sweet_spot['Days_Before_Departure'])
    best_price = sweet_spot['Price']
    
    print(f"GLOBAL SWEET SPOT FOUND:")
    print(f"Across all airlines, the absolute cheapest time to book is exactly {best_day} days in advance.")
    print(f"The average price on this day is Rs. {best_price:.2f}\n")
    
    # 3. Calculate Sweet Spot for top 3 specific airlines
    print("AIRLINE SPECIFIC SWEET SPOTS:")
    top_airlines = df['Airline'].value_counts().head(3).index
    
    plt_sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    for airline in top_airlines:
        airline_df = df[df['Airline'] == airline]
        air_days_price = airline_df.groupby('Days_Before_Departure')['Price'].mean().reset_index()
        air_sweet_spot = air_days_price.loc[air_days_price['Price'].idxmin()]
        
        print(f" - {airline}: Book {int(air_sweet_spot['Days_Before_Departure'])} days in advance (Avg: Rs. {air_sweet_spot['Price']:.2f})")
        
        # Add to our combined plot
        plt_sns.lineplot(data=air_days_price, x='Days_Before_Departure', y='Price', label=airline)

    # 4. Save a beautiful visualization
    plt.title('Cheapest Booking Time Analysis (Top 3 Airlines)')
    plt.xlabel('Days Before Departure')
    plt.ylabel('Average Ticket Price')
    plt.gca().invert_xaxis() # Countdown to 0 days
    
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    
    plt.savefig('graphs/7_part3_booking_analysis.png')
    plt.close()
    print("\nSaved detailed analysis graph to 'graphs/7_part3_booking_analysis.png'")

if __name__ == "__main__":
    analyze_booking_time()
