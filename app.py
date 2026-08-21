import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# PAGE CONFIGURATION & UI
# ==========================================
st.set_page_config(page_title="AI Travel Analyst", page_icon="✈️", layout="wide")

# Custom CSS for Premium Vercel/SaaS Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Premium Travel Background Image with Dark Overlay */
    .stApp {
        background: linear-gradient(to bottom, rgba(14, 17, 23, 0.85), rgba(14, 17, 23, 1)), 
                    url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=2074&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Remove top padding for edge-to-edge look */
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(14, 17, 23, 0.4) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Style Buttons */
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(90deg, #00F0FF, #0077FF);
        color: white;
        border: none;
        transition: 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 240, 255, 0.3);
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 240, 255, 0.6);
        border: none;
    }
    
    /* Custom Vercel-Style Typography & Cards */
    .hero-text {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #FFFFFF, #00F0FF, #0077FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
        line-height: 1.2;
    }
    .sub-text {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    .vercel-card {
        background: rgba(20, 24, 33, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        transition: all 0.3s ease;
        backdrop-filter: blur(12px);
        margin-bottom: 15px;
    }
    .vercel-card:hover {
        border-color: rgba(0, 240, 255, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 10px 30px -10px rgba(0, 240, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Set Dark Mode for Matplotlib/Seaborn Graphs so they blend seamlessly
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#0E1117", 
    "figure.facecolor": "#0E1117",
    "grid.color": "#2D3436", 
    "text.color": "#E0E0E0",
    "axes.labelcolor": "#00F0FF",
    "xtick.color": "#E0E0E0",
    "ytick.color": "#E0E0E0"
})

# ==========================================
# CACHED FUNCTIONS
# ==========================================
@st.cache_data
def clean_data(df):
    if 'Total_Stops' in df.columns:
        stop_mapping = {'0': 0, 'non-stop': 0, '1': 1, '1 stop': 1, '2': 2, '2 stops': 2, '3': 3, '3 stops': 3}
        df['Total_Stops'] = df['Total_Stops'].astype(str).str.lower().str.strip().map(stop_mapping).fillna(0).astype(int)
    
    if 'Duration' in df.columns:
        def convert_duration_to_minutes(val):
            try:
                val = str(val).strip().lower()
                if 'h' in val or 'm' in val:
                    hours = int(val.split('h')[0].strip()) if 'h' in val else 0
                    minutes = int(val.split('h')[1].replace('m','').strip()) if 'm' in val and 'h' in val else (int(val.replace('m','').strip()) if 'm' in val else 0)
                    return (hours * 60) + minutes
                return float(val) * 60
            except:
                return np.nan
        df['Duration_minutes'] = df['Duration'].apply(convert_duration_to_minutes)
        df = df.drop('Duration', axis=1)

    if 'Price' in df.columns:
        df['Price'] = df['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
    if 'Distance_km' in df.columns:
        df['Distance_km'] = pd.to_numeric(df['Distance_km'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df['Distance_km'] = df['Distance_km'].fillna(df['Distance_km'].mean())
        
    if 'Days_Before_Departure' in df.columns:
        df['Days_Before_Departure'] = pd.to_numeric(df['Days_Before_Departure'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df['Days_Before_Departure'] = df['Days_Before_Departure'].fillna(df['Days_Before_Departure'].mean())

    df = df.dropna(subset=['Price']) 
    return df

@st.cache_resource
def train_compare_models(df):
    categorical_columns = ['Airline', 'Source', 'Destination', 'Travel_Class', 'Season', 'Weekday']
    existing_cats = [col for col in categorical_columns if col in df.columns]
    
    df_model = df.drop(['Flight_ID', 'Departure_Date', 'Departure_Time', 'Arrival_Time', 'Aircraft_Type', 'Booking_Channel'], axis=1, errors='ignore')
    
    for col in df_model.columns:
        if col not in existing_cats and col != 'Price':
            df_model[col] = pd.to_numeric(df_model[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
            df_model[col] = df_model[col].fillna(0)

    df_model = df_model.dropna()
    df_encoded = pd.get_dummies(df_model, columns=existing_cats, drop_first=True)
    
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
    }
    
    results = []
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred) * 100
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"Model": name, "Accuracy (%)": r2, "Mean Error (Rs)": mae})
        
        if name == "Random Forest":
            best_model = model
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
            
    results_df = pd.DataFrame(results).sort_values(by="Accuracy (%)", ascending=False)
    return best_model, X.columns, results_df, importance_df

# ==========================================
# SIDEBAR LOGIN SYSTEM
# ==========================================
st.sidebar.title("🔐 Portal")
role = st.sidebar.radio("Select your role:", ["Traveler", "Administrator"])

# ==========================================
# ROLE: ADMINISTRATOR (MLOps Backend)
# ==========================================
if role == "Administrator":
    password = st.sidebar.text_input("Admin Password", type="password")
    
    if password == "admin123":
        st.sidebar.success("Verified. Welcome Admin.")
        st.title("👨‍💻 Admin: MLOps Backend")
        st.markdown("Upload new flight data to automatically retrain the AI models for your consumers.")
        
        uploaded_file = st.file_uploader("Upload New Dataset (CSV)", type=['csv'])
        if uploaded_file is not None:
            with st.spinner("Ingesting and cleaning new data..."):
                raw_df = pd.read_csv(uploaded_file)
                clean_df = clean_data(raw_df)
                clean_df.to_csv("cleaned_flights.csv", index=False)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"Data Pipeline Success! {clean_df.shape[0]:,} clean records saved.")
        
        if os.path.exists("cleaned_flights.csv"):
            df = pd.read_csv("cleaned_flights.csv")
            df = clean_data(df)
            
            st.divider()
            st.subheader("📊 Exploratory Data Analysis")
            
            # ROW 1
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                sns.histplot(df['Price'], bins=30, kde=True, ax=ax1, color='#00F0FF')
                ax1.set_title("1. Price Distribution")
                st.pyplot(fig1)
            with col2:
                if 'Airline' in df.columns:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    airline_prices = df.groupby('Airline')['Price'].mean().sort_values(ascending=False).reset_index()
                    sns.barplot(data=airline_prices, x='Price', y='Airline', ax=ax2, palette='viridis')
                    ax2.set_title("2. Avg Price by Airline")
                    ax2.tick_params(axis='y', labelsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2)
            
            # ROW 2
            col3, col4 = st.columns(2)
            with col3:
                if 'Travel_Class' in df.columns:
                    fig3, ax3 = plt.subplots(figsize=(6, 5))
                    sns.boxplot(data=df, x='Travel_Class', y='Price', ax=ax3, palette='Set2', showfliers=False)
                    ax3.set_title("3. Price vs Travel Class")
                    plt.tight_layout()
                    st.pyplot(fig3)
            with col4:
                if 'Total_Stops' in df.columns:
                    fig4, ax4 = plt.subplots()
                    sns.barplot(data=df, x='Total_Stops', y='Price', ax=ax4, palette='magma')
                    ax4.set_title("4. Price vs Total Stops")
                    st.pyplot(fig4)
                    
            # ROW 3
            st.markdown("**5. Feature Correlation Matrix**")
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
            st.pyplot(fig5)
            
            st.markdown("""
            <div style='text-align: center; color: #A0AEC0; font-size: 0.9rem;'>
                <b>Dark Red (Approaching 1.0):</b> Strong positive correlation &nbsp;|&nbsp; 
                <b>Dark Blue (Approaching -1.0):</b> Strong negative correlation
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.subheader("🧠 Model Race & Evaluation")
            with st.spinner("Racing AI Models..."):
                best_model, model_cols, results_df, importance_df = train_compare_models(df)
            
            st.dataframe(results_df.style.highlight_max(subset=['Accuracy (%)'], color='lightgreen'))
            
            fig_ml, ax_ml = plt.subplots(figsize=(8, 4))
            sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax_ml, palette='magma')
            ax_ml.set_title("Feature Importance Matrix")
            st.pyplot(fig_ml)
            
    elif password != "":
        st.sidebar.error("Incorrect Password")

# ==========================================
# ROLE: TRAVELER (Vercel-Style Frontend)
# ==========================================
elif role == "Traveler":
    # Ultra-Premium Landing Page Hero
    st.markdown("""
        <div>
            <h1 class='hero-text'>Sky-High Savings</h1>
            <p class='sub-text'>Powered by advanced Machine Learning to find you the absolute best deals.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not os.path.exists("cleaned_flights.csv"):
        st.warning("⚠️ The system is currently down for maintenance. The Admin needs to upload a dataset.")
    else:
        df = pd.read_csv("cleaned_flights.csv")
        df = clean_data(df) 
        
        with st.spinner("Loading AI Engines..."):
            best_model, model_cols, _, _ = train_compare_models(df)
        
        tab1, tab2, tab3 = st.tabs(["🔍 Flight Finder", "🔮 AI Price Predictor", "📅 Booking Advice"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                search_source = st.selectbox("Departure City", df['Source'].dropna().unique() if 'Source' in df.columns else ["Unknown"])
            with f_col2:
                search_dest = st.selectbox("Arrival City", df['Destination'].dropna().unique() if 'Destination' in df.columns else ["Unknown"])
            
            if st.button("Search Flights"):
                route_df = df[(df['Source'] == search_source) & (df['Destination'] == search_dest)]
                
                if route_df.empty:
                    st.error("Sorry, no flights found for this specific route in our database.")
                else:
                    cheapest_air = route_df.groupby('Airline')['Price'].mean().idxmin()
                    cheapest_avg = route_df.groupby('Airline')['Price'].mean().min()
                    st.info(f"💡 **Route Cheat Code:** For flights from {search_source} to {search_dest}, **{cheapest_air}** is historically the cheapest airline (Avg: Rs. {cheapest_avg:,.2f}).")
                    
                    route_df = route_df.copy()
                    if 'Duration_minutes' in route_df.columns:
                        route_df['Value_Score'] = route_df['Price'] + (route_df['Duration_minutes'] * 10)
                        top_flights = route_df.sort_values(by=['Value_Score', 'Price']).head(3)
                    else:
                        top_flights = route_df.sort_values(by=['Price']).head(3)
                    
                    st.markdown("### 🏆 Top 3 Recommended Flights")
                    for i, (_, flight) in enumerate(top_flights.iterrows()):
                        duration_text = f"<b>Duration:</b> {int(flight['Duration_minutes'])} mins" if 'Duration_minutes' in flight else ""
                        stops_text = f"<b>Stops:</b> {flight['Total_Stops']}" if 'Total_Stops' in flight else ""
                        
                        # Vercel Style Custom Cards
                        st.markdown(f"""
                        <div class="vercel-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h3 style="margin: 0; color: #FFFFFF; font-weight: 600;">{flight.get('Airline', 'Unknown')}</h3>
                                    <p style="margin: 0; color: #A0AEC0; font-size: 0.9rem;">{flight.get('Travel_Class', 'Economy')} Class</p>
                                </div>
                                <div style="text-align: right;">
                                    <h2 style="margin: 0; color: #00F0FF; font-weight: 800;">Rs. {flight['Price']:,.2f}</h2>
                                    <p style="margin: 0; color: #A0AEC0; font-size: 0.9rem;">{duration_text} | {stops_text}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                u_airline = st.selectbox("Airline", df['Airline'].dropna().unique() if 'Airline' in df.columns else ["Unknown"])
                u_class = st.selectbox("Travel Class", df['Travel_Class'].dropna().unique() if 'Travel_Class' in df.columns else ["Unknown"])
                u_source = st.selectbox("From (Source)", df['Source'].dropna().unique() if 'Source' in df.columns else ["Unknown"])
                u_dest = st.selectbox("To (Destination)", df['Destination'].dropna().unique() if 'Destination' in df.columns else ["Unknown"])
            with c2:
                u_days = st.number_input("Days Before Departure", min_value=0, max_value=365, value=15)
                u_stops = st.selectbox("Number of Stops", [0, 1, 2, 3])
                u_dist = st.number_input("Flight Distance (km)", min_value=100, max_value=15000, value=1000)
                
            if st.button("🔮 Predict Price"):
                if u_days < 14:
                    st.error(f"🚨 **Warning:** You are booking only {u_days} days in advance. Our AI detects massive price surges inside the 14-day window.")
                elif u_days > 45:
                    st.success(f"🟢 **Great time to buy!** Booking {u_days} days in advance avoids the last-minute price surge.")
                else:
                    st.warning(f"🟡 **Fair timing:** You are {u_days} days out. Prices are starting to rise.")

                input_data = pd.DataFrame(0, index=[0], columns=model_cols)
                if 'Days_Before_Departure' in input_data.columns: input_data.loc[0, 'Days_Before_Departure'] = u_days
                if 'Total_Stops' in input_data.columns: input_data.loc[0, 'Total_Stops'] = u_stops
                if 'Distance_km' in input_data.columns: input_data.loc[0, 'Distance_km'] = u_dist
                
                if f"Airline_{u_airline}" in input_data.columns: input_data.loc[0, f"Airline_{u_airline}"] = 1
                if f"Travel_Class_{u_class}" in input_data.columns: input_data.loc[0, f"Travel_Class_{u_class}"] = 1
                if f"Source_{u_source}" in input_data.columns: input_data.loc[0, f"Source_{u_source}"] = 1
                if f"Destination_{u_dest}" in input_data.columns: input_data.loc[0, f"Destination_{u_dest}"] = 1
                
                prediction = best_model.predict(input_data)[0]
                st.markdown(f"""
                <div style="text-align: center; margin-top: 30px; padding: 40px; background: rgba(0, 240, 255, 0.05); border-radius: 16px; border: 1px solid rgba(0, 240, 255, 0.2);">
                    <h4 style="color: #A0AEC0; margin-bottom: 10px;">AI Estimated Ticket Price</h4>
                    <h1 style="font-size: 3.5rem; margin: 0; background: linear-gradient(90deg, #FFFFFF, #00F0FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Rs. {prediction:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            if 'Days_Before_Departure' in df.columns:
                selected_airline = st.selectbox("Find the sweet spot for:", ["All Airlines"] + list(df['Airline'].unique()))
                filter_df = df if selected_airline == "All Airlines" else df[df['Airline'] == selected_airline]
                
                days_price = filter_df.groupby('Days_Before_Departure')['Price'].mean().reset_index()
                sweet_spot = days_price.loc[days_price['Price'].idxmin()]
                
                st.info(f"💡 **Recommendation:** For {selected_airline}, the absolute cheapest time to book is **{int(sweet_spot['Days_Before_Departure'])} days** in advance (Average Price: Rs. {sweet_spot['Price']:.2f}).")
                
                fig5, ax5 = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=days_price, x='Days_Before_Departure', y='Price', color='#00F0FF', marker='o', ax=ax5)
                ax5.invert_xaxis()
                st.pyplot(fig5)
