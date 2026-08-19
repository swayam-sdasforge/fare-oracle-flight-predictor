# AI Travel Analyst ✈️

## Project Overview
This project is a data-driven Data Science and Machine Learning solution built to analyze flight prices, uncover the hidden factors that drive ticket costs, and provide actionable recommendations to help travelers make smarter booking decisions.

## Problem Statement
Flight prices are highly volatile and confusing for the average consumer. This project aims to clean historical flight data, visualize pricing trends, mathematically identify the key drivers of price hikes, and build a Machine Learning model capable of predicting prices.

## Part 1: Exploration & Insights

### Major Factors Affecting Flight Prices
1. **Travel Class:** The single largest driver of price. Business class tickets have a massively higher baseline and a much wider price variance compared to Economy.
2. **Days Before Departure:** The booking window is the second largest factor. Prices show an aggressive, exponential spike in the final 14 days before the flight.
3. **Airline Brand:** Certain premium airlines consistently charge a "brand tax", whereas budget carriers maintain a strict low-cost ceiling.

### Insights & Recommendations
* **The Booking Window:** Travelers should book flights at least **3 to 4 weeks in advance**. The data shows that waiting for a "last-minute deal" is a myth; prices surge in the final 15 days.
* **The Layover Myth:** Do not blindly book flights with layovers assuming they are cheaper. 1-stop flights are often just as expensive as non-stop flights depending on the airline. 
* **Brand Tax on Short Flights:** Travelers should prioritize budget carriers for short-duration flights, as the premium cost of legacy airlines is not mathematically justified for short trips.

## Part 2: Machine Learning Modeling
* **Methodology:** We utilized a **Random Forest Regressor** (an ensemble decision-tree method).
* **Feature Engineering:** We used One-Hot Encoding to transform categorical variables (like Airline and Travel Class) into numerical features.
* **Results:** The model successfully predicts prices and outputs a `Feature Importance` matrix that proves Travel Class and Days Before Departure are the primary price drivers.

## Installation Instructions
1. Clone this repository.
2. Install the required dependencies:
   `pip install pandas matplotlib seaborn scikit-learn streamlit`
3. Run the data cleaning script: `python 01_data_cleaning.py`
4. Run the visualizations: `python 02_visualizations.py`
5. Run the ML model: `python 03_modeling.py`

## Technologies Used
* **Python**
* **Pandas & NumPy** (Data Cleaning & Manipulation)
* **Matplotlib & Seaborn** (Exploratory Data Analysis)
* **Scikit-Learn** (Machine Learning & Feature Engineering)

