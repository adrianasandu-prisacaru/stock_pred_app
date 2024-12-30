import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from model_training import train_model, predict_next_day
from my_functions import fetch_stock_data, preprocess_data

def main():
    st.title("Stock Price Prediction")
    st.sidebar.header("User Inputs")

    # Sidebar Inputs
    ticker = st.sidebar.text_input("Stock Ticker", "TSLA")  # Ensure ticker is defined here
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-12-08"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    features = ['EMA_10', 'EMA_short', 'MACD', 'MACD_Signal', 'EMA_long', 'EMA_20', 'EMA_7']

    if st.sidebar.button("Train and Predict"):
        try:
            # Train the model
            ridge_model, residual_model, data, corrected_predictions, metrics = train_model(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                target_column="Close",
                best_features=features
            )

            # Display metrics
            st.write("Model Evaluation Metrics:")
            st.write(f"Mean Squared Error: {metrics['mse']:.4f}")
            st.write(f"R² Score: {metrics['r2']:.4f}")

            # Plot historical prices and predictions
            st.write("Historical Prices and Model Predictions:")
            plot_prediction_graph(data, corrected_predictions, ticker)

            # Predict the next day's price
            st.write("Predicting the next day's price...")
            next_day_prediction, next_date = predict_next_day(ridge_model, residual_model, data, features, target_column="Close")
            st.success(f"The predicted price for tomorrow is: ${next_day_prediction:.2f}")


        except Exception as e:
            st.error(f"An error occurred: {e}")



def plot_prediction_graph(data, predictions, ticker):
    """
    Plot historical prices and model predictions.
    
    Args:
        data (pd.DataFrame): Data containing historical prices (actual values).
        predictions (pd.Series): Model predictions for the historical period.
        ticker (str): Stock ticker for the title.
    """
    plt.figure(figsize=(12, 6))

    # Plot actual historical prices
    plt.plot(data.index, data['Close'], label="Actual Prices", color="blue")

    # Plot model predictions
    plt.plot(data.index, predictions, label="Predicted Prices", color="orange", linestyle="--")

    # Add title and labels
    plt.title(f"{ticker} Historical Prices and Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    st.pyplot(plt)





if __name__ == "__main__":
    main()
