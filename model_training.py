import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost 
from my_functions import fetch_stock_data, add_indicators, preprocess_data
import numpy as np



def prepare_data(ticker, start_date, end_date, best_features):
    df = fetch_stock_data(ticker, start_date, end_date)
    df = preprocess_data(df, columns_to_lag=["Close"], lags=[1])
    df = add_indicators(df)
    df = df.dropna()

    if 'Close' not in df.columns:
        raise ValueError("'Close' column is missing after preprocessing.")

    selected_features = best_features + ['Close']
    return df[selected_features]



def train_ridge_model(X_train, y_train, alphas=None):
    """
    Train a RidgeCV model.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        alphas (list, optional): List of alpha values for RidgeCV.
    Returns:
        RidgeCV: Trained RidgeCV model.
    """
    if len(X_train) < 5:
        raise ValueError("Not enough data for cross-validation (minimum 5 samples required).")
    alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0]
    cv_splits = min(5, len(X_train))
    model = RidgeCV(alphas=alphas, cv=cv_splits)
    model.fit(X_train, y_train)
    return model


def train_model(ticker, start_date, end_date, target_column, best_features):
    """
    Train RidgeCV and residual models.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        target_column (str): Column to be predicted.
        best_features (list): List of feature columns.
    Returns:
        tuple: Ridge model, residual model, test data, and evaluation metrics.
    """
def train_model(ticker, start_date, end_date, target_column, best_features):
    data = prepare_data(ticker, start_date, end_date, best_features)

    # Define features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Train-test split
    split_idx = int(len(data) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train RidgeCV model
    ridge_model = train_ridge_model(X_train, y_train)

    # Make predictions
    y_train_pred = ridge_model.predict(X_train)
    y_test_pred = ridge_model.predict(X_test)

    # Combine predictions
    predictions = pd.Series(
        data=np.concatenate([y_train_pred, y_test_pred]),
        index=pd.concat([X_train, X_test]).index
    )

    # Train residual model using residuals
    residuals = y_test - y_test_pred
    residual_model = xgboost.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
    residual_model.fit(X_test, residuals)

    # Correct predictions with residual model
    corrected_predictions = predictions + pd.Series(
        residual_model.predict(pd.concat([X_train, X_test])),
        index=pd.concat([X_train, X_test]).index
    )

    # Evaluate model
    metrics = {
        'mse': mean_squared_error(y, corrected_predictions),
        'r2': r2_score(y, corrected_predictions)
    }

    return ridge_model, residual_model, data, corrected_predictions, metrics




def predict_next_day(ridge_model, residual_model, data, features, target_column):
    """
    Predict the next day's target value with corrections.
    Args:
        ridge_model: Trained RidgeCV model.
        residual_model: Trained residual model.
        data (pd.DataFrame): Dataset containing historical data.
        features (list): List of feature columns.
        target_column (str): Column to be predicted.
    Returns:
        tuple: Corrected predicted value and the next day's date.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
    data = data.dropna().sort_index()

    if not isinstance(data.index[-1], pd.Timestamp):
        raise ValueError(f"Index is not in proper datetime format. Last index: {data.index[-1]}")

    # Prepare the last row for prediction
    last_row = data.iloc[-1][features].to_frame().T
    ridge_prediction = ridge_model.predict(last_row)[0]
    residual_correction = residual_model.predict(last_row)[0]
    corrected_prediction = ridge_prediction + residual_correction
    next_date = pd.Timestamp(data.index[-1]) + pd.Timedelta(days=1)

    return corrected_prediction, next_date
