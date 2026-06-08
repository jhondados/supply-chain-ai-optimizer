"""Demand forecasting ensemble: LSTM + Prophet + XGBoost."""
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class DemandForecastEnsemble:
    def __init__(self, forecast_horizon: int = 90):
        self.horizon = forecast_horizon
        self.prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                               daily_seasonality=False, changepoint_prior_scale=0.1)
        self.xgb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                              max_depth=6, subsample=0.8)
        self.stacker = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05)
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):
        """df must have: ds (date), y (demand), plus feature columns."""
        prophet_df = df[["ds", "y"]].copy()
        self.prophet.fit(prophet_df)
        feature_cols = [c for c in df.columns if c not in ["ds", "y"]]
        X = self.scaler.fit_transform(df[feature_cols])
        self.xgb.fit(X, df["y"].values)
        prophet_pred = self.prophet.predict(df[["ds"]])["yhat"].values
        xgb_pred = self.xgb.predict(X)
        meta_X = np.column_stack([prophet_pred, xgb_pred])
        self.stacker.fit(meta_X, df["y"].values)
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        prophet_pred = self.prophet.predict(future_df[["ds"]])["yhat"].values
        feature_cols = [c for c in future_df.columns if c not in ["ds"]]
        X = self.scaler.transform(future_df[feature_cols])
        xgb_pred = self.xgb.predict(X)
        meta_X = np.column_stack([prophet_pred, xgb_pred])
        return np.maximum(self.stacker.predict(meta_X), 0)
