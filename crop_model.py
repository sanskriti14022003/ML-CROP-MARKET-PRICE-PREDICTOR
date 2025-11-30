import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

import joblib

DATA_PATH = "crop_price_data.csv"   
TARGET_KEYWORDS = ("price","mrp","rate","amount")  # used to auto-detect
DATE_KEYWORD = "date"
TOP_N_CROPS = 10    # build per-crop models for top N crops; set None for all
TEST_SPLIT_LAST_PCT = 0.2  # last 20% by date used as test
SEASON_MAPPING = {
    # default India-like seasonal mapping (adjust as needed)
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "PostMonsoon", 11: "PostMonsoon"
}
OUTPUT_DIR = "/mnt/data/season_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_columns(df):
    # detect target
    price_cols = [c for c in df.columns if any(k in c.lower() for k in TARGET_KEYWORDS)]
    price_numeric = [c for c in price_cols if pd.api.types.is_numeric_dtype(df[c])]
    target_col = price_numeric[0] if price_numeric else (price_cols[0] if price_cols else None)
    # date
    date_col = next((c for c in df.columns if DATE_KEYWORD in c.lower()), None)
    crop_col = next((c for c in df.columns if any(k in c.lower() for k in ("crop","commodity","product","item","variety"))), None)
    market_col = next((c for c in df.columns if any(k in c.lower() for k in ("market","district","mandi","city","place"))), None)
    return target_col, date_col, crop_col, market_col

def month_to_season(month):
    return SEASON_MAPPING.get(int(month), "Unknown")

def load_and_basic_clean(path):
    df = pd.read_csv(path)
    target_col, date_col, crop_col, market_col = detect_columns(df)
    if target_col is None:
        raise ValueError("No price-like column found. Please provide a numerical price column.")
    # parse date if exists
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df, target_col, date_col, crop_col, market_col

def add_season_features(df, date_col):
    if date_col:
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dayofweek"] = df[date_col].dt.dayofweek
        # cyclical encoding for month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        # season label
        df["season"] = df["month"].apply(month_to_season)
    else:
        df["season"] = "Unknown"
    return df

def create_lag_and_seasonal_aggregates(df, target_col, date_col, crop_col, market_col):
    # Ensure sorted for shifting
    if date_col and crop_col and market_col:
        df = df.sort_values([crop_col, market_col, date_col]).reset_index(drop=True)
        group_cols = [crop_col, market_col]
    elif crop_col:
        df = df.sort_values([crop_col, date_col if date_col else df.index.name or df.index]).reset_index(drop=True)
        group_cols = [crop_col]
    else:
        df = df.sort_values(date_col if date_col else df.index).reset_index(drop=True)
        group_cols = []
    # simple lags and rolling
    if group_cols:
        df['prev_price'] = df.groupby(group_cols)[target_col].shift(1)
        df['lag_7'] = df.groupby(group_cols)[target_col].shift(7)
        df['roll_7'] = df.groupby(group_cols)[target_col].rolling(7, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        df['roll_30'] = df.groupby(group_cols)[target_col].rolling(30, min_periods=1).mean().reset_index(level=group_cols, drop=True)
    else:
        df['prev_price'] = df[target_col].shift(1)
        df['lag_7'] = df[target_col].shift(7)
        df['roll_7'] = df[target_col].rolling(7, min_periods=1).mean()
        df['roll_30'] = df[target_col].rolling(30, min_periods=1).mean()

    # seasonal average for same (crop, market, season) in prior years
    if crop_col and market_col and date_col:
        # compute seasonal-year means per (crop, market, season, year)
        seasonal_mean = df.groupby([crop_col, market_col, "season", "year"])[target_col].mean().reset_index(name="season_year_mean")
        # merge back to df
        df = df.merge(seasonal_mean, on=[crop_col, market_col, "season", "year"], how="left")
        # compute prior-year seasonal mean per group by shifting by year
        df = df.sort_values([crop_col, market_col, "season", "year"])
        df['season_mean_prev_year'] = df.groupby([crop_col, market_col, "season"])['season_year_mean'].shift(1)
    else:
        df['season_mean_prev_year'] = np.nan

    # Fill NA for lags / rolling with global median to avoid dropping rows
    for c in ['prev_price','lag_7','roll_7','roll_30','season_mean_prev_year']:
        if c in df.columns:
            df[c] = df[c].fillna(df[target_col].median())

    return df

def prepare_features(df, target_col, date_col, crop_col, market_col):
    # drop exact target column and date (date used for splits)
    drop_cols = [date_col] if date_col else []
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col].copy()
    return X, y

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object','category']).columns.tolist()
    # simple imputer / ordinal encoder for trees
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                         ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    preproc = ColumnTransformer([("num", num_pipe, numeric_features),
                                ("cat", cat_pipe, cat_features)], remainder="drop")
    return preproc, numeric_features, cat_features

def time_aware_split(df, date_col, test_last_pct=0.2):
    if date_col:
        cutoff = df[date_col].quantile(1 - test_last_pct)
        train_mask = df[date_col] <= cutoff
        return train_mask
    else:
        # fallback random split: use last pct of rows as test
        n = len(df)
        cutoff_idx = int(n * (1 - test_last_pct))
        idx = df.index
        train_idx = idx <= idx[cutoff_idx - 1]
        return train_idx

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    within5 = np.mean(np.abs(y_pred - y_true) <= 0.05 * np.abs(y_true)) * 100
    within10 = np.mean(np.abs(y_pred - y_true) <= 0.10 * np.abs(y_true)) * 100
    return dict(r2=r2, rmse=rmse, mae=mae, mape=mape, within5=within5, within10=within10)


def train_per_crop_models(path=DATA_PATH, top_n=TOP_N_CROPS):
    df, target_col, date_col, crop_col, market_col = load_and_basic_clean(path)
    print("Detected:", target_col, date_col, crop_col, market_col)
    df = add_season_features(df, date_col)
    df = create_lag_and_seasonal_aggregates(df, target_col, date_col, crop_col, market_col)

    # Remove rows with missing or non-positive target
    df = df[df[target_col] > 0].copy().reset_index(drop=True)

    # If top_n specified, pick top crops by frequency
    if crop_col and top_n:
        top_crops = df[crop_col].value_counts().nlargest(top_n).index.tolist()
    elif crop_col:
        top_crops = df[crop_col].unique().tolist()
    else:
        top_crops = [None]  

    summary = []
    for crop in top_crops:
        print("\n=== Crop:", crop, "===")
        if crop is not None:
            dsub = df[df[crop_col] == crop].copy().reset_index(drop=True)
        else:
            dsub = df.copy()

        if len(dsub) < 200:
            print(" - Skipping (insufficient rows):", len(dsub))
            continue

        X, y = prepare_features(dsub, target_col, date_col, crop_col, market_col)
        preproc, num_feats, cat_feats = build_preprocessor(X)

       
        if LGB_AVAILABLE:
            model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=300, max_depth=16, n_jobs=-1, random_state=42)

        pipeline = Pipeline([("pre", preproc), ("model", model)])

        # time-aware split
        train_mask = time_aware_split(dsub, date_col, TEST_SPLIT_LAST_PCT)
        X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
        y_train, y_test = y.loc[train_mask], y.loc[~train_mask]
        print(" Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

        # Fit
        pipeline.fit(X_train, y_train)

        # Predict & evaluate
        preds = pipeline.predict(X_test)
        metrics = evaluate(y_test.values, preds)
        print(" Metrics:", metrics)
        
    bins = [0, 2000, 4000, np.inf]  

    labels = ["Low","Medium","High"]

    y_true_cls = pd.cut(y_test, bins=bins, labels=labels)
    y_pred_cls = pd.cut(preds, bins=bins, labels=labels)

    print("Price range classification report (Low/Medium/High):")
    print(classification_report(y_true_cls, y_pred_cls))

        # Per-season metrics on test set
    if "season" in dsub.columns:
            X_test2 = X_test.copy()
            # We need season in X_test -> it's categorical column present in X (preproc will remove)
            # get season from original dsub rows
            test_idx = X_test.index
            seasons = dsub.loc[test_idx, "season"].values
            df_season_eval = pd.DataFrame({"season": seasons, "y_true": y_test.values, "y_pred": preds})
            season_stats = df_season_eval.groupby("season").apply(lambda g: pd.Series(evaluate(g.y_true.values, g.y_pred.values))).transpose()
            print(" Per-season metrics (test):")
            print(season_stats.T)

        # Save model
    model_name = f"{crop if crop is not None else 'GLOBAL'}_model.joblib"
    model_path = os.path.join(OUTPUT_DIR, model_name)
    joblib.dump(pipeline, model_path)
    print(" Saved model to", model_path)

    summary.append(dict(crop=crop if crop is not None else "GLOBAL", n_rows=len(dsub), **metrics))
    return pd.DataFrame(summary)

# Run per-crop training (call this)
if __name__ == "__main__":
    df_summary = train_per_crop_models()
    print("\nSummary of trained models:")
    print(df_summary)
   


