import streamlit as st
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ======================
# Page Configuration
# ======================
st.set_page_config(
    page_title="Agri-Market Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# Custom CSS (Dark Main + Light Sidebar)
# ======================
st.markdown(
    """
    <style>
    /* --- GLOBAL VARIABLES --- */
    :root {
        --primary-color: #4CAF50; /* Green */
        --bg-dark: #0E1117;       /* Main Dark BG */
        --bg-card: #262730;       /* Dark Card BG */
        --text-light: #FAFAFA;
        --text-dark: #333333;
        --sidebar-bg: #FFFFFF;
    }

    /* --- MAIN CONTAINER (Dark Mode) --- */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-light);
    }

    /* --- SIDEBAR (Light Mode) --- */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid #ddd;
    }
    
    /* Sidebar Text Overrides */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-dark) !important;
    }

    /* --- HEADERS --- */
    h1, h2, h3 {
        color: var(--primary-color) !important;
        font-weight: 700;
    }
    
    /* Custom Divider */
    hr {
        border-top: 2px solid var(--primary-color);
        margin-top: 0;
    }

    /* --- METRICS --- */
    div[data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #A0A0A0 !important;
    }

    /* --- INPUTS (Dark Theme Adaptation) --- */
    /* Dropdowns & Text Inputs in Main Area */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput div[data-baseweb="input"] > div,
    .stDateInput div[data-baseweb="input"] > div {
        background-color: var(--bg-card) !important;
        color: white !important;
        border: 1px solid #444;
    }
    
    /* Input Text Color */
    input {
        color: white !important;
    }
    
    /* Dropdown Text Color */
    div[data-baseweb="select"] span {
        color: white !important;
    }
    
    /* Dropdown Menu Items */
    ul[data-baseweb="menu"] {
        background-color: var(--bg-card) !important;
    }
    ul[data-baseweb="menu"] li {
        color: white !important;
    }

    /* Labels */
    .stSelectbox label, .stTextInput label, .stDateInput label {
        color: #E0E0E0 !important;
    }

    /* --- BUTTONS --- */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
        padding: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #45a049 !important;
    }

    /* --- FILE UPLOADER (Sidebar) --- */
    .stFileUploader {
        padding: 1rem;
        border-radius: 10px;
    }
    .stFileUploader label {
        color: var(--text-dark) !important;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background-color: #F5F5F5;
        border: 1px dashed #4CAF50;
    }
    .stFileUploader div[data-testid="stFileUploaderDropzone"] div {
        color: #333 !important;
    }

    /* --- RESULT BOXES --- */
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 5px solid #2196F3;
    }
    .note-box {
        padding: 0.5rem;
        border-radius: 5px;
        background-color: #E8F5E9;
        color: #2E7D32;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# Config / Globals
# ======================
TARGET_KEYWORDS = ("price", "mrp", "rate", "amount")
DATE_KEYWORD = "date"
SEASON_MAPPING = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "PostMonsoon", 11: "PostMonsoon"
}

# ======================
# Helper Functions
# ======================
def month_to_season(month):
    return SEASON_MAPPING.get(int(month), "Unknown")

def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return dict(r2=r2, rmse=rmse, mae=mae)

def detect_columns(df):
    price_cols = [c for c in df.columns if any(k in c.lower() for k in TARGET_KEYWORDS)]
    price_numeric = [c for c in price_cols if pd.api.types.is_numeric_dtype(df[c])]
    target_col = price_numeric[0] if price_numeric else (price_cols[0] if price_cols else None)
    date_col = next((c for c in df.columns if DATE_KEYWORD in c.lower()), None)
    crop_col = next((c for c in df.columns if any(k in c.lower() for k in ("crop","commodity","product","item","variety"))), None)
    market_col = next((c for c in df.columns if any(k in c.lower() for k in ("market","district","mandi","city","place"))), None)
    return target_col, date_col, crop_col, market_col

def add_features(df, date_col, crop_col, market_col, target_col):
    # Date Features
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dayofweek"] = df[date_col].dt.dayofweek
        df["season"] = df["month"].apply(month_to_season)
    else:
        df["season"] = "Unknown"

    # Sort for rolling/lag
    sort_cols = [c for c in [crop_col, market_col, date_col] if c]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Lags
    group_cols = [c for c in [crop_col, market_col] if c]
    if group_cols:
        df['prev_price'] = df.groupby(group_cols)[target_col].shift(1)
        df['roll_3'] = df.groupby(group_cols)[target_col].rolling(3, min_periods=1).mean().reset_index(level=group_cols, drop=True)
    else:
        df['prev_price'] = df[target_col].shift(1)
        df['roll_3'] = df[target_col].rolling(3, min_periods=1).mean()

    # Fill NaNs from lags
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# ======================
# Sidebar Layout
# ======================
with st.sidebar:
    st.header("📂 Data Control")
    
    uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
    
    if not uploaded_file:
        # CUSTOM BLACK TEXT MESSAGE
        st.markdown(
            """
            <div style='background-color: #E3F2FD; padding: 1rem; border-radius: 5px; border-left: 4px solid #2196F3; color: black; font-weight: 500;'>
                 Upload a CSV file to begin.
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### ℹ️ Instructions")
    st.markdown("""
    1. Upload a clean CSV file.
    2. Ensure it has 'Price' and 'Date' columns.
    3. The model trains automatically.
    4. Use the main panel to predict.
    """)

# ======================
# Main Layout
# ======================
st.title("🌾 Modern Agri-Market Predictor")
st.markdown("<hr>", unsafe_allow_html=True)

if uploaded_file:
    # 1. Load & Detect
    df_raw = pd.read_csv(uploaded_file)
    target_col, date_col, crop_col, market_col = detect_columns(df_raw)
    
    if not target_col:
        st.error("Could not detect a Price column.")
        st.stop()

    # 2. Preprocess
    df = add_features(df_raw.copy(), date_col, crop_col, market_col, target_col)
    df = df.dropna(subset=[target_col])
    
    X = df.drop(columns=[target_col] + ([date_col] if date_col else []))
    y = df[target_col]
    
    # Training means for fallback
    training_means = X.select_dtypes(include=np.number).mean().to_dict()

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. Pipeline
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    preproc = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, cat_features)
    ], remainder="drop")

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", preproc), ("model", model)])
    
    with st.spinner("Training model..."):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)

    # --- Section: Model Performance ---
    st.subheader("📊 Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("R2 Score", f"{metrics['r2']:.2f}")
    m2.metric("RMSE Error", f"{metrics['rmse']:.2f}")
    m3.metric("MAE Error", f"{metrics['mae']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Section: Prediction Simulator ---
    st.subheader("Price Prediction Simulator")
    
    st.markdown(
        """
        <div class="note-box">
        <b>Note:</b> This prediction uses the model trained <i>just now</i> on your uploaded data.
        </div>
        """, 
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        p_crop = st.selectbox("Select Crop", sorted(df[crop_col].unique()) if crop_col else ["Unknown"])
        p_date = st.text_input("Prediction Date", datetime.today().strftime('%Y/%m/%d'))
    with c2:
        p_market = st.selectbox("Select Market", sorted(df[market_col].unique()) if market_col else ["Unknown"])

    if st.button("✨ Calculate Prediction"):
        try:
            p_dt = pd.to_datetime(p_date)
            month = p_dt.month
            input_row = {
                "year": p_dt.year, "month": month, "day": p_dt.day, 
                "dayofweek": p_dt.weekday(), "season": month_to_season(month)
            }
            if crop_col: input_row[crop_col] = p_crop
            if market_col: input_row[market_col] = p_market
            
            # DataFrame Construction
            X_input = pd.DataFrame([input_row])
            
            # Align Columns & Fill Lags
            for col in X_train.columns:
                if col not in X_input.columns:
                    X_input[col] = training_means.get(col, 0)
            
            X_input = X_input[X_train.columns]
            
            # Predict
            pred = pipe.predict(X_input)[0]
            
            # Display
            st.markdown(f"""
                <div class="success-box">
                    <h3 style="margin:0; color:#1b5e20;">Estimated Price: ₹ {pred:,.2f}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            avg_price = training_means.get(target_col, y.mean())
            status = "above" if pred > avg_price else "below"
            
            st.markdown(f"""
                <div class="info-box">
                    ℹ️ This price is <b>{status}</b> the historical average.
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")

else:
    # --- CENTERED WELCOME MESSAGE ---
    st.markdown(
        """
        <div style='text-align: center; padding-top: 5rem;'>
            <h1 style='font-size: 3rem;'>👋 Welcome!</h1>
            <h3 style='color: #CCCCCC; margin-top: 1rem;'>Please upload your file in the sidebar to begin.</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )