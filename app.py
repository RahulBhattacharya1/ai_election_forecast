import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Generic Ballot: Weekly Margin Forecast", layout="wide")
st.title("Generic Ballot: Weekly Dem–Rep Margin Forecast")

# -----------------------------
# Helper functions (match training logic)
# -----------------------------
DATE_COLS_CANDIDATES = ["end_date", "start_date"]

def _to_datetime_safe(series):
    return pd.to_datetime(series, errors="coerce")

def load_training_metadata(meta_path="models/train_metadata.json"):
    if not os.path.exists(meta_path):
        st.error("Missing models/train_metadata.json. Upload it to your repo's models/ folder.")
        st.stop()
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta

def load_model(model_path="models/national_margin_forecaster.joblib"):
    if not os.path.exists(model_path):
        st.error("Missing models/national_margin_forecaster.joblib. Upload it to your repo's models/ folder.")
        st.stop()
    return joblib.load(model_path)

def load_poll_csv():
    """
    Tries to read data/generic_ballot_polls.csv from the repo.
    If not present, lets the user upload the CSV at runtime.
    """
    default_path = "data/generic_ballot_polls.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info("Loaded bundled data from data/generic_ballot_polls.csv")
        return df

    uploaded = st.file_uploader("Upload generic_ballot_polls.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("CSV uploaded.")
        return df

    st.warning("Please upload generic_ballot_polls.csv, or add it to data/ in the repo.")
    st.stop()

def basic_clean_and_weekly(df):
    """
    Match the training preprocessing:
      - parse dates
      - margin = dem - rep
      - weekly national average across pollsters
    """
    # Parse date columns if present
    for col in ["start_date", "end_date", "election_date"]:
        if col in df.columns:
            df[col] = _to_datetime_safe(df[col])

    # Numeric coercions used in training (safe)
    for col in ["dem", "rep", "ind", "numeric_grade", "pollscore", "transparency_score", "sample_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Margin
    if not {"dem", "rep"}.issubset(df.columns):
        st.error("CSV must include 'dem' and 'rep' columns.")
        st.stop()
    df["margin"] = df["dem"] - df["rep"]

    # Pick date column preference like training
    date_col = None
    if "end_date" in df.columns:
        date_col = "end_date"
    elif "start_date" in df.columns:
        date_col = "start_date"
    else:
        st.error("CSV must have end_date or start_date column.")
        st.stop()

    df = df.dropna(subset=[date_col, "margin"]).copy()

    # Weekly alignment (week start for stability)
    df["date"] = df[date_col].dt.to_period("W").dt.start_time

    # Aggregate national weekly averages
    weekly = (
        df.groupby("date", as_index=False)
          .agg(
              margin=("margin", "mean"),
              polls_per_week=("margin", "size"),
              avg_sample=("sample_size", "mean") if "sample_size" in df.columns else ("margin", "size"),
              avg_grade=("numeric_grade", "mean") if "numeric_grade" in df.columns else ("margin", "size"),
              avg_transparency=("transparency_score", "mean") if "transparency_score" in df.columns else ("margin", "size"),
              avg_pollscore=("pollscore", "mean") if "pollscore" in df.columns else ("margin", "size"),
          )
          .sort_values("date")
          .reset_index(drop=True)
    )

    # If helper cols are entirely NaN, set to 0.0 (like training)
    for c in ["avg_sample", "avg_grade", "avg_transparency", "avg_pollscore"]:
        if c not in weekly.columns:
            weekly[c] = 0.0
        elif weekly[c].isna().all():
            weekly[c] = 0.0

    return weekly

def make_lags(df_ts, target_col="margin", lags=8, roll_windows=(3, 5, 8)):
    out = df_ts.copy()
    for L in range(1, lags + 1):
        out[f"{target_col}_lag{L}"] = out[target_col].shift(L)
    for w in roll_windows:
        out[f"{target_col}_rollmean_{w}"] = out[target_col].rolling(w).mean().shift(1)
    return out

def infer_lags_and_windows_from_features(feature_cols):
    """
    Reads back the lags and rolling windows that were used during training
    from metadata['feature_cols'].
    """
    lags = set()
    rolls = set()
    for c in feature_cols:
        if c.startswith("margin_lag"):
            try:
                lags.add(int(c.replace("margin_lag", "")))
            except:
                pass
        if c.startswith("margin_rollmean_"):
            try:
                rolls.add(int(c.replace("margin_rollmean_", "")))
            except:
                pass
    lags = sorted(list(lags)) if lags else list(range(1, 9))
    rolls = sorted(list(rolls)) if rolls else [3, 5, 8]
    return lags, rolls

def recursive_forecast(last_history_df, model, horizon_weeks, feature_cols):
    """
    last_history_df: weekly df with columns: date, margin, polls_per_week, avg_* ...
    We build lag/rolling features each step using the growing history (actual + predicted).
    Returns a dataframe with future dates and predicted margins.
    """
    # Determine lags and rolling windows from training features
    lags, rolls = infer_lags_and_windows_from_features(feature_cols)
    max_lag = max(lags) if lags else 8
    rolls = tuple(rolls) if rolls else (3, 5, 8)

    # Work on a copy
    hist = last_history_df.copy().reset_index(drop=True)

    # Ensure we have enough history to create features
    if hist.shape[0] < max(max_lag, max(rolls)):
        raise ValueError(f"Not enough weekly history to create features. Need at least {max(max_lag, max(rolls))} weeks, have {hist.shape[0]}.")

    # Start forecasting one week at a time
    preds = []
    last_date = hist["date"].max()
    for step in range(1, horizon_weeks + 1):
        next_date = last_date + pd.Timedelta(days=7)

        # Build a temporary series with all rows so far
        tmp = pd.concat([hist], ignore_index=True)
        tmp = make_lags(tmp, target_col="margin", lags=max_lag, roll_windows=rolls)

        # Feature row is the last row (after adding placeholder for next week)
        # Create a placeholder row for next_date by copying aux features from latest row
        latest_aux = tmp.iloc[-1][["polls_per_week", "avg_sample", "avg_grade", "avg_transparency", "avg_pollscore"]].to_dict()
        next_row = {
            "date": next_date,
            "margin": np.nan,  # unknown yet
            "polls_per_week": latest_aux.get("polls_per_week", 0.0),
            "avg_sample": latest_aux.get("avg_sample", 0.0),
            "avg_grade": latest_aux.get("avg_grade", 0.0),
            "avg_transparency": latest_aux.get("avg_transparency", 0.0),
            "avg_pollscore": latest_aux.get("avg_pollscore", 0.0),
        }
        tmp = pd.concat([tmp, pd.DataFrame([next_row])], ignore_index=True)

        # Recompute lags/rolls including the appended row
        tmp = make_lags(tmp, target_col="margin", lags=max_lag, roll_windows=rolls)

        # Select features for the last row (new week)
        X_row = tmp.iloc[[-1]][feature_cols].copy()

        # If any lag/roll features are still NaN (very early horizon), fill with last known values
        X_row = X_row.fillna(method="ffill", axis=1).fillna(0.0)

        # Predict next margin
        yhat = float(model.predict(X_row)[0])

        preds.append({"date": next_date, "pred_margin": yhat})

        # Append prediction into history so next step can use it
        hist = pd.concat(
            [hist, pd.DataFrame([{
                "date": next_date,
                "margin": yhat,
                "polls_per_week": next_row["polls_per_week"],
                "avg_sample": next_row["avg_sample"],
                "avg_grade": next_row["avg_grade"],
                "avg_transparency": next_row["avg_transparency"],
                "avg_pollscore": next_row["avg_pollscore"],
            }])],
            ignore_index=True
        )
        last_date = next_date

    return pd.DataFrame(preds)

# -----------------------------
# Load artifacts
# -----------------------------
meta = load_training_metadata("models/train_metadata.json")
feature_cols = meta.get("feature_cols", [])
model = load_model("models/national_margin_forecaster.joblib")

# -----------------------------
# Data input
# -----------------------------
df_raw = load_poll_csv()
weekly = basic_clean_and_weekly(df_raw)

if weekly.empty:
    st.error("No weekly data after cleaning. Check your CSV.")
    st.stop()

# Show a quick peek
with st.expander("Preview weekly aggregated data"):
    st.dataframe(weekly.tail(10))

# -----------------------------
# Forecast horizon & run button
# -----------------------------
col1, col2, col3 = st.columns([1,1,2])
with col1:
    horizon = st.slider("Forecast horizon (weeks ahead)", min_value=1, max_value=24, value=8, step=1)

with col2:
    run_forecast = st.button("Run Forecast")

# -----------------------------
# Compute lags for plotting latest fit vs actual (optional)
# -----------------------------
# (Not used for prediction directly; model already trained. We just plot recent history.)
# This also validates that our feature columns exist in the engineered frame.
lags, rolls = infer_lags_and_windows_from_features(feature_cols)
max_lag_needed = max(lags) if lags else 8
weekly_lagged = make_lags(weekly, target_col="margin", lags=max_lag_needed, roll_windows=tuple(rolls))
engineered_cols_ok = all([(c in weekly_lagged.columns) for c in feature_cols])

if not engineered_cols_ok:
    st.warning("Some training features are missing in the current engineered data. Forecast may be limited. Proceeding with available history.")

# -----------------------------
# Forecast and visualize
# -----------------------------
if run_forecast:
    try:
        preds_df = recursive_forecast(weekly, model, horizon, feature_cols)

        # Merge recent actuals and forecast for plotting
        recent_actual = weekly[["date", "margin"]].tail(40).copy()
        recent_actual = recent_actual.rename(columns={"margin": "Actual margin"})

        plot_df = pd.merge(
            recent_actual,
            preds_df.rename(columns={"pred_margin": "Forecast margin"}),
            on="date",
            how="outer"
        ).sort_values("date")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_df["date"], plot_df["Actual margin"], label="Actual margin")
        ax.plot(plot_df["date"], plot_df["Forecast margin"], label="Forecast margin")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("Weekly Dem–Rep Margin: Actual vs Forecast")
        ax.set_xlabel("Week")
        ax.set_ylabel("Margin (Dem - Rep)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Show forecast table
        st.subheader("Forecast Table")
        show_df = preds_df.copy()
        show_df["date"] = show_df["date"].dt.date
        show_df["pred_margin"] = show_df["pred_margin"].round(3)
        st.dataframe(show_df)

        # Simple interpretation
        last_row = show_df.tail(1).iloc[0]
        sign = "Democratic lead" if last_row["pred_margin"] > 0 else ("Republican lead" if last_row["pred_margin"] < 0 else "Tie")
        st.info(f"On {last_row['date']}, model forecasts margin {last_row['pred_margin']} → {sign}.")

    except Exception as e:
        st.error(f"Forecast failed: {e}")

# -----------------------------
# Notes panel
# -----------------------------
with st.expander("How this app works (summary)"):
    st.markdown(
        """
- Uses the same weekly aggregation and lag/rolling features as your training notebook.
- The model is a tiny Ridge regressor (scaled), so the model file is very small and under 25 MB.
- Forecasts are computed recursively, each week using lags of recent actual/predicted margins.
- To keep it reproducible, keep your `generic_ballot_polls.csv` in `data/` or upload it.
        """
    )
