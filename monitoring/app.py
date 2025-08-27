import streamlit as st
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

LOG_PATH = "/logs/predictions_logs.json"
DATASET_CSV = "IMDB Dataset.csv"

# --- Initialize session state ---
if "last_log_timestamp" not in st.session_state:
    st.session_state.last_log_timestamp = None

# --- Title ---
st.title("Sentiment Monitoring Dashboard")

# --- Description ---
st.write(
    """
    This app monitors a sentiment analysis model by visualizing performance,
    data drift, and feedback accuracy.
    """
)


# --- Load Training Data ---
@st.cache_data
def load_training_data():
    df = pd.read_csv(DATASET_CSV)
    df["sentence_length"] = df["review"].apply(lambda x: len(x.split()))
    df["sentiment"] = df["sentiment"].str.lower()
    return df


# --- Load Logs ---
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "request_text",
                "predicted_sentiment",
                "true_sentiment",
            ]
        )

    logs = []
    with open(LOG_PATH, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                st.warning(f"Skipped malformed JSON on line {i+1}")
                continue

    df = pd.DataFrame(logs)

    # Add sentence length column
    if "request_text" in df.columns:
        df["sentence_length"] = df["request_text"].apply(
            lambda x: len(x.split())
        )

    return df


# --- Load DataFrames ---
imdb_df = load_training_data()
logs_df = load_logs()

# --- Check if new logs have been added ---
if not logs_df.empty and "timestamp" in logs_df.columns:
    latest_ts = logs_df["timestamp"].iloc[-1]
    if st.session_state.last_log_timestamp != latest_ts:
        st.session_state.last_log_timestamp = latest_ts
        # Refresh dashboard if new logs are detected
        st.rerun()

# --- Sidebar: Log Info ---
st.sidebar.markdown("### Log Information")
st.sidebar.write(f"Total Logs: {len(logs_df)}")
if not logs_df.empty and "timestamp" in logs_df.columns:
    st.sidebar.write(f"Last Updated: {logs_df['timestamp'].iloc[-1]}")

# --- Section: Data Drift ---
st.subheader("Data Drift: Sentence Length Distribution")
fig1, ax1 = plt.subplots()
sns.kdeplot(
    imdb_df["sentence_length"],
    ax=ax1, label="Training Data",
    color="blue")
if not logs_df.empty:
    sns.kdeplot(
        logs_df["sentence_length"],
        ax=ax1, label="Logs Data",
        color="orange"
    )
ax1.set_xlabel("Sentence Length (Words)")
ax1.set_ylabel("Density")
ax1.set_title("Sentence Length Distribution")
ax1.legend()
st.pyplot(fig1)

# --- Section: Target Drift ---
# Combine datasets for grouped bar plot
train_counts = imdb_df["sentiment"].value_counts().reset_index()
train_counts.columns = ["sentiment", "count"]
train_counts["source"] = "Training"

log_counts = logs_df["true_sentiment"].value_counts().reset_index()
log_counts.columns = ["sentiment", "count"]
log_counts["source"] = "Logged"

combined = pd.concat([train_counts, log_counts])

# Plot
fig2, ax2 = plt.subplots()
sns.barplot(x="sentiment", y="count", hue="source", data=combined, ax=ax2)
ax2.set_title("Sentiment Distribution")
st.pyplot(fig2)


# --- Section: Performance Metrics ---
st.subheader("Model Accuracy and Precision from User Feedback")

if not logs_df.empty and "true_sentiment" in logs_df.columns:
    logs_df_filtered = logs_df.dropna(subset=["true_sentiment"])
    if not logs_df_filtered.empty:
        y_true = logs_df_filtered["true_sentiment"].str.lower()
        y_pred = logs_df_filtered["predicted_sentiment"].str.lower()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, pos_label="positive", zero_division=0
        )

        if accuracy < 0.8:
            st.error(
                f"Model accuracy dropped to {accuracy:.2f}"
            )

        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Precision", f"{precision:.2f}")
    else:
        st.info("No valid user feedback available yet.")
else:
    st.info("No log data available to compute performance metrics.")

# --- Add a refresh button ---
if st.button("Refresh Dashboard"):
    st.rerun()
