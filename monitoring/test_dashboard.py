import os
from pathlib import Path
import matplotlib
import pandas as pd
from streamlit.testing.v1 import AppTest


# ------------ Launch Without Errors Test Case ------------
def test_dashboard_launch(tmp_path, monkeypatch):
    """
    Test to ensure the dashboard launches without errors.
    """
    # Use a headless backend for matplotlib to avoid display issues
    # This ensures that the app doesn't try to open windows or require
    # a display. This is because Matplotlib has interactive GUI backends
    # (TkAgg, Qt5Agg, etc.) and non-interactive backends (Agg, Cairo, etc.)
    monkeypatch.setenv(
        "MPLBACKEND", "agg"
    )  # Tells Matplotlib which backend to pick when it's imported
    matplotlib.use(
        "Agg", force=True
    )  # Forces Matplotlib to use the non-interactive backend

    # Stub the training dataset so the app doesn't a real CSV
    stub_df = pd.DataFrame(
        {
            "review": ["This is a great movie!", "I did not like this film."],
            "sentiment": ["positive", "negative"],
        }
    )

    # Force the app to use the stub DataFrame
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: stub_df, raising=False)

    # Pretend the logs file does not exist so load_logs() returns an empty DF
    monkeypatch.setattr(os.path, "exists", lambda path: False, raising=False)

    # Run the Streamlit app file located next to this test
    app_path = Path(__file__).with_name("app.py") 
    # Increase timeout to 15 seconds for slower environments
    at = AppTest.from_file(str(app_path)).run(timeout=15)

    # Basic sanity checks
    assert not list(at.exception), f"App raised exceptions: {at.exception}"
    assert at.title  # there is a title element
    assert at.title[0].value.strip() == "Sentiment Monitoring Dashboard"

    # Check if the app runs without raising any exceptions
    assert any(
        getattr(b, "label", "") == "Refresh Dashboard" for b in at.button
    )
