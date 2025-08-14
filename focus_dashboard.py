import streamlit as st
import pandas as pd
import time

LOG_FILE = "focus_log.csv"

st.title("Concentration Tracking Dashboard")

placeholder = st.empty()

while True:
    try:
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            latest = df.iloc[-1]
            st.metric("Latest Concentration", f"{latest['Score']}%")
            st.metric("Total Blinks", f"{latest['Blinks']}")
            st.metric("Mouth Opens", f"{latest['Mouth Opens']}")
            st.metric("Distractions", f"{latest['Distractions']}")

            st.line_chart(df["Score"])
            st.dataframe(df.tail(10))
        else:
            st.write("Waiting for log data...")

        time.sleep(5)
    except Exception as e:
        st.error(f"Error: {e}")
        time.sleep(5)
