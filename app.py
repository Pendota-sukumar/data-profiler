import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

st.title("📊 Data Profiler App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Data Preview", df.head())

    profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
    st_profile_report(profile)
