import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Optional: for SQL, GCP, AWS
from sqlalchemy import create_engine
from google.cloud import bigquery
import boto3
import io

st.set_page_config(page_title="Multi-Source EDA & ML", layout="wide")
st.title("🔌 Connect Your Data - Auto EDA & ML")

# Step 1: Select Data Source
data_source = st.selectbox("Choose your data source:", ["Upload CSV", "SQL Database", "Google Drive (CSV)", "AWS S3", "Google BigQuery"])

df = None

# Step 2: Get Data Based on Source
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

elif data_source == "SQL Database":
    db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL"])
    host = st.text_input("Host")
    port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database")
    query = st.text_area("Enter SQL Query")

    if st.button("Connect and Load"):
        try:
            url = f"{'mysql+pymysql' if db_type == 'MySQL' else 'postgresql'}://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(url)
            df = pd.read_sql(query, con=engine)
            st.success("✅ Data loaded from SQL")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif data_source == "Google Drive (CSV)":
    public_url = st.text_input("Paste the shared Google Drive file link (public)")
    if public_url:
        file_id = public_url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        df = pd.read_csv(download_url)
        st.success("✅ Data loaded from Google Drive")

elif data_source == "AWS S3":
    bucket = st.text_input("S3 Bucket Name")
    file_key = st.text_input("CSV File Path (Key)")
    aws_access = st.text_input("AWS Access Key ID")
    aws_secret = st.text_input("AWS Secret Access Key", type="password")

    if st.button("Connect and Load"):
        try:
            s3 = boto3.client('s3', aws_access_key_id=aws_access, aws_secret_access_key=aws_secret)
            obj = s3.get_object(Bucket=bucket, Key=file_key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            st.success("✅ Data loaded from S3")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif data_source == "Google BigQuery":
    project_id = st.text_input("GCP Project ID")
    query = st.text_area("Enter your BigQuery SQL query")

    if st.button("Run Query"):
        try:
            client = bigquery.Client(project=project_id)
            df = client.query(query).to_dataframe()
            st.success("✅ Data loaded from BigQuery")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Step 3: If DataFrame Loaded
if df is not None:
    st.subheader("📊 Preview of Data")
    st.write(df.head())

    st.subheader("🧼 Data Summary")
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())

    df_clean = df.dropna()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(fig)

    st.subheader("🤖 Auto Modeling")

    X = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]

    model = RandomForestClassifier() if y.nunique() <= 10 else RandomForestRegressor()
    task_type = "Classification" if y.nunique() <= 10 else "Regression"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"🔍 Detected task: **{task_type}**")
    if task_type == "Classification":
        st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)
    else:
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        st.pyplot(fig)
