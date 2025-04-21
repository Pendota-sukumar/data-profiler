import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Optional external connectors
from sqlalchemy import create_engine
from google.cloud import bigquery
import boto3
import io

# Streamlit setup
st.set_page_config(page_title="Multi-Source EDA + AutoML", layout="wide")
st.title("🔌 Connect Data | 📊 Auto EDA | 🤖 AutoML")

# Step 1: Data source selection
source = st.sidebar.selectbox("Choose Data Source", [
    "Upload CSV",
    "SQL Database",
    "Google Drive CSV",
    "AWS S3",
    "Google BigQuery"
])

df = None

# Step 2: Data connection logic
if source == "Upload CSV":
    file = st.file_uploader("Upload your CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.success("✅ CSV loaded")

elif source == "SQL Database":
    st.subheader("🔌 SQL Database Connector")
    db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL"])
    host = st.text_input("Host")
    port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
    user = st.text_input("User")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database Name")
    sql_query = st.text_area("Enter SQL Query")

    if st.button("Connect & Load"):
        try:
            db_url = f"{'mysql+pymysql' if db_type=='MySQL' else 'postgresql'}://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(db_url)
            df = pd.read_sql(sql_query, engine)
            st.success("✅ Data loaded from SQL")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif source == "Google Drive CSV":
    st.subheader("📁 Google Drive Public File")
    drive_link = st.text_input("Paste public Google Drive link")
    if drive_link:
        try:
            file_id = drive_link.split('/d/')[1].split('/')[0]
            download_url = f"https://drive.google.com/uc?id={file_id}"
            df = pd.read_csv(download_url)
            st.success("✅ Google Drive file loaded")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif source == "AWS S3":
    st.subheader("☁️ AWS S3 CSV Loader")
    bucket = st.text_input("Bucket Name")
    key = st.text_input("File Key (e.g. folder/data.csv)")
    aws_access = st.text_input("AWS Access Key ID")
    aws_secret = st.text_input("AWS Secret Key", type="password")
    if st.button("Load from S3"):
        try:
            s3 = boto3.client('s3', aws_access_key_id=aws_access, aws_secret_access_key=aws_secret)
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            st.success("✅ Data loaded from AWS S3")
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif source == "Google BigQuery":
    st.subheader("📡 Google BigQuery Connector")
    gcp_project = st.text_input("Project ID")
    bq_query = st.text_area("BigQuery SQL Query")
    if st.button("Run Query"):
        try:
            client = bigquery.Client(project=gcp_project)
            df = client.query(bq_query).to_dataframe()
            st.success("✅ Data loaded from BigQuery")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Step 3: If data is loaded
if df is not None:
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("🧠 EDA Summary")
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum())

    st.write("Descriptive Stats:")
    st.dataframe(df.describe(include='all'))

    df_clean = df.dropna()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

    st.subheader("📊 Correlation (Filtered Heatmap)")
    
    # Limit to numeric cols
    corr = df_clean.corr()

    # Option to filter only strong correlations
    threshold = st.slider("Correlation threshold (abs)", 0.0, 1.0, 0.5, 0.05)
    filtered_corr = corr[(abs(corr) >= threshold) & (abs(corr) != 1.0)]

    # Optional: mask upper triangle for clarity
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size here
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=.5, cbar_kws={"shrink": .8})
    st.pyplot(fig)

    st.subheader("📈 Visualizations (5 Types)")
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            col_hist = st.selectbox("📌 Histogram Column", num_cols)
            fig1 = plt.figure(figsize=(8, 5))  # Smaller size
            sns.histplot(df_clean[col_hist], kde=True)
            st.pyplot(fig1)

        with col2:
            col_box = st.selectbox("📌 Boxplot Column", num_cols)
            fig2 = plt.figure(figsize=(8, 5))  # Smaller size
            sns.boxplot(y=df_clean[col_box])
            st.pyplot(fig2)

        st.write("📌 Scatter Plot")
        x_axis = st.selectbox("X Axis", num_cols, key="x")
        y_axis = st.selectbox("Y Axis", num_cols, key="y")
        fig3 = plt.figure(figsize=(8, 5))  # Smaller size
        sns.scatterplot(x=df_clean[x_axis], y=df_clean[y_axis])
        st.pyplot(fig3)

        st.write("📌 Violin Plot")
        col_vio = st.selectbox("📌 Violin Plot Column", num_cols, key="vio")
        fig4 = plt.figure(figsize=(8, 5))  # Smaller size
        sns.violinplot(y=df_clean[col_vio])
        st.pyplot(fig4)

        st.write("📌 Pairplot (sampled)")
        fig5 = sns.pairplot(df_clean.sample(min(100, len(df_clean))))
        fig5.fig.set_size_inches(8, 5)  # Smaller size
        st.pyplot(fig5)

    st.subheader("📋 EDA Suggestions")
    eda_notes = []

    if df.isnull().sum().sum() > 0:
        eda_notes.append("🔧 Missing values found — consider imputation or dropping.")

    low_var = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
    if low_var:
        eda_notes.append(f"🧹 Low variance in columns: {', '.join(low_var)} — may remove them.")

    corr = df_clean.corr().abs()
    high_corr = [col for col in corr.columns if any(corr[col] > 0.9) and col != corr.columns[-1]]
    if high_corr:
        eda_notes.append(f"⚠️ High correlation detected in: {', '.join(high_corr)} — check for multicollinearity.")

    target = df_clean.columns[-1]
    if df_clean[target].nunique() <= 10:
        imbalance = df_clean[target].value_counts().max() / df_clean[target].value_counts().min()
        if imbalance > 3:
            eda_notes.append("⚖️ Class imbalance detected — consider SMOTE or resampling.")

    if eda_notes:
        for note in eda_notes:
            st.warning(note)
    else:
        st.success("✅ Data looks balanced and well distributed!")

    st.subheader("🤖 AutoML Model Training")
    X = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]

    task = "classification" if y.nunique() <= 10 else "regression"
    model = RandomForestClassifier() if task == "classification" else RandomForestRegressor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"Detected task: **{task.upper()}**")

    if task == "classification":
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)
        fig6 = plt.figure(figsize=(8, 5))  # Smaller size
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        st.pyplot(fig6)
    else:
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))
        fig7 = plt.figure(figsize=(8, 5))  # Smaller size
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        st.pyplot(fig7)
