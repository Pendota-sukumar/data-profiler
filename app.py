import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Configure Streamlit
st.set_page_config(page_title="Auto EDA + ML", layout="wide")
st.title("📊 Advanced Auto EDA & ML Dashboard")

# Upload File
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully")

    # Show Data
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Basic Info
    st.subheader("📋 Dataset Summary")
    st.write(f"Rows: `{df.shape[0]}` | Columns: `{df.shape[1]}`")
    st.write("**Missing Values:**")
    st.dataframe(df.isnull().sum())

    # Data Types
    st.write("**Data Types:**")
    st.write(df.dtypes)

    # Descriptive Stats
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(df.describe(include='all'))

    # Clean Data
    df_clean = df.dropna()
    cat_cols = df_clean.select_dtypes(include='object').columns
    for col in cat_cols:
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

    # Correlation
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig)

    # EDA - 5 Types of Charts
    st.subheader("📈 Exploratory Charts")

    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            st.write("1️⃣ **Histogram**")
            col_hist = st.selectbox("Choose column for histogram", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df_clean[col_hist], kde=True, ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("2️⃣ **Boxplot**")
            col_box = st.selectbox("Choose column for boxplot", num_cols)
            fig, ax = plt.subplots()
            sns.boxplot(y=df_clean[col_box], ax=ax)
            st.pyplot(fig)

        st.write("3️⃣ **Scatter Plot**")
        x_col = st.selectbox("X axis", num_cols, key='scatter_x')
        y_col = st.selectbox("Y axis", num_cols, key='scatter_y')
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_clean[x_col], y=df_clean[y_col], ax=ax)
        st.pyplot(fig)

        st.write("4️⃣ **Violin Plot**")
        col_violin = st.selectbox("Choose column for violin plot", num_cols)
        fig, ax = plt.subplots()
        sns.violinplot(y=df_clean[col_violin], ax=ax)
        st.pyplot(fig)

        st.write("5️⃣ **Pairplot**")
        if len(num_cols) > 1:
            st.info("Showing pairplot for a sample (max 100 rows)")
            fig = sns.pairplot(df_clean.sample(min(100, len(df_clean))))
            st.pyplot(fig)

    # Smart EDA Summary
    st.subheader("🧠 Smart EDA Summary")
    suggestions = []

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        suggestions.append("🔧 Data has missing values. Consider using mean/median imputation or dropping rows.")

    # Low variance
    low_var_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
    if low_var_cols:
        suggestions.append(f"🧹 Columns with zero/low variance: {', '.join(low_var_cols)}. Consider removing them.")

    # Highly correlated features
    corr_matrix = df_clean.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_corr = [col for col in upper.columns if any(upper[col] > 0.9)]
    if highly_corr:
        suggestions.append(f"🧠 Highly correlated columns detected: {', '.join(highly_corr)}. May cause multicollinearity.")

    # Class imbalance
    target = df_clean.columns[-1]
    if df_clean[target].nunique() <= 10:
        class_counts = df_clean[target].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio > 3:
            suggestions.append("⚖️ Class imbalance detected. Consider SMOTE, undersampling, or class weights.")

    if suggestions:
        for s in suggestions:
            st.warning(s)
    else:
        st.success("✅ No major data issues detected!")

    # AutoML
    st.subheader("🤖 AutoML Model Training")

    X = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]

    if y.nunique() <= 10:
        model = RandomForestClassifier()
        task = "Classification"
    else:
        model = RandomForestRegressor()
        task = "Regression"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"Detected Task: **{task}**")

    if task == "Classification":
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)

    else:
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
