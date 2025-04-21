from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Allowable file extensions for uploading
ALLOWED_EXTENSIONS = {'csv'}

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to perform EDA and generate insights
def perform_eda(df):
    insights = {}

    # Missing values
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0].index.tolist()
    if missing_columns:
        insights["missing_values"] = f"Missing values detected in the following columns: {', '.join(missing_columns)}"
    else:
        insights["missing_values"] = "No missing values."

    # Data types
    data_types = df.dtypes
    incorrect_types = data_types[data_types == 'object']
    if len(incorrect_types) > 0:
        insights["incorrect_data_types"] = f"Incorrect data types detected in the following columns: {', '.join(incorrect_types.index)}"
    else:
        insights["incorrect_data_types"] = "All data types seem correct."

    # Skewed features
    skewed = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_features = skewed[skewed > 1]
    if not skewed_features.empty:
        insights["skewed_features"] = f"Skewed features detected: {', '.join(skewed_features.index)}. Consider applying transformations (e.g., log transformation)."
    else:
        insights["skewed_features"] = "No significant skewness detected."

    # Correlation matrix
    correlation_matrix = df.corr()
    high_correlation = correlation_matrix[correlation_matrix > 0.9]
    highly_correlated_features = high_correlation.stack().index.tolist()
    if highly_correlated_features:
        insights["high_correlation"] = f"Highly correlated features: {', '.join(map(str, highly_correlated_features))}. Consider removing or combining them."
    else:
        insights["high_correlation"] = "No high correlations detected."

    # Outliers (IQR method)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    outlier_columns = outliers[outliers > 0].index.tolist()
    if outlier_columns:
        insights["outliers"] = f"Outliers detected in the following columns: {', '.join(outlier_columns)}. Consider removing or capping them."
    else:
        insights["outliers"] = "No significant outliers detected."

    return insights

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save the uploaded file locally
        file_path = os.path.join('uploads', filename)
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file.save(file_path)

        # Load the dataset
        df = pd.read_csv(file_path)

        # EDA (Exploratory Data Analysis)
        insights = perform_eda(df)

        # EDA Visualizations (Pairplot for numerical columns)
        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(numerical_columns) > 1:
            plt.figure(figsize=(10, 8))
            sns.pairplot(df[numerical_columns])
            plt.savefig('./static/images/pairplot.png')
            plt.close()

        # Auto ML setup for prediction
        df = df.dropna()
        for col in df.select_dtypes(include='object').columns:
            df[col] = LabelEncoder().fit_transform(df[col])

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Auto detect task type
        if y.nunique() <= 10 and y.dtype in [np.int64, np.int32]:
            task = 'classification'
            model = RandomForestClassifier()
        else:
            task = 'regression'
            model = RandomForestRegressor()

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        if task == 'classification':
            class_report = classification_report(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion, annot=True, fmt='d')
            plt.title("Confusion Matrix")
            plt.savefig('./static/images/confusion_matrix.png')
            plt.close()
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            plt.savefig('./static/images/actual_vs_predicted.png')
            plt.close()

        return render_template('result.html', 
                               insights=insights,
                               class_report=class_report if task == 'classification' else None, 
                               mse=mse if task == 'regression' else None,
                               r2=r2 if task == 'regression' else None,
                               image_path1='images/pairplot.png',
                               image_path2='images/confusion_matrix.png' if task == 'classification' else 'images/actual_vs_predicted.png')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

