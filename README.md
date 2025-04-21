# data-profiler
Sure! Here's a **README.md** template for your project. You can copy and paste this into your GitHub repository's `README.md` file:

---

# Multi-Source EDA & AutoML with Streamlit

## 📚 Overview
This project is a **Streamlit** web application that allows users to connect to multiple data sources, perform **Exploratory Data Analysis (EDA)**, and use **AutoML** for machine learning predictions. The app supports data loading from various sources such as **CSV files**, **SQL databases**, **Google Drive**, **AWS S3**, and **Google BigQuery**.

### Features:
- **Data Source Connectivity**: Connect to multiple data sources, including **CSV uploads**, **SQL databases**, **Google Drive**, **AWS S3**, and **Google BigQuery**.
- **EDA**: Perform automatic exploratory data analysis, including data cleaning, missing values analysis, feature correlation, and summary statistics.
- **Data Visualization**: Visualize the data with various charts such as histograms, boxplots, scatter plots, violin plots, and pairplots.
- **AutoML**: Automatically detect the type of machine learning task (classification or regression), train a Random Forest model, and provide evaluation metrics.
- **Suggestions**: Based on EDA, get actionable insights like missing values handling, class imbalance, and feature correlation issues.

## 🛠️ Requirements
Before running the app, ensure that you have the following dependencies installed:

- Python 3.7 or higher
- **Streamlit**
- **Pandas**
- **Numpy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **SQLAlchemy** (for SQL database connection)
- **boto3** (for AWS S3 connection)
- **google-cloud-bigquery** (for Google BigQuery connection)
- **pymysql** (for MySQL connection)

To install the dependencies, use the following command:
```bash
pip install -r requirements.txt
```

## 📊 How It Works

### 1. **Data Source Selection**
Users can select their preferred data source:
- **Upload CSV**: Upload a CSV file directly.
- **SQL Database**: Connect to a MySQL or PostgreSQL database and query data.
- **Google Drive CSV**: Provide a public Google Drive link to load CSV files.
- **AWS S3**: Provide AWS credentials to load data from S3 buckets.
- **Google BigQuery**: Run SQL queries directly on BigQuery datasets.

### 2. **Exploratory Data Analysis (EDA)**
Once the data is loaded:
- Display basic data summaries such as shape, missing values, and descriptive statistics.
- Show strong correlations between features using a **cleaned correlation heatmap**.
- Generate various types of charts (histograms, boxplots, scatter plots, pairplots) for better understanding of the dataset.
- Provide **actionable EDA insights** such as handling missing values, removing low-variance columns, and identifying multicollinearity.

### 3. **AutoML Model Training**
- Automatically detect the type of prediction task: **classification** or **regression**.
- Train a **Random Forest model** on the dataset.
- Display evaluation metrics like **accuracy**, **precision**, **recall**, **mean squared error**, and **R² score** based on the task.

### 4. **Visualization Options**
Generate visualizations for:
- **Histograms**: Display the distribution of numerical features.
- **Boxplots**: Show feature distributions and outliers.
- **Scatter Plots**: Visualize the relationship between two features.
- **Violin Plots**: Show the distribution of a feature.
- **Pairplot**: Visualize pairwise relationships between features.

### 5. **Actionable Insights**
Based on the EDA, the app will suggest:
- **Missing data imputation strategies**.
- **Multicollinearity handling**.
- **Feature importance** and potential data issues like **imbalanced classes** for classification tasks.

## 🚀 Running the App

To run the app locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   cd your-repository-name
   ```

2. **Install dependencies**:
   Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate     # For Windows
   ```

   Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Open the app** in your browser at `http://localhost:8501`.

## 🔧 Customizing the App

You can modify the following to customize the app further:
- **Machine Learning Models**: Change the model training section to use different algorithms (e.g., XGBoost, LightGBM).
- **Charts**: Add more visualizations like heatmaps for categorical features, time series plots, etc.
- **Data Sources**: Add more data connectors (e.g., for connecting to additional cloud storage, APIs, or other databases).

## 📝 Project Structure

The project structure looks like this:

```
├── app.py                # Main Streamlit application file
├── requirements.txt      # List of required dependencies
├── README.md             # Project documentation
└── templates/            # Folder for templates (optional if used in app)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📢 Feedback & Contributions

Feel free to open issues or contribute to the project! Pull requests are welcome. If you have suggestions or feedback, please feel free to **raise an issue** or **contact me**.

---

### **Author**
[Your Name](https://github.com/Pendota-sukumar?tab=repositories) 
