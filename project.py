import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# first thing is to clean the data
#eda
# LOAD DATA
# -------------------
df = pd.read_excel(r"C:\Users\HARSHA KONDIPATI\Downloads\python project\stock_prediction_dataset.xlsx")

# Convert date column if exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# -----------------------------
# DATA EXPLORATION
# -----------------------------

print(df.info())
print(df.describe())

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.ffill()
df = df.dropna()

# Convert categorical sentiment → numeric
for col in ['SocialMediaSentiment', 'FinancialNewsSentiment', 'BlogSentiment']:
    if col in df.columns:
        df[col] = df[col].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})

# Feature engineering
if 'StockPrice' in df.columns:
    df['Returns'] = df['StockPrice'].pct_change()
    df['Returns'] = df['Returns'].fillna(0)

# -----------------------------
# VISUALIZATION
# -----------------------------
if 'StockPrice' in df.columns:
    plt.figure()
    plt.plot(df.index, df['StockPrice'])
    plt.title("Stock Price Over Time")
    plt.show()

plt.figure()
sns.histplot(df['Returns'], kde=True)
plt.title("Returns Distribution")
plt.show()

plt.figure()
sns.boxplot(x=df['Returns'])
plt.title("Returns Boxplot")
plt.show()

plt.figure()
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# EDA
# -----------------------------
print(df.corr(numeric_only=True))
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()

# -----------------------------
# OUTLIER DETECTION
# -----------------------------
z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
outliers = np.where(z_scores > 3)
print("Outliers indices:", outliers)

# -----------------------------
# TIME ANALYSIS
# -----------------------------
if isinstance(df.index, pd.DatetimeIndex):
    monthly = df.resample('M').mean()
    print(monthly.head())

# -----------------------------
# STATISTICAL TESTS
# -----------------------------
mid = len(df)//2
returns1 = df['Returns'][:mid]
returns2 = df['Returns'][mid:]

t_stat, p_val = stats.ttest_ind(returns1, returns2)
print("T-test:", t_stat, p_val)

shapiro_stat, shapiro_p = stats.shapiro(df['Returns'])
print("Shapiro:", shapiro_stat, shapiro_p)

# -----------------------------
# VIF
# -----------------------------
X_vif = df.select_dtypes(include=np.number).drop(columns=['Returns'], errors='ignore')

if len(X_vif.columns) > 1:
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    print(vif_data)

# -----------------------------
# REGRESSION
# -----------------------------
if 'StockPrice' in df.columns:
    df_ml = df.select_dtypes(include=np.number).dropna()

    X = df_ml.drop(columns=['StockPrice'])
    y = df_ml['StockPrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, preds))

# -----------------------------
# CLASSIFICATION
# -----------------------------
df['Target'] = np.where(df['Returns'] > 0, 1, 0)

df_clf = df.select_dtypes(include=np.number).dropna()

X = df_clf.drop(columns=['Target'])
y = df_clf['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
