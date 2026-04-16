import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(r"C:\Users\HARSHA KONDIPATI\Downloads\python project\stock_prediction_dataset.xlsx")

# Convert date column if exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

# -----------------------------
# DATA EXPLORATION
# -----------------------------
print(df.head())
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

# -----------------------------
# FEATURE (BASIC)
# -----------------------------
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
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

plt.figure()
sns.histplot(df['Returns'], kde=True)
plt.title("Returns Distribution")
plt.show()

plt.figure()
sns.boxplot(x=df['Returns'])
plt.title("Returns Boxplot")
plt.show()

# -----------------------------
# CORRELATION
# -----------------------------
corr = df.corr(numeric_only=True)
print(corr)

plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# OUTLIER DETECTION
# -----------------------------
numeric_df = df.select_dtypes(include=np.number)
z_scores = np.abs(stats.zscore(numeric_df))
outliers = np.where(z_scores > 3)
print("Outliers indices:", outliers)

# -----------------------------
# TIME ANALYSIS (FIXED)
# -----------------------------
if isinstance(df.index, pd.DatetimeIndex):
    monthly = df.resample('ME').mean()
    print(monthly.head())

# -----------------------------
# HYPOTHESIS TESTING
# -----------------------------
mid = len(df) // 2
returns1 = df['Returns'][:mid]
returns2 = df['Returns'][mid:]

# T-test
t_stat, p_val = stats.ttest_ind(returns1, returns2)
print("T-test:", t_stat, p_val)

# Normality test (only if valid size)
if len(df) <= 5000:
    shapiro_stat, shapiro_p = stats.shapiro(df['Returns'])
    print("Shapiro:", shapiro_stat, shapiro_p)
else:
    print("Shapiro test skipped (N > 5000)")
