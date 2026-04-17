import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_excel(r"C:\Users\HARSHA KONDIPATI\Downloads\python project\stock_prediction_dataset.xlsx")

# -----------------------------
# DATE HANDLING
# -----------------------------
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.ffill()
df = df.dropna()

# Convert sentiment to numeric
sentiment_cols = ['SocialMediaSentiment', 'FinancialNewsSentiment', 'BlogSentiment']

for col in sentiment_cols:
    if col in df.columns:
        df[col] = df[col].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
if 'StockPrice' in df.columns:
    df['Returns'] = df['StockPrice'].pct_change().fillna(0)

# -----------------------------
# EDA
# -----------------------------
print(df.describe())

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
# OUTLIER DETECTION (Z-SCORE)
# -----------------------------
numeric_df = df.select_dtypes(include=np.number)
z_scores = np.abs(stats.zscore(numeric_df))
outliers = np.where(z_scores > 3)
print("Outliers indices:", outliers)

# -----------------------------
# HYPOTHESIS TESTING (T-TEST + NORMALITY)
# -----------------------------
mid = len(df) // 2
returns1 = df['Returns'][:mid]
returns2 = df['Returns'][mid:]

t_stat, p_val = stats.ttest_ind(returns1, returns2)
print("T-test:", t_stat, p_val)

if len(df) <= 5000:
    shapiro_stat, shapiro_p = stats.shapiro(df['Returns'])
    print("Shapiro:", shapiro_stat, shapiro_p)

# -----------------------------
# REGRESSION MODEL (PREDICT RETURNS)
# -----------------------------
features = df.select_dtypes(include=np.number).drop(columns=['Returns'], errors='ignore')
target = df['Returns']

X = features.dropna()
y = target.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Feature importance
print("Feature Importance:")
print(pd.Series(model.coef_, index=X.columns))

# -----------------------------
# ACTUAL VS PREDICTED
# -----------------------------
plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Returns")
plt.show()
