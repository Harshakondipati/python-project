# Stock Market Analysis Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, ttest_ind, shapiro

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# loading dataset
data = pd.read_excel("stock_prediction_dataset.xlsx")

print("\nDataset Information\n")
print(data.info())

print("\nFirst 5 Records\n")
print(data.head())

print("\nSummary Statistics\n")
print(data.describe())


# converting date column
data["Date"] = pd.to_datetime(data["Date"])

# sorting values based on date
data = data.sort_values("Date")

# setting date as index
data.set_index("Date", inplace=True)


# checking missing values
print("\nMissing Values\n")
print(data.isnull().sum())

# filling missing values
data = data.ffill()


# sentiment conversion
sentiment_map = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1
}

sentiment_cols = [
    "SocialMediaSentiment",
    "FinancialNewsSentiment",
    "BlogSentiment"
]

for col in sentiment_cols:
    data[col] = data[col].map(sentiment_map)

print("\nUpdated Dataset\n")
print(data.head())


# feature engineering
data["Returns"] = data["StockPrice"].pct_change()

# replacing NaN in returns
data["Returns"] = data["Returns"].fillna(0)

print("\nReturns Column Added\n")
print(data.head())


# ======================================================
# VISUALIZATION SECTION


# stock price over time
plt.figure(figsize=(12, 5))

plt.plot(data.index, data["StockPrice"])

plt.title("Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Stock Price")

plt.grid(True)

plt.show()


# historical vs current stock price
plt.figure(figsize=(12, 5))

plt.plot(
    data.index,
    data["HistoricalStockPrice"],
    label="Historical Price"
)

plt.plot(
    data.index,
    data["StockPrice"],
    label="Current Price"
)

plt.title("Historical vs Current Stock Price")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.show()


# returns distribution graph
plt.figure(figsize=(10, 5))

sns.histplot(   
    data["Returns"],
    bins=30,
    kde=True
)

plt.title("Distribution of Returns")

plt.xlabel("Returns")
plt.ylabel("Frequency")

plt.show()


# boxplot for returns
plt.figure(figsize=(8, 4))

sns.boxplot(x=data["Returns"])

plt.title("Returns Boxplot")

plt.show()


# sentiment count plot
plt.figure(figsize=(10, 5))

sns.countplot(
    x="SocialMediaSentiment",
    data=data
)

plt.title("Social Media Sentiment Distribution")

plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.show()


# correlation analysis
corr_matrix = data.corr(numeric_only=True)

print("\nCorrelation Matrix\n")
print(corr_matrix)

plt.figure(figsize=(10, 7))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


# ======================================================
# OUTLIER DETECTION


numeric_data = data.select_dtypes(include=np.number)

z_scores = np.abs(zscore(numeric_data))

outliers = np.where(z_scores > 3)

print("\nOutlier Detection\n")
print(outliers)


# ======================================================
# HYPOTHESIS TESTING


mid_point = len(data) // 2

first_half = data["Returns"][:mid_point]

second_half = data["Returns"][mid_point:]

t_stat, p_value = ttest_ind(first_half, second_half)

print("\nT-Test Result\n")

print("T Statistic :", t_stat)
print("P Value :", p_value)


# shapiro normality test
sample_returns = data["Returns"].sample(
    5000,
    random_state=42
)

shapiro_stat, shapiro_p = shapiro(sample_returns)

print("\nShapiro Normality Test\n")

print("Statistic :", shapiro_stat)
print("P Value :", shapiro_p)


# ======================================================
# MACHINE LEARNING SECTION


# selecting features
X = data[[
    "SocialMediaSentiment",
    "FinancialNewsSentiment",
    "BlogSentiment",
    "ForumSentiment",
    "HistoricalStockPrice"
]]

# target variable
y = data["StockPrice"]


# splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False,
    random_state=42
)


# model creation
model = LinearRegression()

# training model
model.fit(X_train, y_train)


# predictions
y_pred = model.predict(X_test)


# ======================================================
# MODEL EVALUATION


mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation\n")

print("Mean Squared Error :", mse)

print("R2 Score :", r2)


# actual vs predicted graph
plt.figure(figsize=(12, 5))

plt.plot(
    y_test.values,
    label="Actual Price"
)

plt.plot(
    y_pred,
    label="Predicted Price"
)

plt.title("Actual vs Predicted Stock Price")

plt.xlabel("Observations")
plt.ylabel("Stock Price")

plt.legend()

plt.show()


# scatter plot
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Stock Price")

plt.ylabel("Predicted Stock Price")

plt.title("Actual vs Predicted Scatter Plot")

plt.show()



