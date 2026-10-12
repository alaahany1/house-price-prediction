import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load and inspect the data
# ---------------------------------------------------------
df = pd.read_csv("Housing.csv")

print("Shape (rows, columns):", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# ---------------------------------------------------------
# 2. Preprocessing: missing values
# ---------------------------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

df = df.dropna(subset=["price"])

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after cleaning:", df.isnull().sum().sum())

# ---------------------------------------------------------
# 3. Preprocessing: categorical data
# ---------------------------------------------------------
binary_cols = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True, dtype=int)

print(df.head())

# ---------------------------------------------------------
# 4. Data visualization
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df["price"], bins=30, color="steelblue", edgecolor="black")
axes[0, 0].set_title("Distribution of House Prices")
axes[0, 0].set_xlabel("Price")
axes[0, 0].set_ylabel("Number of houses")

axes[0, 1].scatter(df["area"], df["price"], alpha=0.5, color="darkorange")
axes[0, 1].set_title("Area vs Price")
axes[0, 1].set_xlabel("Area")
axes[0, 1].set_ylabel("Price")

avg_bath = df.groupby("bathrooms")["price"].mean()
axes[1, 0].bar(avg_bath.index, avg_bath.values, color="seagreen")
axes[1, 0].set_title("Average Price by Number of Bathrooms")
axes[1, 0].set_xlabel("Bathrooms")
axes[1, 0].set_ylabel("Average price")

avg_ac = df.groupby("airconditioning")["price"].mean()
axes[1, 1].bar(["No AC", "AC"], avg_ac.values, color="purple")
axes[1, 1].set_title("Average Price: Air Conditioning")
axes[1, 1].set_ylabel("Average price")

plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print(corr["price"].sort_values(ascending=False))

# ---------------------------------------------------------
# 5. Split the data
# ---------------------------------------------------------
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# ---------------------------------------------------------
# 6. Train the Linear Regression model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

coef_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("Intercept:", model.intercept_)
print(coef_table)

# ---------------------------------------------------------
# 7. Evaluate the model
# ---------------------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print("Test MSE :", mse)
print("Test RMSE:", rmse)
print("Test R²  :", r2)
print("Train R² :", r2_score(y_train, y_train_pred))

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, color="teal")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", label="Perfect prediction")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()