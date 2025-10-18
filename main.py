import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


df = pd.read_csv('Student_Performance.csv')

print("Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(df['Hours_Studied'], df['Exam_Score'], alpha=0.5)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')

plt.subplot(2, 3, 2)
plt.scatter(df['Attendance'], df['Exam_Score'], alpha=0.5)
plt.xlabel('Attendance (%)')
plt.ylabel('Exam Score')
plt.title('Attendance vs Exam Score')

plt.subplot(2, 3, 3)
plt.scatter(df['Sleep_Hours'], df['Exam_Score'], alpha=0.5)
plt.xlabel('Sleep Hours')
plt.ylabel('Exam Score')
plt.title('Sleep Hours vs Exam Score')

plt.subplot(2, 3, 4)
plt.hist(df['Exam_Score'], bins=20, edgecolor='black')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.title('Distribution of Exam Scores')

plt.tight_layout()
plt.savefig('task1_exploratory_analysis.png')
plt.show()


feature_columns = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 
                   'Previous_Scores', 'Tutoring_Sessions', 
                   'Physical_Activity']

X = df[feature_columns]
y = df['Exam_Score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n=== Model Performance ===")
print(f"Training R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Testing R² Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"Testing Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"Testing Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Exam Scores')

plt.subplot(1, 3, 2)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Scores')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(1, 3, 3)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')

plt.tight_layout()
plt.savefig('task1_model_results.png')
plt.show()

print("\n=== Feature Coefficients ===")
for feature, coef in zip(feature_columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_model.fit(X_train, y_train)
y_test_pred_poly = poly_model.predict(X_test)

print("\n=== Polynomial Regression Performance ===")
print(f"Testing R² Score: {r2_score(y_test, y_test_pred_poly):.4f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred_poly)):.4f}")

print("\nTask 1 completed successfully!")
