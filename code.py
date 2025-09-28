 Regression Models
# Focus: Predict Placement Score (Linear Regression) and Placement Status (Logistic Regression)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy import stats # Used for outlier detection (Z-Score)

# Set visualization and pandas options
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

print("--- Starting Task 2: Regression Models ---")

# =================================================================
#                       Part A: Data Preprocessing
# =================================================================

# Load the dataset (Assumes 'student_career_performance.csv' is in the same directory)
try:
    df_placement = pd.read_csv("student_career_performance.csv")
    print(f"Dataset loaded successfully with {df_placement.shape[0]} initial rows.")
except FileNotFoundError:
    print("Error: 'student_career_performance.csv' not found. Please ensure the file path is correct.")
    exit()

# 1. Handle Missing Values and Duplicates
df_placement.dropna(inplace=True)
duplicates_count = df_placement.duplicated().sum()
if duplicates_count > 0:
    df_placement.drop_duplicates(inplace=True)
    print(f"Removed {duplicates_count} duplicate rows.")
print(f"Cleaned dataset size: {df_placement.shape[0]} rows.")

# 2. Handle Outliers (Using Z-Score capping to prevent extreme values)
numeric_cols = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA', 'Placement_Score']
for col in numeric_cols:
    z = np.abs(stats.zscore(df_placement[col]))
    outliers = np.where(z > 3)[0] # Identify indices where z-score is > 3

    if len(outliers) > 0:
        # Cap outliers at the 99th percentile and 1st percentile
        upper_bound = df_placement[col].quantile(0.99)
        lower_bound = df_placement[col].quantile(0.01)
        df_placement[col] = np.where(df_placement[col] > upper_bound, upper_bound, df_placement[col])
        df_placement[col] = np.where(df_placement[col] < lower_bound, lower_bound, df_placement[col])
        print(f"Capped {len(outliers)} outliers in {col}.")

# 3. Prepare Features for Modeling
features = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA']
X = df_placement[features] # Independent variables

# =================================================================
#                       Part B: Linear Regression (Predict Placement_Score)
# =================================================================

y_lin = df_placement['Placement_Score'] # Target variable for Linear Regression

# Split data into training and testing sets (80% train, 20% test)
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X, y_lin, test_size=0.2, random_state=42
)

# Standardize features (Scaling input features is crucial for regression models)
scaler_lin = StandardScaler()
X_train_scaled_lin = scaler_lin.fit_transform(X_train_lin)
X_test_scaled_lin = scaler_lin.transform(X_test_lin)

# Build and Train Linear Regression Model
model_lin = LinearRegression()
model_lin.fit(X_train_scaled_lin, y_train_lin)
y_pred_lin = model_lin.predict(X_test_scaled_lin)

# Evaluate Model using Regression Metrics
mse = mean_squared_error(y_test_lin, y_pred_lin)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_lin, y_pred_lin)

print("\n--- Linear Regression Results (Predicting Placement_Score) ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score (R2): {r2:.4f}")

# Visualization: Predicted vs Actual Scores
plt.figure(figsize=(10, 6))
plt.scatter(y_test_lin, y_pred_lin, alpha=0.6)
# Plot the ideal line (where Predicted = Actual)
plt.plot([y_test_lin.min(), y_test_lin.max()], [y_test_lin.min(), y_test_lin.max()], 'r--', lw=2)
plt.xlabel('Actual Placement Score')
plt.ylabel('Predicted Placement Score')
plt.title('Linear Regression: Actual vs. Predicted Placement Score')
plt.show()

# =================================================================
#                       Part C: Logistic Regression (Predict Placed)
# =================================================================

y_log = df_placement['Placed'] # Target variable for Classification (0 or 1)

# Split data (using stratify ensures balanced class representation in train/test sets)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42, stratify=y_log
)

# Standardize features
scaler_log = StandardScaler()
X_train_scaled_log = scaler_log.fit_transform(X_train_log)
X_test_scaled_log = scaler_log.transform(X_test_log)

# Build and Train Logistic Regression Model
model_log = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
model_log.fit(X_train_scaled_log, y_train_log)
y_pred_log = model_log.predict(X_test_scaled_log)

# Evaluate Model using Classification Metrics
accuracy = accuracy_score(y_test_log, y_pred_log)
conf_matrix = confusion_matrix(y_test_log, y_pred_log)

print("\n--- Logistic Regression Results (Predicting Placed) ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred_log))

# Visualization: Confusion Matrix (Required graphical result)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Placed (0)', 'Placed (1)'],
            yticklabels=['Not Placed (0)', 'Placed (1)'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Logistic Regression: Confusion Matrix')
plt.show()

# =================================================================
#                       Part D: Comparison & Insights
# =================================================================

print("\n--- Feature Importance (Coefficients) for Insights ---")

# 1. Linear Regression Coefficients (Impact on Placement Score)
lin_coeffs = pd.DataFrame(
    {'Feature': features, 'Linear_Coeff': model_lin.coef_}
).sort_values(by='Linear_Coeff', ascending=False)
print("\n1. Linear Regression Coefficients (Impact on Placement Score):")
print("These show which features contribute most to a higher Placement_Score.")
print(lin_coeffs)

# 2. Logistic Regression Coefficients (Impact on Placement Probability)
log_coeffs = pd.DataFrame(
    {'Feature': features, 'Logistic_Coeff (Log-Odds)': model_log.coef_[0]}
).sort_values(by='Logistic_Coeff (Log-Odds)', ascending=False)
print("\n2. Logistic Regression Coefficients (Impact on Placement Probability):")
print("These show which features most increase the odds of being Placed.")
print(log_coeffs)

print("\n--- Task 2: Regression Models Complete ---")
print("Please ensure to provide your required insights and comparison in a separate report/document.")
