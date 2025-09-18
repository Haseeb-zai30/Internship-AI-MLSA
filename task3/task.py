# ================================
# Step 1: Import Libraries
# ================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Step 2: Load Dataset
# ================================
# Replace with your dataset path
df = pd.read_csv("StudentPerformanceFactors.csv")

# Clean column names just in case
df.columns = df.columns.str.strip()

# ================================
# Step 3: Create Pass/Fail label
# ================================
df["Pass_Fail"] = df["Exam_Score"].apply(lambda x: 1 if x >= 75 else 0)

print("Pass/Fail distribution:")
print(df["Pass_Fail"].value_counts())

# ================================
# Step 4: Select Features & Labels
# ================================
X = df.drop(columns=["Exam_Score", "Pass_Fail"])
y = df["Pass_Fail"]

# Encode categorical variables
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# Step 5: Train Logistic Regression
# ================================
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)

print("\n--- Logistic Regression Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

# ================================
# Step 6: Train Decision Tree
# ================================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\n--- Decision Tree Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# ================================
# Step 7: Hyperparameter Tuning
# ================================
# Logistic Regression Tuning
param_grid_lr = {"C": [0.01, 0.1, 1, 10, 100]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring="accuracy")
grid_lr.fit(X_train, y_train)
print("\nBest Logistic Regression Params:", grid_lr.best_params_)

# Decision Tree Tuning
param_grid_dt = {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring="accuracy")
grid_dt.fit(X_train, y_train)
print("Best Decision Tree Params:", grid_dt.best_params_)

# ================================
# Step 8: Final Evaluation with Best Models
# ================================
best_log_reg = grid_lr.best_estimator_
best_dt = grid_dt.best_estimator_

y_pred_best_lr = best_log_reg.predict(X_test)
y_pred_best_dt = best_dt.predict(X_test)

print("\n--- Tuned Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_best_lr))
print("Precision:", precision_score(y_test, y_pred_best_lr))
print("Recall:", recall_score(y_test, y_pred_best_lr))

print("\n--- Tuned Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_best_dt))
print("Precision:", precision_score(y_test, y_pred_best_dt))
print("Recall:", recall_score(y_test, y_pred_best_dt))

# ================================
# Step 9: Visualizations
# ================================
# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_best_dt), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot Decision Tree
plt.figure(figsize=(16,8))
plot_tree(best_dt, filled=True, feature_names=df.drop(columns=["Exam_Score","Pass_Fail"]).columns, class_names=["Fail","Pass"])
plt.show()
