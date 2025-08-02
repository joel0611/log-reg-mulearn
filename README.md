# Bank Churn Analysis and Prediction

This document outlines the process of analyzing customer churn from the `bank_churn.csv` dataset. The goal is to build a binary classification model using Logistic Regression to predict whether a customer will churn.

## 1. Data Loading and Preparation

First, we load the dataset using the `pandas` library in Python. The `ID` column is dropped as it is just an identifier, and the remaining columns are used as features to predict the `churn` target variable.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('bank_churn.csv')

# Define features and target
features = ['active_member', 'age', 'balance', 'country', 'credit_card', 'gender']
target = 'churn'

X = df[features]
y = df[target]
```

## 2. Exploratory Data Analysis (EDA) and Visualization
To understand the data better, we create several visualizations to explore the relationships between different features and the `churn` outcome.

### Feature Distributions and Churn Rates
This code generates a series of plots to visualize the data, including the overall churn distribution, age and balance distributions by churn status, and churn rates by country and gender.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Map numeric codes to labels for better plot readability
df['country_label'] = df['country'].map({0: 'France', 1: 'Spain', 2: 'Germany'})
df['gender_label'] = df['gender'].map({0: 'Female', 1: 'Male'})

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Bank Customer Churn Analysis')

# Plot 1: Overall Churn Distribution
sns.countplot(ax=axes[0, 0], x='churn', data=df)
axes[0, 0].set_title('Overall Churn Distribution')

# Plot 2: Age Distribution by Churn
sns.histplot(ax=axes[0, 1], data=df, x='age', hue='churn', multiple='stack')
axes[0, 1].set_title('Age Distribution by Churn Status')

# Plot 3: Churn by Country
sns.countplot(ax=axes[1, 0], x='country_label', hue='churn', data=df)
axes[1, 0].set_title('Churn Distribution by Country')

# Plot 4: Churn by Gender
sns.countplot(ax=axes[1, 1], x='gender_label', hue='churn', data=df)
axes[1, 1].set_title('Churn Distribution by Gender')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

### Feature Correlation Heatmap
A correlation heatmap helps us understand the relationships between the numerical features. This is useful for identifying potential multicollinearity.

```python
# Select features for the correlation matrix
features_for_corr = ['active_member', 'age', 'balance', 'country', 'credit_card', 'gender', 'churn']
corr_df = df[features_for_corr]

# Calculate the correlation matrix
correlation_matrix = corr_df.corr()

# Generate the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Bank Churn Features')
plt.show()
```

## 3. Modeling with Logistic Regression
We use `Logistic Regression` for our binary classification task. The data is split into training and testing sets, and the features are scaled using `StandardScaler` for better model performance.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
```

## 4. Model Evaluation
After training, we evaluate the model's performance on the test set using several key classification metrics.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
```

The `Accuracy` gives the overall percentage of correct predictions. The `Confusion Matrix` shows the number of true positives, true negatives, false positives, and false negatives. The `Classification Report` provides precision, recall, and F1-score for each class.

## 5. Saving Predictions
Finally, we save the model's predictions on the test set, along with the original feature values and the actual churn status, into a new CSV file for inspection.

```python
# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Create a results DataFrame
predictions_df = pd.DataFrame({
    'Actual_Churn': y_test.reset_index(drop=True), 
    'Predicted_Churn': y_pred,
    'Predicted_Churn_Probability': y_pred_proba
})

context_df = X_test.reset_index(drop=True)
results_df = pd.concat([context_df, predictions_df], axis=1)

# Save to CSV
results_df.to_csv('churn_predictions.csv', index=False)

print("Successfully created 'churn_predictions.csv'.")
```
