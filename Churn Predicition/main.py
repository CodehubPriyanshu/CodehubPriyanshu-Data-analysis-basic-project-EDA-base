# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from google.colab import files

# Upload the dataset
uploaded = files.upload()

# Load the data
df = pd.read_excel('Telco_customer_churn.xlsx')

# Drop unnecessary columns
columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'Lat Long', 'Churn Label', 'Churn Reason']
df = df.drop(columns=columns_to_drop)

# Drop rows with missing values
df = df.dropna()

# Separate features (X) and target (y)
X = df.drop('Churn Value', axis=1)
y = df['Churn Value']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Feature importance
importances = rf.feature_importances_
feature_names = X_encoded.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print(feature_importance_df)

# Visualize top 10 feature importances
top_10 = feature_importance_df.head(10)
plt.figure(figsize=(10, 8))
plt.barh(top_10['Feature'], top_10['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from google.colab import files

# Upload the dataset
uploaded = files.upload()

# Load the data
df = pd.read_excel('Telco_customer_churn.xlsx')

# Drop unnecessary columns
columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'Lat Long', 'Churn Label', 'Churn Reason']
df = df.drop(columns=columns_to_drop)

# Drop rows with missing values
df = df.dropna()

# Separate features (X) and target (y)
X = df.drop('Churn Value', axis=1)
y = df['Churn Value']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Feature importance
importances = rf.feature_importances_
feature_names = X_encoded.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print(feature_importance_df)

# Visualize top 10 feature importances
top_10 = feature_importance_df.head(10)
plt.figure(figsize=(10, 8))
plt.barh(top_10['Feature'], top_10['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()