import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('shopping_data.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Convert 'CustomerID' to integer and 'Age' to numeric
df['CustomerID'] = df['CustomerID'].astype(int)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Convert 'Annual Income (k$)' and 'Spending Score (1-100)' to numeric
df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'], errors='coerce')
df['Spending Score (1-100)'] = pd.to_numeric(df['Spending Score (1-100)'], errors='coerce')

# Display updated info after type conversions
print("\nUpdated Dataset Info:")
print(df.info())

# Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Gender distribution
print("\nGender Distribution:")
print(df['Genre'].value_counts())

# Visualize age distribution
plt.figure(figsize=(10, 6))
df['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Scatter plot of Annual Income vs Spending Score
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

print("\nCorrelation between Annual Income and Spending Score:")
print(df['Annual Income (k$)'].corr(df['Spending Score (1-100)']))