# @title Default title text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('Student.csv')

# Print all column names
print("Column names:")
print(df.columns.tolist())

# Check for columns containing 'stress' (case-insensitive)
stress_columns = [col for col in df.columns if 'stress' in col.lower()]
print("\nColumns containing 'stress':")
print(stress_columns)

# If a stress-related column is found, use it; otherwise, use a placeholder
stress_column = stress_columns[0] if stress_columns else 'Stress Level'
print(f"\nUsing stress column: {stress_column}")

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Display summary statistics of numerical columns
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Visualize the distribution of numerical features
numerical_features = ['Height(CM)', '10th Mark', '12th Mark', 'college mark']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    sns.histplot(df[feature], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Visualize the count of categorical features
categorical_features = ['Gender', 'Department', 'Certification Course', 'hobbies', 'daily studing time', 'prefer to study in', 'Do you like your degree?', 'part-time job']
fig, axes = plt.subplots(4, 2, figsize=(20, 30))
for i, feature in enumerate(categorical_features):
    sns.countplot(y=df[feature], ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Count of {feature}')
    axes[i//2, i%2].set_xlabel('Count')
plt.tight_layout()
plt.show()

# Analyze the relationship between study time and college marks
plt.figure(figsize=(12, 6))
sns.boxplot(x='daily studing time', y='college mark', data=df)
plt.title('Relationship between Daily Study Time and College Marks')
plt.show()

# Analyze the relationship between stress level and college marks
if stress_columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=stress_column, y='college mark', data=df)
    plt.title(f'Relationship between {stress_column} and College Marks')
    plt.show()
else:
    print(f"Couldn't find a stress level column.")

# Analyze the relationship between financial status and part-time job
plt.figure(figsize=(10, 6))
sns.countplot(x='Financial Status', hue='part-time job', data=df)
plt.title('Relationship between Financial Status and Part-time Job')
plt.show()

# Analyze the correlation between numerical features
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Analyze the relationship between gender and salary expectation
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='salary expectation', data=df)
plt.title('Relationship between Gender and Salary Expectation')
plt.show()

# Analyze the relationship between certification course and salary expectation
plt.figure(figsize=(10, 6))
sns.boxplot(x='Certification Course', y='salary expectation', data=df)
plt.title('Relationship between Certification Course and Salary Expectation')
plt.show()

# Analyze the distribution of stress levels
if stress_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=stress_column, data=df)
    plt.title(f'Distribution of {stress_column}')
    plt.show()
else:
    print(f"Couldn't find a stress level column.")

# Analyze the relationship between department and college marks
plt.figure(figsize=(12, 6))
sns.boxplot(x='Department', y='college mark', data=df)
plt.title('Relationship between Department and College Marks')
plt.xticks(rotation=45)
plt.show()

# Print some interesting insights
print("Interesting Insights:")
print(f"1. Average college mark: {df['college mark'].mean():.2f}")
print(f"2. Most common hobby: {df['hobbies'].mode()[0]}")
print(f"3. Percentage of students who like their degree: {(df['Do you like your degree?'] == 'Yes').mean()*100:.2f}%")
print(f"4. Average salary expectation: {df['salary expectation'].mean():.2f}")
if stress_columns:
    print(f"5. Most common {stress_column}: {df[stress_column].mode()[0]}")
else:
    print("5. Couldn't find a stress level column.")

