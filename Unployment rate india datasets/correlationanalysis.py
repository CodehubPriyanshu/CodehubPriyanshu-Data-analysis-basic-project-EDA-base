import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/content/Unemployment.csv')

# Display the first few rows and data info
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

print("\nColumn names:")
print(df.columns)

# Identify numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

# Convert numeric columns to float if they're not already
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Compute correlation matrix
correlation_matrix = df[numeric_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Unemployment Data')
plt.show()

# Print correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Interpret results
print("\nInterpretation:")
for i in range(len(numeric_columns)):
    for j in range(i+1, len(numeric_columns)):
        corr = correlation_matrix.iloc[i, j]
        print(f"Correlation between {numeric_columns[i]} and {numeric_columns[j]}: {corr:.2f}")
        if abs(corr) > 0.5:
            strength = "strong"
        elif abs(corr) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        print(f"This indicates a {strength} {'positive' if corr > 0 else 'negative'} correlation.")
        print()