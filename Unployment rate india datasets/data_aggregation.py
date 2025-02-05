import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('Unemployment.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

print("\nColumn Names:")
print(df.columns)

print("\nFirst few rows of the dataset:")
print(df.head())

# Check if 'Date' column exists, if not, we'll use the first date-like column
date_column = 'Date' if 'Date' in df.columns else df.select_dtypes(include=['object']).columns[0]

print(f"\nUsing '{date_column}' as the date column.")

# Convert date column to datetime
df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y', errors='coerce')

# Convert numeric columns to appropriate data types
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 1. Average Unemployment Rate by Region
if 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
    avg_unemployment_by_region = df.groupby('Region').agg({'Estimated Unemployment Rate (%)': 'mean'}).sort_values('Estimated Unemployment Rate (%)', ascending=False)
    print("\nAverage Unemployment Rate by Region:")
    print(avg_unemployment_by_region)

# 2. Monthly Average Unemployment Rate
if 'Estimated Unemployment Rate (%)' in df.columns:
    df['Month'] = df[date_column].dt.to_period('M')
    monthly_avg_unemployment = df.groupby('Month').agg({'Estimated Unemployment Rate (%)': 'mean'})
    print("\nMonthly Average Unemployment Rate:")
    print(monthly_avg_unemployment)

# 3. Unemployment Rate by Area (Urban vs Rural)
if 'Area' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
    avg_unemployment_by_area = df.groupby('Area').agg({'Estimated Unemployment Rate (%)': 'mean'})
    print("\nAverage Unemployment Rate by Area:")
    print(avg_unemployment_by_area)

# 4. Top 5 Regions with Highest Average Unemployment Rate
if 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
    top_5_unemployment = avg_unemployment_by_region.head()
    print("\nTop 5 Regions with Highest Average Unemployment Rate:")
    print(top_5_unemployment)

# 5. Average Labour Participation Rate by Region
if 'Region' in df.columns and 'Estimated Labour Participation Rate (%)' in df.columns:
    avg_labour_participation = df.groupby('Region').agg({'Estimated Labour Participation Rate (%)': 'mean'}).sort_values('Estimated Labour Participation Rate (%)', ascending=False)
    print("\nAverage Labour Participation Rate by Region:")
    print(avg_labour_participation)

print("\nAnalysis complete. Please check the output and visualizations.")