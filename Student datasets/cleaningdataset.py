import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Student.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Clean up column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Convert height and weight to numeric, replacing any non-numeric values with NaN
df['height(cm)'] = pd.to_numeric(df['height(cm)'], errors='coerce')
df['weight(kg)'] = pd.to_numeric(df['weight(kg)'], errors='coerce')

# Convert marks to numeric, replacing any non-numeric values with NaN
df['10th_mark'] = pd.to_numeric(df['10th_mark'], errors='coerce')
df['12th_mark'] = pd.to_numeric(df['12th_mark'], errors='coerce')
df['college_mark'] = pd.to_numeric(df['college_mark'], errors='coerce')

# Clean up categorical variables
df['certification_course'] = df['certification_course'].str.lower()
df['gender'] = df['gender'].str.lower()
df['department'] = df['department'].str.lower()

# Clean up and categorize 'daily studying time'
df['daily_studing_time'] = df['daily_studing_time'].str.lower().str.replace(' hour', '').str.replace(' - ', '-')

# Clean up 'salary expectation'
df['salary_expectation'] = pd.to_numeric(df['salary_expectation'], errors='coerce')

# Clean up 'willingness to pursue a career based on their degree'
df['willingness_to_pursue_a_career_based_on_their_degree'] = df['willingness_to_pursue_a_career_based_on_their_degree'].str.rstrip('%').astype('float') / 100.0

# Clean up 'social media & video'
df['social_medai_&_video'] = df['social_medai_&_video'].str.lower().str.replace(' minute', '')

# Clean up 'Travelling Time'
df['travelling_time'] = df['travelling_time'].str.lower().str.replace(' hour', '')

# Display the cleaned dataset info
print("\nCleaned Dataset Info:")
print(df.info())

# Display the first few rows of the cleaned dataset
print("\nFirst few rows of the cleaned dataset:")
print(df.head())

# Save the cleaned dataset
df.to_csv('cleaned_student_data.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_student_data.csv'")