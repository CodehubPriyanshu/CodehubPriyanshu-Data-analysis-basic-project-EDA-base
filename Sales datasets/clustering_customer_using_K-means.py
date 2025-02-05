import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
print("Loading the data...")
df = pd.read_csv('Sales.csv')
print("Data loaded successfully.")

# Step 2: Data Preprocessing
print("\nPreprocessing the data...")
# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert 'Age' to numeric, replacing any non-numeric values with NaN
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Group by Customer ID and aggregate
customer_data = df.groupby('Customer ID').agg({
    'Total Amount': 'sum',
    'Quantity': 'sum',
    'Age': 'first',  # Assuming age doesn't change
    'Gender': 'first'  # Assuming gender doesn't change
}).reset_index()

# Convert Gender to numeric (0 for Female, 1 for Male)
customer_data['Gender'] = (customer_data['Gender'] == 'Male').astype(int)

# Remove any rows with NaN values
customer_data = customer_data.dropna()

print("Data preprocessing completed.")

# Step 3: Feature Scaling
print("\nScaling the features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_data[['Total Amount', 'Quantity', 'Age']])

# Step 4: K-means Clustering
print("\nPerforming K-means clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_features)

print("Clustering completed.")

# Step 5: Visualizing the results
print("\nVisualizing the results...")
plt.figure(figsize=(12, 8))
sns.scatterplot(data=customer_data, x='Total Amount', y='Quantity', hue='Cluster', palette='viridis')
plt.title('Customer Segments: Total Amount vs Quantity')
plt.xlabel('Total Amount Spent')
plt.ylabel('Total Quantity Purchased')
plt.show()

# Print cluster statistics
print("\nCluster Statistics:")
print(customer_data.groupby('Cluster').agg({
    'Total Amount': 'mean',
    'Quantity': 'mean',
    'Age': 'mean'
}).round(2))

print("\nClustering analysis completed.")