import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Convert 'Age' and 'Annual Income (k$)' to numeric type
    df['Age'] = pd.to_numeric(df['Age'])
    df['Annual Income (k$)'] = pd.to_numeric(df['Annual Income (k$)'])

    # Create a target variable based on 'Spending Score'
    df['SpendingCategory'] = pd.cut(df['Spending Score (1-100)'].astype(int),
                                    bins=[0, 33, 66, 100],
                                    labels=['Low', 'Medium', 'High'])

    return df

# Function to prepare features and target
def prepare_features_and_target(df):
    X = df[['Age', 'Annual Income (k$)']]
    y = df['SpendingCategory']
    return X, y

# Function to train and evaluate the model
def train_and_evaluate_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = knn.predict(X_test_scaled)

    # Print the results
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Main function to run the classification
def main():
    file_path = 'shopping_data.csv'  # Assuming the file is in the current directory
    df = load_and_preprocess_data(file_path)
    X, y = prepare_features_and_target(df)
    train_and_evaluate_model(X, y)

# Run the main function
if __name__ == "__main__":
    main()