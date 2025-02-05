import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the CSV file
df = pd.read_csv('Traffic.csv')

# Convert 'Date' to string and ensure 'Time' is string
df['Date'] = df['Date'].astype(str)
df['Time'] = df['Time'].astype(str)

# Pad 'Date' with leading zeros if necessary to ensure it's always 2 digits
df['Date'] = df['Date'].str.zfill(2)

# Create a year column (assuming current year, update as needed)
df['Year'] = '2023'

# Combine Year, Date, and Time to create DateTime
df['DateTime'] = pd.to_datetime(df['Year'] + '-' + df['Date'] + ' ' + df['Time'], format='%Y-%d %I:%M:%S %p')
df.set_index('DateTime', inplace=True)

# Convert 'Total' column to numeric, replacing any non-numeric values with NaN
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

# Resample data to hourly frequency and sum the 'Total' column
hourly_data = df['Total'].resample('H').sum()

# Plot the hourly traffic data
plt.figure(figsize=(12, 6))
plt.plot(hourly_data)
plt.title('Hourly Traffic Data')
plt.xlabel('Date and Time')
plt.ylabel('Total Traffic Count')
plt.show()

# Perform seasonal decomposition
try:
    decomposition = seasonal_decompose(hourly_data, model='additive', period=24)
    decomposition.plot()
    plt.tight_layout()
    plt.show()
except ValueError as e:
    print(f"Error in seasonal decomposition: {e}")
    print("Skipping seasonal decomposition plot.")

# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(hourly_data, ax=ax1, lags=48)
plot_pacf(hourly_data, ax=ax2, lags=48)
plt.tight_layout()
plt.show()

# Fit ARIMA model
try:
    model = ARIMA(hourly_data, order=(1, 1, 1))
    results = model.fit()

    # Print model summary
    print(results.summary())

    # Forecast next 24 hours
    forecast = results.forecast(steps=24)

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_data.index[-168:], hourly_data[-168:], label='Observed')
    plt.plot(forecast.index, forecast, color='red', label='Forecast')
    plt.title('Traffic Forecast for Next 24 Hours')
    plt.xlabel('Date and Time')
    plt.ylabel('Total Traffic Count')
    plt.legend()
    plt.show()

    # Print the forecast values
    print("Forecast for the next 24 hours:")
    print(forecast)

except Exception as e:
    print(f"Error in ARIMA modeling: {e}")
    print("Unable to perform forecasting.")

# Print data info for debugging
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nUnique values in 'Total' column:")
print(df['Total'].unique())

print("\nDescriptive statistics of 'Total' column:")
print(df['Total'].describe())

