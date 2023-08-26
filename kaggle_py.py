import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the CSV file
file_path = "C:/Users/39388/Downloads/playground-series-s3e19/train.csv"
train_data = pd.read_csv(file_path)
train_data['date'] = pd.to_datetime(train_data['date'])

# Display the head, summary, and structure of the data
print(train_data.head())
print(train_data.describe())
print(train_data.info())

# Check for missing values
missing_values = train_data.isnull().sum()
print(missing_values)

# Boxplot by country and store
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.boxplot(train_data['num_sold'], vert=False)
plt.xlabel('Number of Units Sold')
plt.title('Boxplot by Country')
plt.subplot(1, 2, 2)
plt.boxplot(train_data['num_sold'], vert=False)
plt.xlabel('Number of Units Sold')
plt.title('Boxplot by Store')
plt.tight_layout()
plt.show()

# Grouped boxplot using seaborn
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(x='country', y='num_sold', hue='product', data=train_data)
plt.title('Boxplot of Number of Units Sold by Country, Store, and Product')
plt.xlabel('Country')
plt.ylabel('Number of Units Sold')
plt.xticks(rotation=45, ha='right')
plt.show()

# Grouped boxplot using pandas
grouped_data_product = train_data.groupby('product')
summary_stats_product = grouped_data_product['num_sold'].agg(['mean', 'median', 'max', 'min'])
print(summary_stats_product)

grouped_data_store = train_data.groupby('store')
summary_stats_store = grouped_data_store['num_sold'].agg(['mean', 'median', 'max', 'min'])
print(summary_stats_store)

grouped_data_country = train_data.groupby('country')
summary_stats_country = grouped_data_country['num_sold'].agg(['mean', 'median', 'max', 'min'])
print(summary_stats_country)

# Combine country, product, and store columns into a single column "combination"
train_data_combined = train_data.copy()
train_data_combined['combination'] = train_data_combined['country'] + '_' + train_data_combined['product'] + '_' + train_data_combined['store']

# Pivot_wider to get separate variables for each unique combination
train_data_wide = train_data_combined.pivot_table(index='date', columns='combination', values='num_sold', aggfunc='sum')
print(train_data_wide)

# Calculate the sum of num_sold for each date
train_data_combined = train_data_wide.sum(axis=1).reset_index(name='sum_num')

# Remove duplicated rows
train_data_combined = train_data_combined.drop_duplicates()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(train_data_combined['date'], train_data_combined['sum_num'])
plt.title('Time Series of Total Number of Units Sold')
plt.xlabel('Date')
plt.ylabel('Total Number of Units Sold')
plt.xticks(rotation=45)
plt.show()

# Create the time series data and fit ARIMA model
ts_data = pd.Series(train_data_combined['sum_num'].values, index=train_data_combined['date'])
arima_model = ARIMA(ts_data, order=(1, 1, 1))
arima_result = arima_model.fit()
print(arima_result.summary())

# Forecast future values with ARIMA
future_arima = arima_result.forecast(steps=365)
plt.figure(figsize=(12, 6))
plt.plot(ts_data, label='Observed')
plt.plot(future_arima, label='Forecast')
plt.title('ARIMA Forecast for the Next 365 Days')
plt.xlabel('Date')
plt.ylabel('Total Number of Units Sold')
plt.xticks(rotation=45)
plt.legend()
plt.show()
# Convert date to datetime format
train_data_combined['date'] = pd.to_datetime(train_data_combined['date'])

# List of columns to forecast
columns_to_forecast = train_data_combined.columns[1:]

# Function to perform ARIMA model fitting and forecasting
def forecast_arima(ts_data, col):
    # Create the time series object
    ts = pd.Series(ts_data, index=train_data_combined['date'])
    
    # Model ARIMA
    model = sm.tsa.ARIMA(ts, order=(1, 0, 1))
    arima_model = model.fit(disp=0)
    
    # Print summary
    print(f"Summary for variable: {col}")
    print(arima_model.summary())
    
    # Make forecasts
    forecasts = arima_model.forecast(steps=30) # Change 30 with the number of days to forecast
    
    # Plot forecasts
    plt.figure(figsize=(10, 5))
    plt.plot(ts, label='Actual Data')
    plt.plot(pd.date_range(ts.index[-1], periods=31, closed='right'), forecasts, label='Forecast')
    plt.title(f"Forecast for variable: {col}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
# Perform ARIMA model fitting and forecasting for each column
for col in columns_to_forecast:
    forecast_arima(train_data_combined[col], col)