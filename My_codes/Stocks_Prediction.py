import yfinance as yf

# Download historical stock data
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Show the data
print(data.head())

import pandas as pd

# Calculate additional features (e.g., moving averages)
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Define a function to detect candlestick patterns
def detect_hammer(df):
    conditions = [
        (df['Close'] > df['Open']),
        ((df['High'] - df['Close']) < (df['Close'] - df['Open'])),
        ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open']))
    ]
    return all(conditions)

# Apply the function
data['Hammer'] = data.apply(detect_hammer, axis=1)

# Drop rows with NaN values (due to rolling mean)
data = data.dropna()

from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']])

# Create the target variable (shifted by one day for prediction)
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

# Define features and target
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
target = data['Target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(14,7))
plt.plot(y_test.values, label='True Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
