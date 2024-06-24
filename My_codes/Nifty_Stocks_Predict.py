import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Download historical stock data for Nifty 50
nifty_data = yf.download('^NSEI', start='2020-01-01', end='2023-01-01')

# Calculate additional features (e.g., moving averages)
nifty_data['MA20'] = nifty_data['Close'].rolling(window=20).mean()
nifty_data['MA50'] = nifty_data['Close'].rolling(window=50).mean()

# Define a function to detect candlestick patterns
def detect_hammer(df):
    conditions = [
        (df['Close'] > df['Open']),
        ((df['High'] - df['Close']) < (df['Close'] - df['Open'])),
        ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open']))
    ]
    return all(conditions)

# Apply the function
nifty_data['Hammer'] = nifty_data.apply(detect_hammer, axis=1)

# Drop rows with NaN values (due to rolling mean)
nifty_data = nifty_data.dropna()

# Normalize the data
scaler = MinMaxScaler()
nifty_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']] = scaler.fit_transform(nifty_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']])

# Create the target variable (shifted by one day for prediction)
nifty_data['Target'] = nifty_data['Close'].shift(-1)
nifty_data = nifty_data.dropna()

# Define features and target
features = nifty_data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50']]
target = nifty_data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

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
plt.figure(figsize=(14,7))
plt.plot(y_test.values, label='True Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
