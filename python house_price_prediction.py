import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load the dataset
data = pd.read_csv('Housing.csv')

# Data exploration
print(data.head())
print(data.info())
print(data.describe())

# Handle missing values by filling with the median
data = data.fillna(data.median())

# Convert categorical data to numerical using one-hot encoding
data = pd.get_dummies(data)

# Define features and target variable
# Adjust features as necessary based on dataset columns
features = data.columns.drop('price')  # Assuming 'price' is the target column
X = data[features]
y = data['price']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer
    Dense(32, activation='relu'),                             # Hidden layer
    Dense(1)                                                  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction for a new house
# Assuming new_house has the same structure as the features
new_house = [[...]]  # Example feature values - fill in with appropriate values
new_house_scaled = scaler.transform(new_house)
price_prediction = model.predict(new_house_scaled)
print(f'Predicted House Price: {price_prediction[0][0]}')

# Save the model and scaler for future use
model.save('house_price_model.h5')
joblib.dump(scaler, 'scaler.pkl')
