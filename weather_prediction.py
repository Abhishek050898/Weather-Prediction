
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (example dataset - you should load your weather dataset)
# Ensure the dataset has features like temperature, humidity, etc.
# Replace with actual data or path to your CSV
# Sample structure: columns = ['Max Temp', 'Min Temp', 'Max Humidity', 'Min Humidity', ... , 'Label']
data = pd.read_csv('weather_dataset.csv')

# Feature columns and label column
X = data.drop(columns=['Label'])  # Drop the label column, assuming 'Label' is the target
y = data['Label']  # Label column

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Artificial Neural Network (ANN) model
# Using Multi-layer Perceptron (MLP) for classification
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)

# Train the model
mlp.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = mlp.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print(classification_report(y_test, y_pred))

# Example of prediction for new input (replace with real values)
new_data = np.array([[37, 25, 44, 15, 1009, 1003]])  # Example input
new_data_scaled = scaler.transform(new_data)
prediction = mlp.predict(new_data_scaled)
print(f'Predicted Weather Condition: {prediction[0]}')
