import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
df = pd.read_csv("pet_health_data.csv")

# Convert Timestamp to datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Visualization: Plot Heart Rate, Temperature, and Movement
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(df["Timestamp"], df["Heart Rate"], marker="o", color="r", label="Heart Rate (bpm)")
plt.xticks(rotation=45)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df["Timestamp"], df["Temperature"], marker="s", color="g", label="Temperature (°C)")
plt.xticks(rotation=45)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df["Timestamp"], df["Movement"], marker="^", color="b", label="Movement Level")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# Simple Prediction using Linear Regression (Predicting next Heart Rate)
X = np.arange(len(df)).reshape(-1, 1)  # Time steps
y = df["Heart Rate"].values

model = LinearRegression()
model.fit(X, y)

# Predict the next heart rate
next_time_step = np.array([[len(df)]])
predicted_heart_rate = model.predict(next_time_step)[0]

print(f"Predicted Heart Rate for Next Timestamp: {predicted_heart_rate:.2f} bpm")

# Determine Pet Health Status based on predicted heart rate
if predicted_heart_rate < 70:
    health_status = "Low Activity / Possible Fatigue"
    health_distribution = [70, 30]  # Unhealthy, Healthy
    colors = ["red", "green"]
elif 70 <= predicted_heart_rate <= 100:
    health_status = "Healthy & Active"
    health_distribution = [20, 80]  # Unhealthy, Healthy
    colors = ["red", "green"]
else:
    health_status = "High Stress / Overactive"
    health_distribution = [60, 40]  # Unhealthy, Healthy
    colors = ["red", "green"]

# Pie Chart for Pet Health Prediction
plt.figure(figsize=(6, 6))
plt.pie(health_distribution, labels=["Unhealthy", "Healthy"], autopct="%1.1f%%", colors=colors)
plt.title(f"Predicted Pet Health Status: {health_status}")
plt.show()
