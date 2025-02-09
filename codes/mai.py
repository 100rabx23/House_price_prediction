import pandas as pd
import numpy as np
import joblib
import threading
import requests
from flask import Flask, request, jsonify
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk, messagebox

# Train and Save the Model
def train_model():
    california_housing = fetch_california_housing()
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    data['MedHouseVal'] = california_housing.target

    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    joblib.dump(model, 'california_housing_model.pkl')

# Flask API for Predictions
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('california_housing_model.pkl')
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

# Function to Start Flask Server in a Thread
def start_flask():
    app.run(debug=False, port=5000, use_reloader=False)

# Function to Send Prediction Request from Tkinter UI
def get_prediction():
    try:
        features = [float(entry.get()) for entry in entries]
        response = requests.post('http://127.0.0.1:5000/predict', json={'features': features})
        prediction = response.json()['prediction'][0]
        result_label.config(text=f"Predicted Price: ${prediction:.2f}", foreground="blue")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Tkinter UI Setup
root = tk.Tk()
root.title("California Housing Price Prediction")
root.geometry("500x600")
root.configure(bg="#f4f4f4")

# Centering the window
window_width = 500
window_height = 600
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = (screen_width - window_width) // 2
y_cordinate = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# Title Label
title_label = ttk.Label(root, text="🏡 Housing Price Prediction", font=("Arial", 16, "bold"), background="#f4f4f4")
title_label.pack(pady=10)

# Main Frame
frame = ttk.Frame(root, padding=20, relief="raised")
frame.pack(padx=10, pady=10)

# Input Fields
labels = ["Median Income", "House Age", "Average Rooms", "Average Bedrooms",
          "Population", "Average Occupancy", "Latitude", "Longitude"]
entries = []

for label_text in labels:
    label = ttk.Label(frame, text=label_text + ":", font=("Arial", 12))
    label.pack(anchor="w", pady=2)
    entry = ttk.Entry(frame, font=("Arial", 12), width=25)
    entry.pack(pady=2)
    entries.append(entry)

# Predict Button
predict_button = ttk.Button(frame, text="Predict", command=get_prediction, style="TButton")
predict_button.pack(pady=10)

# Prediction Result Label
result_label = ttk.Label(root, text="", font=("Arial", 14, "bold"), background="#f4f4f4")
result_label.pack(pady=10)

# Style
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=5)
style.configure("TLabel", background="#f4f4f4")

# Train Model
train_model()

# Run Flask Server in a Thread
flask_thread = threading.Thread(target=start_flask, daemon=True)
flask_thread.start()

# Run Tkinter App
root.mainloop()
