import pandas as pd
import numpy as np
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# File path to the dataset
file_path = "rainfall_data_2012_2022_.csv"  # Update with your file path

# Function to train the model
def train_model():
    try:
        # Load dataset
        data = pd.read_csv(file_path)

        # Check for missing values
        if data.isnull().sum().any():
            messagebox.showerror("Error", "Dataset contains missing values!")
            return

        # Split data into independent (X) and dependent (y) variables
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train the model
        xgb_model = XGBRegressor(n_estimators=100, random_state=0)
        xgb_model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = xgb_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print("\nR² Score:", r2)

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE (Root Mean Squared Error):", rmse)

        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE (Mean Absolute Error):", mae)

        # Save the model
        joblib.dump(xgb_model, "rainfall_xgb_model.pkl")

        messagebox.showinfo("Success", f"Model trained successfully!\nR² Score: {r2:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Function to predict rainfall using new input
def predict_rainfall():
    try:
        # Load the trained model
        xgb_model = joblib.load("rainfall_xgb_model.pkl")

        # Retrieve inputs from the GUI
        year = int(year_var.get())
        month = int(month_var.get())
        day = int(day_var.get())
        tempavg = float(tempavg_var.get())
        dpavg = float(dpavg_var.get())
        humidity = float(humidity_var.get())
        slpavg = float(slpavg_var.get())
        visibilityavg = float(visibilityavg_var.get())
        windavg = float(windavg_var.get())

        # Prepare the input data
        new_data = [[year, month, day, tempavg, dpavg, humidity, slpavg, visibilityavg, windavg]]

        # Predict rainfall
        prediction = xgb_model.predict(new_data)
        result_var.set(f"Predicted Rainfall: {prediction[0]:.2f} mm")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Initialize the Tkinter window
root = Tk()
root.title("Rainfall Prediction")
root.geometry("400x600")

# Input Fields
Label(root, text="Year").grid(row=0, column=0, padx=10, pady=5)
year_var = StringVar()
Entry(root, textvariable=year_var).grid(row=0, column=1, padx=10, pady=5)

Label(root, text="Month").grid(row=1, column=0, padx=10, pady=5)
month_var = StringVar()
Entry(root, textvariable=month_var).grid(row=1, column=1, padx=10, pady=5)

Label(root, text="Day").grid(row=2, column=0, padx=10, pady=5)
day_var = StringVar()
Entry(root, textvariable=day_var).grid(row=2, column=1, padx=10, pady=5)

Label(root, text="Temperature Avg (°C)").grid(row=3, column=0, padx=10, pady=5)
tempavg_var = StringVar()
Entry(root, textvariable=tempavg_var).grid(row=3, column=1, padx=10, pady=5)

Label(root, text="DP Avg (°C)").grid(row=4, column=0, padx=10, pady=5)
dpavg_var = StringVar()
Entry(root, textvariable=dpavg_var).grid(row=4, column=1, padx=10, pady=5)

Label(root, text="Humidity (%)").grid(row=5, column=0, padx=10, pady=5)
humidity_var = StringVar()
Entry(root, textvariable=humidity_var).grid(row=5, column=1, padx=10, pady=5)

Label(root, text="SLP Avg (hPa)").grid(row=6, column=0, padx=10, pady=5)
slpavg_var = StringVar()
Entry(root, textvariable=slpavg_var).grid(row=6, column=1, padx=10, pady=5)

Label(root, text="Visibility Avg (km)").grid(row=7, column=0, padx=10, pady=5)
visibilityavg_var = StringVar()
Entry(root, textvariable=visibilityavg_var).grid(row=7, column=1, padx=10, pady=5)

Label(root, text="Wind Avg (km/h)").grid(row=8, column=0, padx=10, pady=5)
windavg_var = StringVar()
Entry(root, textvariable=windavg_var).grid(row=8, column=1, padx=10, pady=5)

# Result Field
result_var = StringVar()
Label(root, textvariable=result_var, fg="blue").grid(row=10, columnspan=2, padx=10, pady=10)

# Buttons
Button(root, text="Train Model", command=train_model, bg="green", fg="white").grid(row=9, column=0, padx=10, pady=10)
Button(root, text="Predict", command=predict_rainfall, bg="blue", fg="white").grid(row=9, column=1, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
