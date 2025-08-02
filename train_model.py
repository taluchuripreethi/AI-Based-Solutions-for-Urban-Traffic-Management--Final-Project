import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(file_path):
    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Ensure the data has no missing values and is ready for training
    data = data.dropna()

    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Save the trained model
    model.save_model('models/traffic_volume_model.json')

    return "Model trained successfully", mse
