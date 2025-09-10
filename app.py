import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = joblib.load('linear_regression_model (2).joblib')

# Load the original data to fit the scaler and get column order
# In a real application, you would save and load the scaler and column list
# instead of reloading and reprocessing the data.
try:
    df_original = pd.read_csv("FuelConsumption (1) (1).csv")
except FileNotFoundError:
    st.error("Original data file not found. Please ensure 'FuelConsumption (1).csv' is in the correct directory.")
    st.stop()

# Strip whitespace from column names
df_original.columns = df_original.columns.str.strip()

# Replicate preprocessing steps from the notebook
# Handle duplicate values
df_original.drop_duplicates(inplace=True)

# Feature engineering: Create FUEL_EFFICIENCY
df_original['FUEL_EFFICIENCY'] = 1 / df_original['FUEL CONSUMPTION']

# Identify numerical and categorical columns
# Ensure 'COEMISSIONS' is correctly referenced without the trailing space
numerical_cols = ['Year', 'ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'COEMISSIONS', 'FUEL_EFFICIENCY']
categorical_cols = df_original.select_dtypes(include='object').columns

# Apply one-hot encoding
df_processed = pd.get_dummies(df_original, columns=categorical_cols, drop_first=True)

# Separate features and target *before* scaling
X_processed = df_processed.drop('COEMISSIONS', axis=1)
y_processed = df_processed['COEMISSIONS']

# Apply scaling to numerical columns *after* separating features and target
scaler = MinMaxScaler()
# Fit on the processed training data features
# Need to ensure numerical_cols only contains columns present in X_processed
numerical_cols_in_X = [col for col in numerical_cols if col in X_processed.columns]
X_processed[numerical_cols_in_X] = scaler.fit_transform(X_processed[numerical_cols_in_X])


st.title('CO2 Emission Prediction')

st.write("Enter the vehicle features to predict CO2 emissions:")

# Add input widgets for features - ensure these match the original data's relevant columns
year = st.slider('Year', int(df_original['Year'].min()), int(df_original['Year'].max()), int(df_original['Year'].mean()))
make = st.selectbox('Make', df_original['MAKE'].unique())
# Filter models based on selected make
model_name = st.selectbox('Model', df_original[df_original['MAKE'] == make]['MODEL'].unique())
vehicle_class = st.selectbox('Vehicle Class', df_original['VEHICLE CLASS'].unique())
engine_size = st.number_input('Engine Size', min_value=float(df_original['ENGINE SIZE'].min()), max_value=float(df_original['ENGINE SIZE'].max()), value=float(df_original['ENGINE SIZE'].mean()), step=0.1)
cylinders = st.slider('Cylinders', int(df_original['CYLINDERS'].min()), int(df_original['CYLINDERS'].max()), int(df_original['CYLINDERS'].mean()))
transmission = st.selectbox('Transmission', df_original['TRANSMISSION'].unique())
fuel = st.selectbox('Fuel Type', df_original['FUEL'].unique())
fuel_consumption = st.number_input('Fuel Consumption (L/100 km)', min_value=float(df_original['FUEL CONSUMPTION'].min()), max_value=float(df_original['FUEL CONSUMPTION'].max()), value=float(df_original['FUEL CONSUMPTION'].mean()), step=0.1)


# Create a button to predict
if st.button('Predict CO2 Emissions'):
    # Prepare the user input into a DataFrame matching the training data structure
    user_input = pd.DataFrame([[year, make, model_name, vehicle_class, engine_size, cylinders, transmission, fuel, fuel_consumption]],
                              columns=['Year', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS', 'TRANSMISSION', 'FUEL', 'FUEL CONSUMPTION'])

    # Replicate feature engineering for user input
    user_input['FUEL_EFFICIENCY'] = 1 / user_input['FUEL CONSUMPTION']

    # Apply one-hot encoding to user input - crucial to use the same dummy columns as training
    user_input_processed = pd.get_dummies(user_input, columns=categorical_cols, drop_first=True)

    # Align columns - add missing dummy columns and ensure the order is the same as X_processed
    # This is important because the model expects features in a specific order.
    # Add missing columns from training set (fill with 0)
    missing_cols = set(X_processed.columns) - set(user_input_processed.columns)
    for c in missing_cols:
        user_input_processed[c] = 0
    # Ensure the order of columns is the same as the training set
    user_input_processed = user_input_processed[X_processed.columns]

    # Apply scaling to numerical columns in user input
    # Note: We use the SAME scaler fitted on the training data
    # Need to ensure numerical_cols only contains columns present in user_input_processed
    numerical_cols_in_user_input = [col for col in numerical_cols if col in user_input_processed.columns]
    user_input_processed[numerical_cols_in_user_input] = scaler.transform(user_input_processed[numerical_cols_in_user_input])


    # Make prediction
    predicted_co2 = model.predict(user_input_processed)

    # Display the prediction
    # Note: The prediction is on the scaled target variable.
    # If you want the original scale, you would need to inverse transform the prediction.
    # However, the prompt asked to predict the target variable which was scaled during training.
    st.success(f'Predicted Scaled CO2 Emissions: {predicted_co2[0]:.4f}')
