# Insurance Cost Predictor - Advanced Implementation

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import time
import datetime
import uuid
from PIL import Image
import os
import json
import base64
from io import BytesIO
import sys

# ===========================================================================
# CONFIG AND SETUP
# ===========================================================================

# Set page configuration
st.set_page_config(
    page_title="Health Insurance Premium Estimator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme colors for consistent UI
THEME = {
    "primary": "#3498db",   # Blue
    "secondary": "#2ecc71", # Green
    "warning": "#f39c12",   # Orange
    "danger": "#e74c3c",    # Red
    "info": "#3498db",      # Blue
    "light": "#ecf0f1",     # Light Gray
    "dark": "#2c3e50",      # Dark Blue
    "background": "#f8f9fa", # Off-White
    "text": "#2c3e50"       # Dark Blue
}

# Custom CSS for advanced styling
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
    }
    .main-header {
        font-size: 2.75rem;
        color: #3498db;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 1.5rem;
        color: #2c3e50;  /* Changed from light gray to dark blue for better visibility */
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 400;  /* Increased from 300 for better readability */
    }
    
    /* Cards and Containers */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.1), 0 3px 6px rgba(0,0,0,0.1);
    }
    .card-header {
        border-bottom: 1px solid #eee;
        margin-bottom: 15px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #3498db;
    }
    .prediction-result {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 30px;
        border-radius: 10px;
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #2c3e50;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05), 0 6px 6px rgba(0,0,0,0.1);
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1);
        padding: 20px;
        margin: 15px 0;
    }
    
    /* Form Elements */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        height: 3em;
        font-weight: 600;
        background-color: #3498db;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
    }
    div[data-testid="stSlider"] > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stRadio"] > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Helper Classes */
    .tooltip {
        color: #2c3e50;  /* Changed from light gray to dark blue for better visibility */
        font-size: 0.9rem;
        padding: 10px;
        border-left: 3px solid #3498db;
        background-color: #f8f9fa;
    }
    .sidebar-info {
        font-size: 0.9rem;
        color: #2c3e50;  /* Changed from light gray to dark blue for better visibility */
    }
    
    /* Custom Widgets */
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3498db;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #2c3e50;  /* Changed from light gray to dark blue for better visibility */
    }
    
    /* Annotations and Documentation */
    .documentation-note {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #3498db;
        color: #2c3e50;
    }
    
    /* Regional Analysis Specific */
    .region-stat-card {
        background-color: white;  /* Changed from brown for better visibility */
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #3498db;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .animate-fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a1a1a1;
    }
    
    /* Responsive Adjustments */
    @media screen and (max-width: 1200px) {
        .main-header {
            font-size: 2.2rem;
        }
        .subheader {
            font-size: 1.2rem;
        }
    }
    
    /* Error Handling */
    .error-message {
        background-color: #fff3f3;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #c0392b;
    }
</style>
""", unsafe_allow_html=True)

# Add JavaScript to detect browser information for better diagnostics
st.markdown("""
<script>
    // Browser detection script
    document.addEventListener('DOMContentLoaded', function() {
        const getBrowserInfo = () => {
            const ua = navigator.userAgent;
            let browserName = "Unknown";
            let browserVersion = "Unknown";
            let osName = "Unknown";
            
            // Detect browser
            if (ua.indexOf("Chrome") > -1) {
                browserName = "Chrome";
                browserVersion = ua.match(/Chrome\/([0-9.]+)/)[1];
            } else if (ua.indexOf("Firefox") > -1) {
                browserName = "Firefox";
                browserVersion = ua.match(/Firefox\/([0-9.]+)/)[1];
            } else if (ua.indexOf("Safari") > -1) {
                browserName = "Safari";
                browserVersion = ua.match(/Version\/([0-9.]+)/)[1];
            } else if (ua.indexOf("MSIE") > -1 || ua.indexOf("Trident") > -1) {
                browserName = "Internet Explorer";
                browserVersion = ua.match(/(?:MSIE |rv:)([0-9.]+)/)[1];
            } else if (ua.indexOf("Edge") > -1) {
                browserName = "Edge";
                browserVersion = ua.match(/Edge\/([0-9.]+)/)[1];
            }
            
            // Detect OS
            if (ua.indexOf("Windows") > -1) osName = "Windows";
            else if (ua.indexOf("Mac") > -1) osName = "macOS";
            else if (ua.indexOf("Linux") > -1) osName = "Linux";
            else if (ua.indexOf("Android") > -1) osName = "Android";
            else if (ua.indexOf("iOS") > -1) osName = "iOS";
            
            return {
                browser: browserName,
                version: browserVersion,
                os: osName,
                userAgent: ua,
                screenSize: `${window.screen.width}x${window.screen.height}`,
                windowSize: `${window.innerWidth}x${window.innerHeight}`,
                devicePixelRatio: window.devicePixelRatio
            };
        };
        
        // Send browser info to Streamlit via session state
        const browserInfo = getBrowserInfo();
        if (window.parent.stApp) {
            window.parent.stApp.setComponentValue('browser_info', JSON.stringify(browserInfo));
        }
    });
</script>
""", unsafe_allow_html=True)

# Enhanced styling for the entire app
st.markdown("""
<style>
    /* Additional UI enhancements */
    .enhanced-card {
        background: linear-gradient(to right, #ffffff, #f8f9fa);
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        padding: 25px;
        margin-bottom: 30px;
        border-top: 5px solid #3498db;
        transition: all 0.3s ease;
    }
    .enhanced-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    /* Improved button styling */
    .stButton button:hover {
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    
    /* Enhanced Table Styling */
    div[data-testid="stTable"] table {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    div[data-testid="stTable"] thead tr th {
        background-color: #f0f7ff !important;
        color: #2c3e50;
        text-transform: uppercase;
        font-size: 0.8em;
        letter-spacing: 0.5px;
        padding: 12px 15px;
        border-bottom: 2px solid #e1e8ef;
    }
    div[data-testid="stTable"] tbody tr:nth-child(even) {
        background-color: #f8fafc;
    }
    div[data-testid="stTable"] tbody tr:hover {
        background-color: #f0f7ff;
    }
    
    /* Custom Tooltip */
    .custom-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .custom-tooltip .tooltip-text {
        visibility: hidden;
        width: 200px;
        background-color: #2c3e50;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .custom-tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .loading-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
    
    /* Error recovery message */
    .recovery-message {
        background-color: #fff8f0;
        border-left: 5px solid #f39c12;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #8a6d3b;
        font-size: 0.9rem;
    }
    
    /* Success message with animation */
    .success-message {
        background-color: #f0fff4;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #27ae60;
        animation: fadeIn 0.5s;
    }
    
    /* Data insight box styling */
    .insight-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        margin: 20px 0;
        border-left: 4px solid #3498db;
    }
    .insight-box h4 {
        margin-top: 0;
        color: #3498db;
        font-weight: 600;
    }
    .insight-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .insight-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    .insight-change {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .positive-change { color: #2ecc71; }
    .negative-change { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# Create necessary directories
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("exports"):
    os.makedirs("exports")
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up more robust error logging
def log_error(error, context="general"):
    """
    Logs errors to a file for better tracking and debugging
    
    Parameters:
    -----------
    error : Exception
        The exception to log
    context : str
        Description of where the error occurred
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_id = str(uuid.uuid4())[:8]
        
        # Add to session state error log
        st.session_state['error_log'].append({
            "id": error_id,
            "timestamp": timestamp,
            "error": str(error),
            "context": context,
            "traceback": str(getattr(error, "__traceback__", "No traceback"))
        })
        
        # Create a more detailed log for file output
        error_detail = {
            "id": error_id,
            "timestamp": timestamp,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "user_inputs": getattr(st.session_state, 'last_inputs', {}),
            "browser_info": st.session_state.get('browser_info', 'Unknown'),
            "app_version": st.session_state.get('app_version', 'Unknown')
        }
        
        # Write to log file
        log_file = f"logs/error_log_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except:
                logs = []
        else:
            logs = []
        
        logs.append(error_detail)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)
            
        return error_id
    except Exception as e:
        # Fallback if error logging fails
        print(f"Error logging failed: {str(e)}")
        return None

# ===========================================================================
# SESSION STATE AND DATA MANAGEMENT
# ===========================================================================

# Initialize session state
def init_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None
    if 'show_welcome' not in st.session_state:
        st.session_state['show_welcome'] = True
    if 'comparison_mode' not in st.session_state:
        st.session_state['comparison_mode'] = False
    if 'comparison_data' not in st.session_state:
        st.session_state['comparison_data'] = []
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'light'
    if 'error_log' not in st.session_state:
        st.session_state['error_log'] = []
    if 'browser_info' not in st.session_state:
        # Default browser info - will be updated by JavaScript
        st.session_state['browser_info'] = 'Unknown'
    if 'app_version' not in st.session_state:
        # Track app version for troubleshooting
        st.session_state['app_version'] = '2.5.0'

init_session_state()

# Function to save prediction history
def save_prediction(user_inputs, prediction, model_version="v1"):
    """
    Saves prediction to session state and persistent storage
    
    Parameters:
    -----------
    user_inputs : dict
        Dictionary containing all user input parameters
    prediction : float
        Predicted insurance cost
    model_version : str
        Version of the model used for prediction
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prediction_id = str(uuid.uuid4())[:8]
        
        history_item = {
            "id": prediction_id,
            "timestamp": timestamp,
            "inputs": user_inputs,
            "prediction": prediction,
            "model_version": model_version
        }
        
        st.session_state['history'].append(history_item)
        st.session_state['last_prediction'] = history_item
        
        # Save to file for persistence
        history_file = "data/prediction_history.json"
        history_data = []
        
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history_data = json.load(f)
        
        history_data.append(history_item)
        
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=4)
            
        return True
    except Exception as e:
        log_error(e, context="save_prediction")
        return False

# ===========================================================================
# MODEL LOADING AND PREPROCESSING
# ===========================================================================

# Load saved model with caching
@st.cache_resource
def load_model():
    """
    Loads the prediction model with caching for better performance
    
    Returns:
    --------
    model : object
        Trained model for insurance premium prediction
    """
    try:
        return joblib.load('insurance_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'insurance_model.pkl' exists.")
        # Return a dummy model that always predicts the same value if real model is missing
        class DummyModel:
            def predict(self, X):
                return np.array([12000])
        return DummyModel()

model = load_model()

# Create preprocessing function that translates user inputs to model inputs
def preprocess_input(age, sex, bmi, children, smoker, region):
    """
    Convert UI inputs to model-compatible numeric input
    
    Parameters:
    -----------
    age : int
        Age of the beneficiary
    sex : str
        Gender of the beneficiary ('male' or 'female')
    bmi : float
        Body Mass Index
    children : int
        Number of dependents
    smoker : str
        Smoking status ('yes' or 'no')
    region : str
        Geographic region
        
    Returns:
    --------
    pd.DataFrame
        Formatted input dataframe for the model
    """
    try:
        # Create a pandas DataFrame with proper categorical dtypes
        input_df = pd.DataFrame({
            'age': [float(age)],
            'sex': [str(sex)],
            'bmi': [float(bmi)],
            'children': [int(children)],
            'smoker': [str(smoker)],
            'region': [str(region)]
        })
        
        # Explicitly set categorical columns as category dtype
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            input_df[col] = input_df[col].astype('category')
            
        return input_df
        
    except Exception as e:
        log_error(e, context="preprocess_input")
        # Return a safe fallback with correct data types
        return pd.DataFrame({
            'age': [30.0],
            'sex': ['female'],
            'bmi': [25.0],
            'children': [0],
            'smoker': ['no'],
            'region': ['northeast']
        })

# Load sample data for visualizations
@st.cache_data
def load_sample_data():
    """
    Loads or generates sample insurance data for visualizations
    
    Returns:
    --------
    pd.DataFrame
        Dataset containing insurance information
    """
    try:
        # Try to load existing sample data if available
        if os.path.exists("data/sample_insurance_data.csv"):
            return pd.read_csv("data/sample_insurance_data.csv")
        else:
            # Create synthetic data similar to insurance data
            np.random.seed(42)
            n = 1000
            
            ages = np.random.randint(18, 65, n)
            sexes = np.random.choice(['male', 'female'], n)
            bmis = np.random.normal(26, 4, n)
            children = np.random.choice(range(5), n, p=[0.3, 0.3, 0.2, 0.15, 0.05])
            smokers = np.random.choice(['yes', 'no'], n, p=[0.2, 0.8])
            regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n)
            
            # Generate charges based on these features (simplified model)
            charges = 5000 + 100 * ages + 300 * (sexes == 'male') + 200 * bmis + 500 * children + 20000 * (smokers == 'yes')
            charges = charges + np.random.normal(0, 4000, n)  # Add some noise
            
            df = pd.DataFrame({
                'age': ages,
                'sex': sexes,
                'bmi': bmis,
                'children': children,
                'smoker': smokers,
                'region': regions,
                'charges': charges
            })
            
            # Save for future use
            df.to_csv("data/sample_insurance_data.csv", index=False)
            return df
    except Exception as e:
        log_error(e, context="load_sample_data")
        # Return minimal fallback data
        return pd.DataFrame({
            'age': [30, 40],
            'bmi': [25, 30],
            'charges': [5000, 10000],
            'smoker': ['no', 'yes'],
            'region': ['northeast', 'southwest'],
            'sex': ['male', 'female'],
            'children': [0, 2]
        })

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

# Function to get BMI category
def get_bmi_category(bmi):
    """
    Categorizes BMI according to standard medical classifications
    
    Parameters:
    -----------
    bmi : float
        Body Mass Index value
        
    Returns:
    --------
    str
        BMI category description
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    elif bmi < 35:
        return "Obese Class I"
    elif bmi < 40:
        return "Obese Class II"
    else:
        return "Obese Class III"

# Function to calculate and render BMI
def calculate_bmi(weight, height):
    """
    Calculates BMI from weight and height
    
    Parameters:
    -----------
    weight : float
        Weight in kilograms
    height : float
        Height in centimeters
        
    Returns:
    --------
    float
        Calculated BMI rounded to 1 decimal place
    """
    if height == 0:
        return 0
    bmi = weight / ((height/100) ** 2)
    return round(bmi, 1)

# Function to calculate premium factors
def calculate_premium_factors(age, sex, bmi, children, smoker, region):
    """
    Calculates the impact of different factors on insurance premium
    
    Parameters:
    -----------
    age : int
        Age of the beneficiary
    sex : str
        Gender of the beneficiary
    bmi : float
        Body Mass Index
    children : int
        Number of dependents
    smoker : str
        Smoking status
    region : str
        Geographic region
        
    Returns:
    --------
    dict
        Dictionary of factors with their weights and descriptions
    """
    factors = {}
    
    # Age factor (higher for older ages)
    if age < 30:
        factors["Age"] = {"weight": 0.15, "description": "Young Adult"}
    elif age < 45:
        factors["Age"] = {"weight": 0.25, "description": "Middle-Aged Adult"}
    elif age < 60:
        factors["Age"] = {"weight": 0.35, "description": "Senior Adult"}
    else:
        factors["Age"] = {"weight": 0.45, "description": "Senior Citizen"}
        
    # BMI factor
    if bmi < 18.5:
        factors["BMI"] = {"weight": 0.20, "description": "Underweight"}
    elif bmi < 25:
        factors["BMI"] = {"weight": 0.05, "description": "Normal Weight"}
    elif bmi < 30:
        factors["BMI"] = {"weight": 0.15, "description": "Overweight"}
    elif bmi < 35:
        factors["BMI"] = {"weight": 0.30, "description": "Obese Class I"}
    else:
        factors["BMI"] = {"weight": 0.40, "description": "Obese Class II/III"}
    
    # Smoking factor (highest impact)
    if smoker == "yes":
        factors["Smoking"] = {"weight": 0.70, "description": "Smoker"}
    else:
        factors["Smoking"] = {"weight": 0.05, "description": "Non-smoker"}
    
    # Children factor
    if children == 0:
        factors["Dependents"] = {"weight": 0.05, "description": "No dependents"}
    elif children <= 2:
        factors["Dependents"] = {"weight": 0.15, "description": f"{children} dependents"}
    else:
        factors["Dependents"] = {"weight": 0.25, "description": f"Multiple dependents ({children})"}
    
    # Gender factor (small impact)
    if sex == "male":
        factors["Sex"] = {"weight": 0.10, "description": "Male"}
    else:
        factors["Sex"] = {"weight": 0.08, "description": "Female"}
    
    # Region factor (smallest impact)
    region_weights = {
        "northeast": 0.12,
        "northwest": 0.08,
        "southeast": 0.10,
        "southwest": 0.07
    }
    factors["Region"] = {"weight": region_weights.get(region, 0.10), 
                         "description": region.capitalize()}
    
    return factors

# Generate downloadable CSV from dataframe
def get_csv_download_link(df, filename="data.csv", text="Download CSV"):
    """
    Creates a download link for a DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to be downloaded
    filename : str
        Name of the file to be downloaded
    text : str
        Text to display on the download button
        
    Returns:
    --------
    str
        HTML code for download link
    """
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display:inline-block; padding:8px 15px; background-color:#3498db; color:black; text-decoration:none; border-radius:5px; font-weight:600;">{text}</a>'
        return href
    except Exception as e:
        log_error(e, context="get_csv_download_link")
        return f'<p style="color: red;">Error generating download link: {str(e)}</p>'

# Create a radar chart for risk factors
def create_risk_radar_chart(factors):
    """
    Creates a radar chart visualization for risk factors
    
    Parameters:
    -----------
    factors : dict
        Dictionary of risk factors with weights
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Radar chart figure
    """
    try:
        categories = list(factors.keys())
        values = [factor["weight"] for factor in factors.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Factors',
            fillcolor='rgba(52, 152, 219, 0.5)',
            line=dict(color=THEME["primary"])
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 0.8]
                )
            ),
            showlegend=False,
            height=300,
            margin=dict(t=30, b=10, l=40, r=40),
        )
        
        return fig
    except Exception as e:
        log_error(e, context="create_risk_radar_chart")
        # Return an empty figure
        return go.Figure()

# Function to safely convert currency strings to float
def currency_to_float(value):
    """
    Converts currency string (e.g. '$1,234.56') to float
    
    Parameters:
    -----------
    value : str or float
        The value to convert
        
    Returns:
    --------
    float
        The numeric value
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    try:
        # Remove currency symbols, commas, and other non-numeric characters
        # except for decimal point
        if isinstance(value, str):
            cleaned_value = value.replace('$', '').replace(',', '').replace(' ', '')
            return float(cleaned_value)
        return float(value)
    except (ValueError, TypeError):
        # Return 0 as a safe fallback
        return 0.0

# ===========================================================================
# NAVIGATION AND UI COMPONENTS
# ===========================================================================

# Main sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/health-insurance.png", width=90)
    st.title("Health Insurance Estimator")
    
    selected = option_menu(
        menu_title=None,
        options=["Predict", "Visualize", "History", "Compare", "About"],
        icons=["calculator-fill", "graph-up", "clock-history", "arrow-left-right", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": THEME["primary"], "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": THEME["primary"], "color": "white"},
        }
    )
    
    st.markdown("---")
    
    # Theme toggle 
    theme_col1, theme_col2 = st.columns([3, 1])
    with theme_col1:
        st.markdown("##### App Settings")
    with theme_col2:
        if st.button("🌙" if st.session_state['theme'] == 'light' else "☀️"):
            st.session_state['theme'] = 'dark' if st.session_state['theme'] == 'light' else 'light'
            st.experimental_rerun()
    
    # Add some basic info about the app
    st.markdown("""
    <div class='sidebar-info'>
        <h4>About this app</h4>
        <p>This application uses machine learning to predict health insurance premium costs based on personal factors like age, BMI, smoking status, and more.</p>
        <p>All calculations are performed locally, and your data isn't shared.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("© 2025 Health Insurance Estimator")

# ===========================================================================
# PREDICT PAGE
# ===========================================================================

if selected == "Predict":
    # Welcome toast (only shown once per session)
    if st.session_state['show_welcome']:
        st.toast("👋 Welcome to the Insurance Estimator App!", icon="👋")
        st.session_state['show_welcome'] = False
    
    st.markdown("<h1 class='main-header'>Insurance Premium Estimator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Predict your health insurance costs based on personal factors</p>", unsafe_allow_html=True)
    
    # Create columns for form layout
    col1, gap, col2 = st.columns([5, 0.5, 5])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>Personal Information</div>", unsafe_allow_html=True)
        
        age = st.slider("Age", min_value=18, max_value=100, value=30, 
                       help="Age of the primary beneficiary")
        
        sex = st.radio("Sex", options=["male", "female"], horizontal=True,
                      help="Biological sex of the primary beneficiary")
        
        # Improved BMI calculator section with better visibility
        bmi_tabs = st.tabs(["BMI Value", "Calculate BMI"])
        
        with bmi_tabs[0]:
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, 
                                 help="Body Mass Index, weight (kg) / (height (m))²")
            
            # Enhanced BMI display with better visibility
            bmi_category = get_bmi_category(bmi)
            
            # Custom styling for BMI category badge based on the category
            category_colors = {
                "Underweight": {"bg": "#FFF3CD", "border": "#FFC107", "text": "#856404"},
                "Normal": {"bg": "#D4EDDA", "border": "#28A745", "text": "#155724"},
                "Overweight": {"bg": "#FFE0B2", "border": "#FF9800", "text": "#7E4D0E"},
                "Obese Class I": {"bg": "#FFCCBC", "border": "#FF5722", "text": "#8B2500"},
                "Obese Class II": {"bg": "#FFCDD2", "border": "#F44336", "text": "#7A1C12"},
                "Obese Class III": {"bg": "#EF9A9A", "border": "#D32F2F", "text": "#7A1C12"}
            }
            
            category_style = category_colors.get(bmi_category, {"bg": "#E0E0E0", "border": "#9E9E9E", "text": "#424242"})
            
            st.markdown(f"""
            <div style="margin-top: 15px; margin-bottom: 20px;">
                <p style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.1rem; font-weight: 500; margin-bottom: 8px;">BMI Category:</p>
                <div style="display: inline-block; padding: 8px 16px; background-color: {category_style['bg']}; 
                            border-left: 5px solid {category_style['border']}; border-radius: 4px; 
                            font-family: 'Segoe UI', Arial, sans-serif; font-weight: 600; color: {category_style['text']}; 
                            font-size: 1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    {bmi_category}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add BMI interpretation with enhanced styling
            if bmi_category == "Normal":
                st.markdown("""
                <div style="background-color: #D4EDDA; border-radius: 8px; padding: 15px; margin-top: 10px; border-left: 5px solid #28A745;">
                    <p style="margin: 0; color: #155724; font-family: 'Segoe UI', Arial, sans-serif; font-size: 1rem;">
                        <span style="font-size: 1.2rem; margin-right: 8px;">✅</span> 
                        Your BMI is within the healthy range. Maintaining a healthy weight can help reduce the risk of chronic diseases and keep your insurance premiums lower.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif bmi_category == "Underweight":
                st.markdown("""
                <div style="background-color: #FFF3CD; border-radius: 8px; padding: 15px; margin-top: 10px; border-left: 5px solid #FFC107;">
                    <p style="margin: 0; color: #856404; font-family: 'Segoe UI', Arial, sans-serif; font-size: 1rem;">
                        <span style="font-size: 1.2rem; margin-right: 8px;">⚠️</span> 
                        Your BMI indicates you're underweight. This might increase certain health risks and potentially affect your insurance assessment.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif bmi_category == "Overweight":
                st.markdown("""
                <div style="background-color: #FFE0B2; border-radius: 8px; padding: 15px; margin-top: 10px; border-left: 5px solid #FF9800;">
                    <p style="margin: 0; color: #7E4D0E; font-family: 'Segoe UI', Arial, sans-serif; font-size: 1rem;">
                        <span style="font-size: 1.2rem; margin-right: 8px;">⚠️</span> 
                        Your BMI indicates you're overweight. This may increase your insurance costs by approximately 10-20%. A BMI below 25 could help reduce your premium.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #FFCDD2; border-radius: 8px; padding: 15px; margin-top: 10px; border-left: 5px solid #F44336;">
                    <p style="margin: 0; color: #7A1C12; font-family: 'Segoe UI', Arial, sans-serif; font-size: 1rem;">
                        <span style="font-size: 1.2rem; margin-right: 8px;">❗</span> 
                        Your BMI indicates obesity, which typically increases insurance costs by 20-50%. Lowering your BMI below 30 could significantly reduce your premium.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with bmi_tabs[1]:
            st.markdown("""
            <div style="background-color: #E8F4F8; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #3498db; font-family: 'Segoe UI', Arial, sans-serif;">
                <h4 style="margin-top: 0; color: #2c3e50; font-weight: 600;">BMI Calculator</h4>
                <p style="color: #34495e; font-size: 0.9rem; margin-bottom: 5px;">Enter your height and weight to calculate your BMI</p>
            </div>
            """, unsafe_allow_html=True)
            
            bmi_col1, bmi_col2 = st.columns(2)
            with bmi_col1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5,
                                        help="Enter your weight in kilograms")
                st.markdown("<p style='color: #7f8c8d; font-size: 0.8rem; font-family: \"Segoe UI\", Arial, sans-serif;'>Healthy weight varies by height</p>", unsafe_allow_html=True)
            with bmi_col2:
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5,
                                        help="Enter your height in centimeters")
                st.markdown("<p style='color: #7f8c8d; font-size: 0.8rem; font-family: \"Segoe UI\", Arial, sans-serif;'>Average adult height: 160-180cm</p>", unsafe_allow_html=True)
            
            calculated_bmi = calculate_bmi(weight, height)
            calc_bmi_category = get_bmi_category(calculated_bmi)
            
            # Visual BMI Result display
            st.markdown(f"""
            <div style="margin-top: 20px; background: linear-gradient(to right, #f6f9fc, #edf3f9); 
                       padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;">
                <h2 style="margin: 0; color: #2c3e50; font-family: 'Segoe UI', Arial, sans-serif; font-weight: 700;">
                    {calculated_bmi}
                </h2>
                <p style="margin: 5px 0 15px 0; color: #7f8c8d; font-family: 'Segoe UI', Arial, sans-serif; font-size: 1rem;">
                    Your Calculated BMI
                </p>
                <div style="background-color: #e3f2fd; display: inline-block; padding: 5px 15px; border-radius: 30px; 
                          font-family: 'Segoe UI', Arial, sans-serif; font-weight: 600; color: #1565c0; font-size: 0.9rem;">
                    {calc_bmi_category}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # BMI Scale visualization
            st.markdown("""
            <div style="margin-top: 20px; margin-bottom: 20px;">
                <p style="font-weight: 500; font-family: 'Segoe UI', Arial, sans-serif; margin-bottom: 8px;">BMI Scale:</p>
                <div style="display: flex; width: 100%; height: 30px; border-radius: 5px; overflow: hidden; margin-bottom: 5px;">
                    <div style="flex-grow: 1; background-color: #81D4FA; text-align: center; padding: 5px 0; font-size: 0.8rem; color: #0D47A1; font-weight: 600; font-family: 'Segoe UI', Arial, sans-serif;">Underweight</div>
                    <div style="flex-grow: 1; background-color: #A5D6A7; text-align: center; padding: 5px 0; font-size: 0.8rem; color: #1B5E20; font-weight: 600; font-family: 'Segoe UI', Arial, sans-serif;">Normal</div>
                    <div style="flex-grow: 1; background-color: #FFE082; text-align: center; padding: 5px 0; font-size: 0.8rem; color: #FF6F00; font-weight: 600; font-family: 'Segoe UI', Arial, sans-serif;">Overweight</div>
                    <div style="flex-grow: 1; background-color: #FFAB91; text-align: center; padding: 5px 0; font-size: 0.8rem; color: #BF360C; font-weight: 600; font-family: 'Segoe UI', Arial, sans-serif;">Obese</div>
                </div>
                <div style="display: flex; width: 100%; justify-content: space-between; font-family: 'Segoe UI', Arial, sans-serif; font-size: 0.7rem; color: #7f8c8d;">
                    <span>16</span>
                    <span>18.5</span>
                    <span>25</span>
                    <span>30</span>
                    <span>40</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Use This BMI button with improved styling
            if st.button("Use This BMI Value", use_container_width=True, 
                        help="Apply this calculated BMI to your insurance estimate"):
                bmi = calculated_bmi
                st.experimental_rerun()
                
            # Add BMI formula explanation
            with st.expander("How is BMI calculated?"):
                st.markdown("""
                <div style="font-family: 'Segoe UI', Arial, sans-serif; color: #2c3e50;">
                    <p>The Body Mass Index (BMI) is calculated using the formula:</p>
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; text-align: center; font-family: 'Courier New', monospace; font-weight: 600;">
                        BMI = weight(kg) / [height(m)]²
                    </div>
                    <p>For example, if you weigh 70kg and are 170cm tall:</p>
                    <ul>
                        <li>Convert height to meters: 170cm = 1.7m</li>
                        <li>Calculate: 70 ÷ (1.7)² = 70 ÷ 2.89 = 24.2</li>
                    </ul>
                    <p><strong>BMI Categories:</strong></p>
                    <ul>
                        <li><strong>Below 18.5:</strong> Underweight</li>
                        <li><strong>18.5-24.9:</strong> Normal weight</li>
                        <li><strong>25-29.9:</strong> Overweight</li>
                        <li><strong>30 and above:</strong> Obese</li>
                    </ul>
                    <p><em>Note: BMI is a screening tool but not a diagnostic of body fatness or health.</em></p>
                </div>
                """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-header'>Additional Factors</div>", unsafe_allow_html=True)
        
        children = st.slider("Number of Dependents", min_value=0, max_value=10, value=0,
                            help="Number of children/dependents covered by the insurance")
        
        smoker_col1, smoker_col2 = st.columns([3, 1])
        with smoker_col1:
            smoker = st.select_slider("Smoking Status", options=["no", "yes"], value="no")
        with smoker_col2:
            if smoker == "yes":
                st.markdown("⚠️")
        
        # Visual indicator for smoking risk
        if smoker == "yes":
            st.warning("⚠️ Smoking significantly increases insurance costs (often up to 50% or more)")
        
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"],
                             help="Geographic region within the US")
        
        # Add tooltip information about regions
        st.markdown("<p class='tooltip'>Different regions may have different average costs due to local regulations, healthcare prices, and cost of living.</p>", 
                   unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Collect all inputs for prediction
    user_inputs = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }
    
    # Create columns for the prediction button and progress
    pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
    
    with pred_col2:
        predict_button = st.button("🔍 Calculate Insurance Premium", use_container_width=True)
    
    # Process prediction when button is clicked
    if predict_button:
        try:
            with st.spinner("Analyzing your profile and calculating insurance costs..."):
                # Process inputs and make prediction
                input_data = preprocess_input(age, sex, bmi, children, smoker, region)
                
                # Show a progress bar to simulate calculation
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                prediction = model.predict(input_data)[0]
                save_prediction(user_inputs, float(prediction))
            
            # Display prediction result with animation
            st.markdown("<div class='prediction-result animate-fade-in'>", unsafe_allow_html=True)
            st.subheader("Estimated Annual Insurance Premium:")
            st.markdown(f"<h1 style='color: #2c3e50; text-align: center;'>${prediction:.2f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>That's approximately <strong>${prediction/12:.2f}</strong> per month</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Create 3 tabs for different analyses
            analysis_tabs = st.tabs(["Risk Analysis", "Cost Breakdown", "Recommendations"])
            
            with analysis_tabs[0]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Risk Factor Analysis")
                
                # Calculate premium factors
                factors = calculate_premium_factors(age, sex, bmi, children, smoker, region)
                
                # Display radar chart
                st.plotly_chart(create_risk_radar_chart(factors), use_container_width=True)
                
                # Explanation of risk factors
                st.markdown("### Key Risk Factors")
                
                for factor, data in sorted(factors.items(), key=lambda x: x[1]["weight"], reverse=True):
                    impact_percentage = int(data["weight"] * 100)
                    
                    # Determine color based on weight
                    if data["weight"] > 0.5:
                        bar_color = THEME["danger"]
                    elif data["weight"] > 0.25:
                        bar_color = THEME["warning"] 
                    else:
                        bar_color = THEME["info"]
                    
                    # Create a custom progress bar with label
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span><strong>{factor}</strong>: {data["description"]}</span>
                            <span>{impact_percentage}% impact</span>
                        </div>
                        <div style="background-color: #e9ecef; border-radius: 5px; height: 10px;">
                            <div style="width: {impact_percentage}%; background-color: {bar_color}; height: 10px; border-radius: 5px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            with analysis_tabs[1]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Cost Breakdown Analysis")
                
                # Load sample data for comparison
                df = load_sample_data()
                
                # Calculate average costs for comparison
                avg_cost = df['charges'].mean()
                similar_profiles = df[
                    (df['age'] >= age - 5) & 
                    (df['age'] <= age + 5) & 
                    (df['smoker'] == smoker)
                ]
                similar_cost = similar_profiles['charges'].mean() if not similar_profiles.empty else avg_cost
                
                # Create comparison metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>${prediction:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Your Estimated Premium</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    diff_pct = ((prediction - avg_cost) / avg_cost) * 100
                    direction = "higher" if diff_pct > 0 else "lower"
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>${avg_cost:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Average Premium</div>", unsafe_allow_html=True)
                    st.markdown(f"<div>Your premium is <strong>{abs(diff_pct):.1f}%</strong> {direction}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    similar_diff_pct = ((prediction - similar_cost) / similar_cost) * 100
                    similar_direction = "higher" if similar_diff_pct > 0 else "lower"
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>${similar_cost:.2f}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Similar Profiles Average</div>", unsafe_allow_html=True)
                    st.markdown(f"<div>Your premium is <strong>{abs(similar_diff_pct):.1f}%</strong> {similar_direction}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Create cost breakdown chart
                st.subheader("Premium Composition")
                
                # Simulate cost breakdown (these are approximate values)
                base_cost = 2500  # Base insurance cost
                
                # Calculate additional costs based on factors
                age_cost = age * 50
                bmi_cost = (bmi - 20) * 300 if bmi > 20 else 0
                children_cost = children * 700
                smoker_cost = 15000 if smoker == "yes" else 0
                
                # Adjust to match prediction total
                total_components = base_cost + age_cost + bmi_cost + children_cost + smoker_cost
                adjustment_factor = prediction / total_components if total_components > 0 else 1
                
                cost_components = [
                    ("Base Premium", base_cost * adjustment_factor),
                    ("Age Factor", age_cost * adjustment_factor),
                    ("BMI Factor", bmi_cost * adjustment_factor),
                    ("Dependents", children_cost * adjustment_factor),
                    ("Smoking Status", smoker_cost * adjustment_factor)
                ]
                
                # Create DataFrame for chart
                cost_df = pd.DataFrame(cost_components, columns=["Component", "Cost"])
                
                # Create pie chart
                fig = px.pie(cost_df, values="Cost", names="Component", 
                            title="Estimated Premium Breakdown",
                            color_discrete_sequence=px.colors.sequential.Blues_r)
                
                fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cost component table
                cost_df["Percentage"] = (cost_df["Cost"] / cost_df["Cost"].sum()) * 100
                cost_df["Cost"] = cost_df["Cost"].map("${:.2f}".format)
                cost_df["Percentage"] = cost_df["Percentage"].map("{:.1f}%".format)
                
                st.dataframe(cost_df, hide_index=True, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            with analysis_tabs[2]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Premium Reduction Strategies")
                
                # Create recommendations based on user inputs
                recommendations = []
                potential_savings = []
                difficulty_levels = []
                
                if smoker == "yes":
                    recommendations.append("Quit smoking or participate in a smoking cessation program")
                    potential_savings.append("Very High (40-50%)")
                    difficulty_levels.append("High")
                
                if bmi > 30:
                    recommendations.append("Improve BMI through diet and exercise")
                    potential_savings.append("Moderate (10-20%)")
                    difficulty_levels.append("Medium")
                
                recommendations.append("Shop around and compare different insurance providers")
                potential_savings.append("Moderate (5-15%)")
                difficulty_levels.append("Low")
                
                recommendations.append("Consider a high-deductible health plan (HDHP)")
                potential_savings.append("Moderate (10-30%)")
                difficulty_levels.append("Low")
                
                recommendations.append("Look for employer-sponsored health insurance options")
                potential_savings.append("High (20-40%)")
                difficulty_levels.append("Medium")
                
                recommendations.append("See if you qualify for government subsidies or programs")
                potential_savings.append("High (Variable)")
                difficulty_levels.append("Medium")
                
                # Display recommendations as a table
                recommendations_df = pd.DataFrame({
                    "Strategy": recommendations,
                    "Potential Savings": potential_savings,
                    "Difficulty": difficulty_levels
                })
                
                st.dataframe(recommendations_df, hide_index=True, use_container_width=True)
                
                # Personalized recommendations
                st.subheader("Personalized Recommendations")
                
                if smoker == "yes":
                    st.error("❗ **Smoking Impact**: Smoking is the single largest factor affecting your premium. Quitting smoking could potentially save you thousands of dollars annually.")
                
                if bmi > 30:
                    st.warning(f"⚠️ **BMI Consideration**: Your current BMI of {bmi:.1f} falls into the '{get_bmi_category(bmi)}' category. Reducing your BMI to below 30 could significantly lower your premium.")
                
                if age > 50:
                    st.info("ℹ️ **Age Strategy**: As age increases, premiums tend to rise. Consider increasing your deductible or exploring Medicare/Medicaid options if eligible.")
                
                if children > 2:
                    st.info("ℹ️ **Family Plan**: With multiple dependents, family insurance plans or Children's Health Insurance Program (CHIP) might provide more cost-effective coverage.")
                
                if bmi <= 25 and smoker == "no":
                    st.success("✅ **Healthy Profile**: Your healthy lifestyle choices are already helping keep your premium costs down!")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add option to save to comparison
            st.markdown("---")
            comparison_cols = st.columns([3, 1])
            with comparison_cols[0]:
                st.markdown("### Save this prediction for comparison")
            with comparison_cols[1]:
                if st.button("Add to Compare", key="add_to_compare"):
                    if user_inputs not in [item["inputs"] for item in st.session_state['comparison_data']]:
                        comparison_item = {
                            "inputs": user_inputs,
                            "prediction": float(prediction),
                            "id": str(uuid.uuid4())[:8]
                        }
                        st.session_state['comparison_data'].append(comparison_item)
                        st.success("Added to comparison! Go to Compare tab to view.")
                    else:
                        st.warning("This profile is already in your comparisons.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            log_error(e, context="prediction_flow")

# ===========================================================================
# VISUALIZATION PAGE
# ===========================================================================

elif selected == "Visualize":
    st.markdown("<h1 class='main-header'>Insurance Cost Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Explore how different factors affect insurance premiums</p>", unsafe_allow_html=True)
    
    try:
        # Load sample data for visualization
        df = load_sample_data()
        
        # Add tab-based navigation for different visualizations
        viz_tabs = st.tabs(["Interactive Explorer", "Key Factors", "Demographic Analysis", "Regional Trends", "Correlation Matrix"])
        
        with viz_tabs[0]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Interactive Insurance Cost Explorer")
            
            # Create flexible visualization options
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-Axis", options=["age", "bmi", "children", "charges"], index=0)
                color_by = st.selectbox("Color By", options=["smoker", "sex", "region", "None"], index=0)
            
            with col2:
                y_axis = st.selectbox("Y-Axis", options=["charges", "age", "bmi", "children"], index=0)
                size_by = st.selectbox("Size By", options=["None", "age", "bmi", "children"], index=0)
                
                # Add advanced visualization option
                chart_type = st.selectbox("Chart Type", options=[
                    "Scatter Plot", 
                    "Bubble Chart",
                    "3D Scatter",
                    "Violin Plot",
                    "Box Plot",
                    "Density Contour"
                ])
            
            # Handle "None" options
            color_col = None if color_by == "None" else color_by
            size_col = None if size_by == "None" else size_by
            
            # Create conditional layout based on chart type
            if chart_type in ["Scatter Plot", "Bubble Chart"]:
                # Advanced configuration for 2D plots
                advanced_options = st.expander("Advanced Options")
                with advanced_options:
                    trend_line = st.checkbox("Show Trend Line", value=True)
                    log_scale = st.checkbox("Logarithmic Scale", value=False)
                    facet_by = st.selectbox("Facet By", options=["None", "smoker", "sex", "region"], index=0)
                    opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
                    
                    # Color scheme selection
                    color_scheme = st.selectbox("Color Scheme", options=[
                        "Blues", "Reds", "Greens", "Viridis", "Plasma", "Turbo"
                    ])
                    
                # Create visualization based on type
                if chart_type == "Scatter Plot":
                    fig = px.scatter(
                        df, x=x_axis, y=y_axis, 
                        color=color_col, 
                        hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                        title=f"Insurance Costs: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                        height=600,
                        opacity=opacity,
                        log_y=log_scale if y_axis == "charges" else False,
                        log_x=log_scale if x_axis == "charges" else False,
                        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                        color_continuous_scale=color_scheme,
                        facet_col=None if facet_by == "None" else facet_by,
                        trendline="ols" if trend_line else None
                    )
                else:  # Bubble Chart
                    if size_by == "None":
                        st.warning("Please select a 'Size By' variable for the bubble chart to display properly.")
                        size_col = "age"  # Default if none selected
                        
                    fig = px.scatter(
                        df, x=x_axis, y=y_axis, 
                        color=color_col, 
                        size=size_col,
                        size_max=25,
                        hover_data=["sex", "children", "region", "smoker", "charges", "bmi", "age"],
                        title=f"Insurance Costs Bubble Chart: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                        height=600,
                        opacity=opacity,
                        log_y=log_scale if y_axis == "charges" else False,
                        log_x=log_scale if x_axis == "charges" else False,
                        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                        color_continuous_scale=color_scheme,
                        facet_col=None if facet_by == "None" else facet_by,
                        trendline="ols" if trend_line else None
                    )
                
                # Enhance figure layout
                fig.update_layout(
                    xaxis_title=x_axis.capitalize(),
                    yaxis_title=y_axis.capitalize(),
                    legend_title=color_by.capitalize() if color_by != "None" else "",
                    template="plotly_white",
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                
            elif chart_type == "3D Scatter":
                # Special options for 3D
                advanced_options = st.expander("3D Plot Options")
                with advanced_options:
                    z_axis = st.selectbox("Z-Axis", options=["age", "bmi", "children", "charges"], 
                                        index=3 if "charges" not in [x_axis, y_axis] else 0)
                    opacity = st.slider("Point Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
                    color_scheme = st.selectbox("Color Scheme", options=[
                        "Viridis", "Plasma", "Inferno", "Magma", "Turbo"
                    ])
                
                fig = px.scatter_3d(
                    df, x=x_axis, y=y_axis, z=z_axis,
                    color=color_col,
                    size=size_col if size_by != "None" else None,
                    opacity=opacity,
                    size_max=10,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"},
                    color_continuous_scale=color_scheme,
                    title=f"3D Plot: {x_axis.capitalize()} vs {y_axis.capitalize()} vs {z_axis.capitalize()}"
                )
                
                fig.update_layout(
                    scene=dict(
                        xaxis_title=x_axis.capitalize(),
                        yaxis_title=y_axis.capitalize(),
                        zaxis_title=z_axis.capitalize(),
                    ),
                    height=700,
                    legend_title=color_by.capitalize() if color_by != "None" else ""
                )
                
            elif chart_type in ["Violin Plot", "Box Plot"]:
                # Options for distribution plots
                advanced_options = st.expander("Distribution Plot Options")
                with advanced_options:
                    points = st.radio("Show Points", options=["all", "outliers", "none"], index=1)
                    box = st.checkbox("Show Box", value=True)
                    notched = st.checkbox("Notched Box", value=False)
                    group_by = st.selectbox("Group By", options=["smoker", "sex", "region"], 
                                          index=0 if color_by != "smoker" else 1)
                
                if chart_type == "Violin Plot":
                    fig = px.violin(
                        df, x=group_by, y=y_axis,
                        color=color_col if color_col != group_by else None,
                        box=box,
                        points=points,
                        hover_data=["age", "bmi", "children", "charges"],
                        title=f"{y_axis.capitalize()} Distribution by {group_by.capitalize()}",
                        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"}
                    )
                else:  # Box Plot
                    fig = px.box(
                        df, x=group_by, y=y_axis,
                        color=color_col if color_col != group_by else None,
                        points=points,
                        notched=notched,
                        hover_data=["age", "bmi", "children", "charges"],
                        title=f"{y_axis.capitalize()} Distribution by {group_by.capitalize()}",
                        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"}
                    )
                
                fig.update_layout(
                    xaxis_title=group_by.capitalize(),
                    yaxis_title=y_axis.capitalize(),
                    legend_title=color_by.capitalize() if color_by != "None" and color_by != group_by else "",
                    template="plotly_white",
                    height=600
                )
                
            else:  # Density Contour
                advanced_options = st.expander("Density Plot Options")
                with advanced_options:
                    contour_coloring = st.radio("Contour Coloring", options=["fill", "lines", "none"], index=0)
                    hist_func = st.selectbox("Histogram Function", options=["count", "avg", "sum", "min", "max"], index=0)
                    marginal = st.selectbox("Marginal Plots", options=["box", "violin", "histogram", "rug", "none"], index=2)
                
                # Handle None for color to use a numeric field
                if color_by == "None":
                    color_col = "charges" if y_axis != "charges" and x_axis != "charges" else "age"
                
                fig = px.density_contour(
                    df, x=x_axis, y=y_axis,
                    color=color_col,
                    marginal_x=None if marginal == "none" else marginal,
                    marginal_y=None if marginal == "none" else marginal,
                    histfunc=hist_func,
                    title=f"Density Contour: {x_axis.capitalize()} vs {y_axis.capitalize()}",
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"}
                )
                
                if contour_coloring != "none":
                    fig.update_traces(
                        contours_coloring=contour_coloring,
                        selector=dict(type='contour')
                    )
                
                fig.update_layout(
                    xaxis_title=x_axis.capitalize(),
                    yaxis_title=y_axis.capitalize(),
                    legend_title=color_by.capitalize() if color_by != "None" else "",
                    template="plotly_white",
                    height=600
                )
            
            # Display the visualization
            st.plotly_chart(fig, use_container_width=True)
            
            # Data filtering options
            st.subheader("Filter Data")
            filter_cols = st.columns(4)
            
            with filter_cols[0]:
                age_range = st.slider("Age Range", min_value=int(df.age.min()), max_value=int(df.age.max()), 
                                    value=(int(df.age.min()), int(df.age.max())))
            
            with filter_cols[1]:
                bmi_range = st.slider("BMI Range", min_value=float(df.bmi.min()), max_value=float(df.bmi.max()), 
                                     value=(float(df.bmi.min()), float(df.bmi.max())))
            
            with filter_cols[2]:
                smoker_filter = st.multiselect("Smoking Status", options=df.smoker.unique(), default=df.smoker.unique())
            
            with filter_cols[3]:
                region_filter = st.multiselect("Region", options=df.region.unique(), default=df.region.unique())
            
            # Apply filters
            filtered_df = df[
                (df.age >= age_range[0]) & 
                (df.age <= age_range[1]) &
                (df.bmi >= bmi_range[0]) & 
                (df.bmi <= bmi_range[1]) &
                (df.smoker.isin(smoker_filter)) &
                (df.region.isin(region_filter))
            ]
            
            # Display statistics about filtered data
            st.subheader("Summary Statistics")
            
            # Create tabbed statistics view
            stat_tabs = st.tabs(["Overview", "Detailed Stats", "Distribution"])
            
            with stat_tabs[0]:
                # Quick metrics
                stats_cols = st.columns(4)
                
                with stats_cols[0]:
                    st.metric("Average Cost", f"${filtered_df.charges.mean():.2f}")
                
                with stats_cols[1]:
                    st.metric("Median Cost", f"${filtered_df.charges.median():.2f}")
                
                with stats_cols[2]:
                    st.metric("Min Cost", f"${filtered_df.charges.min():.2f}")
                
                with stats_cols[3]:
                    st.metric("Max Cost", f"${filtered_df.charges.max():.2f}")
                
                # Add count info
                st.info(f"Displaying statistics for {len(filtered_df)} records (out of {len(df)} total)")
            
            with stat_tabs[1]:
                # More comprehensive statistics
                st.subheader("Detailed Statistics")
                
                # Generate detailed statistics and round to 2 decimal places
                detailed_stats = filtered_df.describe().T
                detailed_stats = detailed_stats.round(2)
                
                # Custom formatting for display
                formatted_stats = detailed_stats.copy()
                formatted_stats.columns = [col.capitalize() for col in formatted_stats.columns]
                
                # Special formatting for charges column
                if "charges" in formatted_stats.index:
                    for col in formatted_stats.columns:
                        if col != "Count":
                            formatted_stats.loc["charges", col] = f"${formatted_stats.loc['charges', col]:,.2f}"
                
                st.dataframe(formatted_stats, use_container_width=True)
                
                # Add correlation data
                st.subheader("Correlation with Insurance Charges")
                
                # Calculate correlations and sort
                corr_with_charges = filtered_df.drop(columns=["charges"]).corrwith(filtered_df.charges).sort_values(ascending=False)
                
                # Format for display
                corr_df = pd.DataFrame({
                    "Factor": corr_with_charges.index,
                    "Correlation": corr_with_charges.values
                })
                
                # Create a horizontal bar chart
                fig = px.bar(
                    corr_df, 
                    x="Correlation", 
                    y="Factor",
                    orientation="h",
                    title="Correlation with Insurance Charges",
                    color="Correlation",
                    color_continuous_scale="RdBu_r",
                    range_color=[-1, 1]
                )
                
                fig.update_layout(
                    xaxis_title="Pearson Correlation",
                    yaxis_title="",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation interpretation
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px;">
                    <p style="margin: 0; color: #2c3e50; font-size: 0.9rem;">
                        <strong>Interpretation:</strong> Correlation coefficients range from -1 to 1:
                        <ul style="margin-bottom: 0;">
                            <li>Values close to 1 indicate a strong positive correlation (as one increases, the other increases)</li>
                            <li>Values close to -1 indicate a strong negative correlation (as one increases, the other decreases)</li>
                            <li>Values close to 0 indicate little to no linear relationship</li>
                        </ul>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_tabs[2]:
                # Show distribution of charges
                st.subheader("Charge Distribution")
                
                # Create histogram of charges
                fig = px.histogram(
                    filtered_df, 
                    x="charges",
                    nbins=50,
                    marginal="box",
                    title="Distribution of Insurance Charges",
                    color_discrete_sequence=["#3498db"]
                )
                
                fig.update_layout(
                    xaxis_title="Insurance Charges ($)",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add distribution statistics
                dist_cols = st.columns(3)
                
                with dist_cols[0]:
                    st.metric("Skewness", f"{filtered_df.charges.skew():.2f}")
                    st.markdown("<div style='font-size: 0.8rem; color: #666;'>Positive values indicate right skew (tail extends to the right)</div>", unsafe_allow_html=True)
                
                with dist_cols[1]:
                    st.metric("Kurtosis", f"{filtered_df.charges.kurtosis():.2f}")
                    st.markdown("<div style='font-size: 0.8rem; color: #666;'>Values > 0 indicate heavier tails than a normal distribution</div>", unsafe_allow_html=True)
                
                with dist_cols[2]:
                    q75, q25 = np.percentile(filtered_df.charges, [75, 25])
                    iqr = q75 - q25
                    st.metric("Interquartile Range", f"${iqr:.2f}")
                    st.markdown("<div style='font-size: 0.8rem; color: #666;'>Difference between 75th and 25th percentiles</div>", unsafe_allow_html=True)
            
            # Export options
            export_cols = st.columns([3, 1])
            
            with export_cols[0]:
                st.markdown("### Export Filtered Data")
                st.markdown("Download the data displayed in your current filtered view:")
            
            with export_cols[1]:
                st.markdown(get_csv_download_link(filtered_df, "filtered_insurance_data.csv", "📥 Download CSV"), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        # Advanced error handling with recovery options
        error_id = log_error(e, context="visualization_page")
        
        # Create visually appealing error message
        st.markdown(f"""
        <div style="background-color: #FFF0F0; border-left: 5px solid #E74C3C; padding: 20px; border-radius: 5px; margin: 20px 0; animation: fadeIn 0.5s ease-in-out;">
            <h3 style="color: #E74C3C; margin-top: 0;"><i>⚠️ Visualization Error</i></h3>
            <p style="margin-bottom: 15px;">We encountered an issue while processing your visualization. This has been logged for our team to investigate (Error ID: {error_id}).</p>
            <details>
                <summary style="cursor: pointer; color: #555; font-weight: 500;">Technical Details</summary>
                <div style="background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin-top: 10px; font-family: monospace; font-size: 0.85em; white-space: pre-wrap;">{str(e)}</div>
            </details>
        </div>
        """, unsafe_allow_html=True)
        
        # Provide recovery options
        recovery_cols = st.columns([2, 1, 1])
        with recovery_cols[0]:
            st.info("💡 You can try adjusting your visualization parameters or filters")
        
        with recovery_cols[1]:
            if st.button("Reset Filters", key="reset_viz_error"):
                st.experimental_rerun()
                
        with recovery_cols[2]:
            if st.button("Load Sample Data", key="load_sample_viz_error"):
                # Force reload the data
                if 'sample_data' in st.session_state:
                    del st.session_state['sample_data']
                
                # Show fallback visualization
                try:
                    # Create a simple fallback visualization
                    fallback_df = pd.DataFrame({
                        'age': [25, 30, 40, 50, 60],
                        'bmi': [22, 28, 30, 26, 24],
                        'charges': [5000, 7500, 15000, 20000, 25000],
                        'smoker': ['no', 'no', 'yes', 'no', 'yes'],
                        'region': ['northeast', 'northwest', 'southeast', 'southwest', 'northeast'],
                        'sex': ['male', 'female', 'male', 'female', 'male'],
                        'children': [0, 1, 2, 1, 0]
                    })
                    
                    st.subheader("Sample Data Visualization")
                    fig = px.scatter(
                        fallback_df, x="age", y="charges", 
                        color="smoker",
                        size="bmi",
                        hover_data=["sex", "children", "region"],
                        title="Sample Insurance Cost Data",
                        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Unable to display fallback visualization")
        
        # Add data insights even when visualization fails
        st.markdown("""
        <div style="background-color: #F7FBFF; padding: 20px; border-radius: 5px; margin-top: 30px; border-left: 5px solid #3498DB;">
            <h3 style="color: #3498DB; margin-top: 0;">💡 Key Insurance Cost Insights</h3>
            <ul style="margin-bottom: 0;">
                <li><strong>Smoking Status:</strong> Typically the most significant factor, often increasing premiums by 200-300%.</li>
                <li><strong>Age:</strong> Insurance costs generally increase with age, with costs rising more rapidly after age 50.</li>
                <li><strong>BMI:</strong> Higher BMI values (especially above 30) are associated with higher insurance costs.</li>
                <li><strong>Dependents:</strong> Each additional dependent typically increases insurance costs by a moderate amount.</li>
                <li><strong>Region:</strong> Geographic differences can affect costs due to varying healthcare prices and regulations.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Troubleshooting guidance
        with st.expander("Troubleshooting Suggestions"):
            st.markdown("""
            - **Try different chart types**: Some visualizations may work better with certain data types
            - **Adjust your filters**: Some filter combinations might cause errors
            - **Check for extreme values**: Outliers in the data may cause visualization issues
            - **Refresh the page**: Sometimes clearing the browser cache helps resolve issues
            - **Contact support**: If the problem persists, contact our support team with your Error ID
            """)
            
        # Add diagnostic tools for more advanced users
        with st.expander("Diagnostic Tools"):
            diagnostics_tabs = st.tabs(["System Info", "Error Log", "Data Sample"])
            
            with diagnostics_tabs[0]:
                st.json({
                    "streamlit_version": st.__version__,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "timestamp": datetime.datetime.now().strftime("%Y-%M-%d %H:%M:%S"),
                    "browser": st.session_state.get('browser_info', 'Unknown'),
                    "platform": sys.platform
                })
                
            with diagnostics_tabs[1]:
                if st.session_state['error_log']:
                    for i, error in enumerate(st.session_state['error_log'][-5:]):
                        st.text(f"Error {i+1}: {error['timestamp']} - {error['context']}")
                        st.code(error['error'], language="python")
                else:
                    st.info("No previous errors recorded in this session")
                    
            with diagnostics_tabs[2]:
                try:
                    sample_df = load_sample_data().head(5)
                    st.dataframe(sample_df)
                except:
                    st.warning("Unable to load data sample")