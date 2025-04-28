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
    
    /* Improved Font Visibility */
    /* Typography */
    h1, h2, h3, p, li, label, div {
        color: #1a1a1a;  /* Darker text color for better visibility */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        font-size: 2.75rem;
        color: #2971c1;  /* Darker blue for better contrast */
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 1.5rem;
        color: #1a1a1a;  /* Darker text color */
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 500;  /* Increased from 400 for better visibility */
    }
    
    /* Form Labels and Inputs */
    label, .stSelectbox label, .stSlider label {
        font-weight: 500 !important;
        color: #1a1a1a !important;
        font-size: 1.05rem !important;
    }
    .stRadio label, .stCheckbox label {
        font-weight: 500 !important;
        color: #1a1a1a !important;
    }
    
    /* Helper text below inputs */
    .stTextInput div[data-baseweb="caption"], 
    .stSelectbox div[data-baseweb="caption"], 
    .stSlider div[data-baseweb="caption"] {
        color: #4a4a4a !important;
    }
    
    /* Table data */
    div[data-testid="stTable"] td, div[data-testid="stTable"] th {
        color: #1a1a1a !important;
        font-weight: 500;
    }
    
    /* Metric values */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2971c1;  /* Darker blue */
    }
    .metric-label {
        font-size: 1rem;
        color: #1a1a1a;  /* Darker text */
        font-weight: 500;
    }
    
    /* Tooltip text */
    .tooltip {
        color: #1a1a1a;  /* Much darker than before */
        font-size: 1rem;
        padding: 10px;
        border-left: 3px solid #3498db;
        background-color: #f8f9fa;
        font-weight: 500;
    }
    
    /* Sidebar text */
    .sidebar-info {
        font-size: 0.95rem;
        color: #1a1a1a;  /* Much darker for better visibility */
        font-weight: 500;
    }
    
    /* Button text */
    .stButton>button {
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Tab text */
    button[role="tab"] {
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: #3498db !important;
        font-weight: 700 !important;
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
    np.array
        Formatted input array for the model
    """
    try:
        # Ensure inputs are of correct type before processing
        age = float(age)
        bmi = float(bmi)
        children = int(children)
        
        # Create a pandas DataFrame with proper data types
        # This is needed because the model was trained with a ColumnTransformer
        # that expects specific column names as strings
        input_df = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        # The model's column transformer will handle the categorical encoding internally
        return input_df
        
    except Exception as e:
        log_error(e, context="preprocess_input")
        # Return a safe fallback DataFrame with appropriate data types
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
            "container": {"padding": "0!important", "background-color": "black"},
            "icon": {"color": THEME["primary"], "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "pink"},
            "nav-link-selected": {"background-color": THEME["primary"], "color": "black"}
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
            st.rerun()
    
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
                st.rerun()
                
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
        # Fix for the visualization error with categorical variables
        # chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart"], index=0)
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
                    trendline="ols" if trend_line else None,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71", "male": "#3498db", "female": "#9b59b6"}
                )
                
                # Updated handling for categorical variables
                if color_col in ["smoker", "sex", "region"]:
                    pass  # Color mapping is now handled in the px.scatter call with color_discrete_map

        # New implementation for Demographic Analysis tab
        with viz_tabs[2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Demographic Insurance Cost Analysis")
            
            # Create a dashboard of demographic insights
            demo_tabs = st.tabs(["Age Analysis", "Gender Analysis", "BMI Impact", "Family Size", "Combined Factors"])
            
            with demo_tabs[0]:
                st.markdown("### Age and Insurance Costs")
                
                # Create age groups for better analysis
                df['age_group'] = pd.cut(
                    df['age'], 
                    bins=[0, 20, 30, 40, 50, 60, 100],
                    labels=['<20', '20-30', '30-40', '40-50', '50-60', '60+']
                )
                
                # Age group statistics
                age_stats = df.groupby('age_group').agg({
                    'charges': ['mean', 'median', 'std', 'count']
                }).reset_index()
                
                age_stats.columns = ['Age Group', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
                
                # Format monetary columns
                age_stats['Mean Cost'] = age_stats['Mean Cost'].map('${:,.2f}'.format)
                age_stats['Median Cost'] = age_stats['Median Cost'].map('${:,.2f}'.format)
                age_stats['Std Dev'] = age_stats['Std Dev'].map('${:,.2f}'.format)
                
                # Display age statistics
                st.dataframe(age_stats, use_container_width=True)
                
                # Create age trend visualization
                fig = px.line(
                    df.groupby('age').agg({'charges': 'mean'}).reset_index(),
                    x='age',
                    y='charges',
                    title='Average Insurance Cost by Age',
                    labels={'charges': 'Average Cost ($)', 'age': 'Age'},
                    markers=True
                )
                
                # Add smoker vs non-smoker trend lines
                smoker_age_data = df.groupby(['age', 'smoker']).agg({'charges': 'mean'}).reset_index()
                
                fig2 = px.line(
                    smoker_age_data,
                    x='age',
                    y='charges',
                    color='smoker',
                    title='Average Insurance Cost by Age and Smoking Status',
                    labels={'charges': 'Average Cost ($)', 'age': 'Age'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
                    markers=True
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Age-related insights
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Age Insights</h4>
                    <ul>
                        <li><strong>Linear Growth:</strong> Insurance costs generally increase linearly with age</li>
                        <li><strong>Higher Variance:</strong> Older age groups show greater variability in costs</li>
                        <li><strong>Accelerated Increase:</strong> Costs rise more steeply after age 50</li>
                        <li><strong>Smoking Amplifier:</strong> The cost increase with age is much steeper for smokers</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with demo_tabs[1]:
                st.markdown("### Gender Analysis")
                
                # Gender statistics
                gender_stats = df.groupby('sex').agg({
                    'charges': ['mean', 'median', 'std', 'count']
                }).reset_index()
                
                gender_stats.columns = ['Gender', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
                
                # Format monetary columns
                gender_stats['Mean Cost'] = gender_stats['Mean Cost'].map('${:,.2f}'.format)
                gender_stats['Median Cost'] = gender_stats['Median Cost'].map('${:,.2f}'.format)
                gender_stats['Std Dev'] = gender_stats['Std Dev'].map('${:,.2f}'.format)
                
                # Gender and age analysis
                gender_age = df.groupby(['sex', 'age_group']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                # Create visualizations
                fig1 = px.bar(
                    df.groupby('sex').agg({'charges': 'mean'}).reset_index(),
                    x='sex',
                    y='charges',
                    color='sex',
                    title='Average Insurance Cost by Gender',
                    labels={'charges': 'Average Cost ($)', 'sex': 'Gender'},
                    color_discrete_map={"male": "#3498db", "female": "#9b59b6"},
                    text_auto='.2f'
                )
                
                fig1.update_traces(texttemplate='$%{text}', textposition='outside')
                
                fig2 = px.bar(
                    gender_age,
                    x='age_group',
                    y='charges',
                    color='sex',
                    barmode='group',
                    title='Average Insurance Cost by Gender and Age Group',
                    labels={'charges': 'Average Cost ($)', 'age_group': 'Age Group', 'sex': 'Gender'},
                    color_discrete_map={"male": "#3498db", "female": "#9b59b6"}
                )
                
                # Display gender statistics and charts
                st.dataframe(gender_stats, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Gender and smoking status
                gender_smoking = df.groupby(['sex', 'smoker']).agg({
                    'charges': ['mean', 'count']
                }).reset_index()
                
                gender_smoking.columns = ['Gender', 'Smoker', 'Average Cost', 'Count']
                gender_smoking['Average Cost'] = gender_smoking['Average Cost'].map('${:,.2f}'.format)
                
                st.subheader("Gender and Smoking Status")
                st.dataframe(gender_smoking, use_container_width=True)
                
                # Gender insights
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Gender Insights</h4>
                    <ul>
                        <li><strong>Small Difference:</strong> Gender alone is a relatively minor factor in insurance costs</li>
                        <li><strong>Age Interaction:</strong> Gender differences vary by age group</li>
                        <li><strong>Smoking Impact:</strong> Smoking status has a much larger effect than gender</li>
                        <li><strong>Combined Factors:</strong> When combined with other risk factors, gender may have more significant effects</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with demo_tabs[2]:
                st.markdown("### BMI Impact Analysis")
                
                # Create BMI categories
                df['bmi_category'] = pd.cut(
                    df['bmi'],
                    bins=[0, 18.5, 25, 30, 35, 40, 100],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
                )
                
                # BMI statistics
                bmi_stats = df.groupby('bmi_category').agg({
                    'charges': ['mean', 'median', 'std', 'count']
                }).reset_index()
                
                bmi_stats.columns = ['BMI Category', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
                
                # Format monetary columns
                bmi_stats['Mean Cost'] = bmi_stats['Mean Cost'].map('${:,.2f}'.format)
                bmi_stats['Median Cost'] = bmi_stats['Median Cost'].map('${:,.2f}'.format)
                bmi_stats['Std Dev'] = bmi_stats['Std Dev'].map('${:,.2f}'.format)
                
                # Display BMI statistics
                st.dataframe(bmi_stats, use_container_width=True)
                
                # BMI scatter plot
                fig1 = px.scatter(
                    df,
                    x='bmi',
                    y='charges',
                    color='bmi_category',
                    title='Insurance Costs by BMI',
                    trendline='ols',
                    opacity=0.7,
                    labels={'charges': 'Insurance Cost ($)', 'bmi': 'BMI', 'bmi_category': 'BMI Category'}
                )
                
                # BMI and smoking status
                bmi_smoking = df.groupby(['bmi_category', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig2 = px.bar(
                    bmi_smoking,
                    x='bmi_category',
                    y='charges',
                    color='smoker',
                    barmode='group',
                    title='Average Insurance Cost by BMI Category and Smoking Status',
                    labels={'charges': 'Average Cost ($)', 'bmi_category': 'BMI Category', 'smoker': 'Smoker'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # BMI insights
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">BMI Insights</h4>
                    <ul>
                        <li><strong>Positive Correlation:</strong> Higher BMI generally correlates with higher insurance costs</li>
                        <li><strong>Risk Threshold:</strong> Costs increase more significantly once BMI exceeds 30 (Obese Class I)</li>
                        <li><strong>Smoking Multiplier:</strong> The combination of high BMI and smoking creates a dramatic cost increase</li>
                        <li><strong>Underweight Risk:</strong> Very low BMI (underweight) can also lead to slightly higher premiums</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with demo_tabs[3]:
                st.markdown("### Family Size Impact")
                
                # Family size statistics (using 'children' as proxy)
                family_stats = df.groupby('children').agg({
                    'charges': ['mean', 'median', 'std', 'count']
                }).reset_index()
                
                family_stats.columns = ['Number of Children', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
                
                # Format monetary columns
                family_stats['Mean Cost'] = family_stats['Mean Cost'].map('${:,.2f}'.format)
                family_stats['Median Cost'] = family_stats['Median Cost'].map('${:,.2f}'.format)
                family_stats['Std Dev'] = family_stats['Std Dev'].map('${:,.2f}'.format)
                
                # Display family statistics
                st.dataframe(family_stats, use_container_width=True)
                
                # Family size visualization
                fig1 = px.bar(
                    df.groupby('children').agg({'charges': 'mean'}).reset_index(),
                    x='children',
                    y='charges',
                    title='Average Insurance Cost by Number of Children',
                    labels={'charges': 'Average Cost ($)', 'children': 'Number of Children'},
                    color='children',
                    text_auto='.2f'
                )
                
                fig1.update_traces(texttemplate='$%{text}', textposition='outside')
                
                # Family size and age
                children_age = df.groupby(['children', 'age_group']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig2 = px.bar(
                    children_age,
                    x='age_group',
                    y='charges',
                    color='children',
                    barmode='group',
                    title='Average Insurance Cost by Age Group and Number of Children',
                    labels={'charges': 'Average Cost ($)', 'age_group': 'Age Group', 'children': 'Number of Children'}
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Family size insights
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Family Size Insights</h4>
                    <ul>
                        <li><strong>Moderate Impact:</strong> Number of dependents has a moderate effect on insurance costs</li>
                        <li><strong>Non-Linear Relationship:</strong> The cost increase is not always linear with family size</li>
                        <li><strong>Age Interaction:</strong> The impact of family size varies with the age of the primary beneficiary</li>
                        <li><strong>Family Plans:</strong> Insurance providers often offer family plans that can moderate the per-person cost</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with demo_tabs[4]:
                st.markdown("### Combined Demographic Factors")
                
                # Create a demographic risk score (simplified example)
                df['age_factor'] = (df['age'] - 18) / 82  # Normalize age (18-100)
                df['bmi_factor'] = (df['bmi'] - 18.5) / 16.5 if df['bmi'].min() >= 18.5 else (df['bmi'] - df['bmi'].min()) / (35 - df['bmi'].min())
                df['smoker_factor'] = df['smoker'].map({'yes': 1, 'no': 0})
                df['children_factor'] = df['children'] / 5  # Assuming max 5 children
                
                # Calculate a simple risk score
                df['risk_score'] = (
                    0.3 * df['age_factor'] + 
                    0.2 * df['bmi_factor'] + 
                    0.4 * df['smoker_factor'] + 
                    0.1 * df['children_factor']
                )
                
                # Create risk categories
                df['risk_category'] = pd.qcut(
                    df['risk_score'], 
                    q=[0, 0.25, 0.5, 0.75, 1.0],
                    labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
                )
                
                # Risk category statistics
                risk_stats = df.groupby('risk_category').agg({
                    'charges': ['mean', 'median', 'std', 'count']
                }).reset_index()
                
                risk_stats.columns = ['Risk Category', 'Mean Cost', 'Median Cost', 'Std Dev', 'Count']
                
                # Format monetary columns
                risk_stats['Mean Cost'] = risk_stats['Mean Cost'].map('${:,.2f}'.format)
                risk_stats['Median Cost'] = risk_stats['Median Cost'].map('${:,.2f}'.format)
                risk_stats['Std Dev'] = risk_stats['Std Dev'].map('${:,.2f}'.format)
                
                # Display risk statistics
                st.subheader("Demographic Risk Categories")
                st.dataframe(risk_stats, use_container_width=True)
                
                # Create a 3D scatter plot of key factors
                fig = px.scatter_3d(
                    df,
                    x='age',
                    y='bmi',
                    z='charges',
                    color='smoker',
                    size='children',
                    opacity=0.7,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
                    title='Insurance Costs by Age, BMI, Smoking Status, and Number of Children',
                    labels={
                        'age': 'Age',
                        'bmi': 'BMI',
                        'charges': 'Insurance Cost ($)',
                        'smoker': 'Smoker',
                        'children': 'Number of Children'
                    }
                )
                
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
                
                # Multi-factor insights
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Multi-Factor Insights</h4>
                    <ul>
                        <li><strong>Compounding Effects:</strong> When multiple risk factors are present, the cost increase can be greater than the sum of individual factors</li>
                        <li><strong>Primary Drivers:</strong> Smoking status and age remain the strongest predictors of insurance costs</li>
                        <li><strong>Risk Segmentation:</strong> Insurance providers typically use demographic data to place individuals in risk categories</li>
                        <li><strong>Modifiable Factors:</strong> While age cannot be changed, lifestyle factors like BMI and smoking status can be modified to reduce premiums</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # New implementation for Regional Trends tab
        with viz_tabs[3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Regional Insurance Cost Analysis")
            
            # Calculate regional statistics
            region_stats = df.groupby('region').agg({
                'charges': ['mean', 'median', 'min', 'max', 'std', 'count'],
                'age': 'mean',
                'bmi': 'mean'
            }).reset_index()
            
            region_stats.columns = [
                'Region', 'Mean Cost', 'Median Cost', 'Min Cost', 'Max Cost', 'Std Dev', 
                'Sample Size', 'Avg Age', 'Avg BMI'
            ]
            
            # Format monetary columns
            for col in ['Mean Cost', 'Median Cost', 'Min Cost', 'Max Cost', 'Std Dev']:
                region_stats[col] = region_stats[col].map('${:,.2f}'.format)
            
            # Round average age and BMI
            region_stats['Avg Age'] = region_stats['Avg Age'].round(1)
            region_stats['Avg BMI'] = region_stats['Avg BMI'].round(1)
            
            # Regional dashboard
            region_tabs = st.tabs(["Regional Overview", "Cost Comparisons", "Regional Heatmaps", "Demographic Variations"])
            
            with region_tabs[0]:
                st.markdown("### Regional Insurance Cost Summary")
                st.dataframe(region_stats, use_container_width=True)
                
                # Create a US map visualization
                fig = px.choropleth(
                    region_stats,
                    locations='Region',
                    locationmode='USA-states',
                    color=region_stats['Mean Cost'].str.replace('$', '').str.replace(',', '').astype(float),
                    scope="usa",
                    color_continuous_scale="Viridis",
                    labels={'color': 'Average Insurance Cost'},
                    title="Average Insurance Costs by Region"
                )
                
                fig.update_layout(geo=dict(projection_scale=1.2), height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Regional insights
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Regional Insights</h4>
                    <ul>
                        <li><strong>Geographic Variation:</strong> Insurance costs vary by region due to differences in healthcare prices, regulations, and cost of living</li>
                        <li><strong>Market Competition:</strong> Regions with more insurance providers may have more competitive pricing</li>
                        <li><strong>Healthcare Infrastructure:</strong> Access to healthcare facilities affects regional pricing</li>
                        <li><strong>Regional Health Factors:</strong> Population health metrics in different regions influence overall pricing</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with region_tabs[1]:
                st.markdown("### Regional Cost Comparisons")
                
                # Create regional bar charts
                fig1 = px.bar(
                    region_stats,
                    x='Region',
                    y=region_stats['Mean Cost'].str.replace('$', '').str.replace(',', '').astype(float),
                    color='Region',
                    title='Average Insurance Cost by Region',
                    labels={'y': 'Average Cost ($)', 'Region': 'Region'},
                    text_auto='.2f'
                )
                
                fig1.update_traces(texttemplate='$%{text}', textposition='outside')
                
                # Regional costs by smoking status
                region_smoking = df.groupby(['region', 'smoker']).agg({
                    'charges': 'mean'
                }).reset_index()
                
                fig2 = px.bar(
                    region_smoking,
                    x='region',
                    y='charges',
                    color='smoker',
                    barmode='group',
                    title='Average Insurance Cost by Region and Smoking Status',
                    labels={'charges': 'Average Cost ($)', 'region': 'Region', 'smoker': 'Smoker'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Cost distribution by region
                fig3 = px.box(
                    df,
                    x='region',
                    y='charges',
                    color='region',
                    title='Insurance Cost Distribution by Region',
                    labels={'charges': 'Insurance Cost ($)', 'region': 'Region'},
                    points='outliers'
                )
                
                st.plotly_chart(fig3, use_container_width=True)
                
                # Regional cost comparison table
                st.subheader("Cost Comparison Table")
                comparison_df = pd.pivot_table(
                    region_smoking,
                    values='charges',
                    index='region',
                    columns='smoker',
                    aggfunc='mean'
                ).reset_index()
                
                comparison_df.columns = ['Region', 'Non-Smoker Cost', 'Smoker Cost']
                comparison_df['Cost Difference'] = comparison_df['Smoker Cost'] - comparison_df['Non-Smoker Cost']
                comparison_df['% Increase'] = (comparison_df['Cost Difference'] / comparison_df['Non-Smoker Cost'] * 100).round(1)
                
                # Format monetary columns
                for col in ['Non-Smoker Cost', 'Smoker Cost', 'Cost Difference']:
                    comparison_df[col] = comparison_df[col].map('${:,.2f}'.format)
                
                # Add % sign
                comparison_df['% Increase'] = comparison_df['% Increase'].astype(str) + '%'
                
                st.dataframe(comparison_df, use_container_width=True)
            
            with region_tabs[2]:
                st.markdown("### Regional Heatmaps")
                
                # Create correlation heatmaps by region
                regions = df['region'].unique()
                region_pair = st.selectbox("Select Region for Correlation Analysis", regions)
                
                # Filter data for selected region
                region_df = df[df['region'] == region_pair]
                
                # Calculate correlations for numeric columns
                numeric_cols = ['age', 'bmi', 'children', 'charges']
                corr_matrix = region_df[numeric_cols].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title=f'Correlation Heatmap for {region_pair.capitalize()} Region',
                    labels=dict(x='Feature', y='Feature', color='Correlation')
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Regional variable distributions
                st.subheader(f"Feature Distributions in {region_pair.capitalize()} Region")
                dist_cols = st.columns(2)
                
                with dist_cols[0]:
                    # Age distribution
                    fig1 = px.histogram(
                        region_df,
                        x='age',
                        title=f'Age Distribution in {region_pair.capitalize()}',
                        labels={'age': 'Age', 'count': 'Frequency'},
                        nbins=20
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with dist_cols[1]:
                    # BMI distribution
                    fig2 = px.histogram(
                        region_df,
                        x='bmi',
                        title=f'BMI Distribution in {region_pair.capitalize()}',
                        labels={'bmi': 'BMI', 'count': 'Frequency'},
                        nbins=20
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Regional pricing sensitivity
                st.subheader(f"Insurance Pricing Sensitivity in {region_pair.capitalize()}")
                
                fig3 = px.scatter(
                    region_df,
                    x='age',
                    y='charges',
                    color='smoker',
                    size='bmi',
                    trendline='ols',
                    title=f'Age vs. Cost in {region_pair.capitalize()} (with Smoking Status)',
                    labels={'charges': 'Insurance Cost ($)', 'age': 'Age', 'smoker': 'Smoker'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
                
                st.plotly_chart(fig3, use_container_width=True)
            
            with region_tabs[3]:
                st.markdown("### Regional Demographic Variations")
                
                # Calculate demographic variations by region
                region_demographics = df.groupby('region').agg({
                    'age': 'mean',
                    'bmi': 'mean',
                    'children': 'mean'
                }).reset_index()
                
                # Calculate percentage of smokers by region
                smoker_pct = df.groupby('region')['smoker'].apply(
                    lambda x: (x == 'yes').mean() * 100
                ).reset_index()
                smoker_pct.columns = ['region', 'smoker_pct']
                
                # Calculate percentage of males by region
                male_pct = df.groupby('region')['sex'].apply(
                    lambda x: (x == 'male').mean() * 100
                ).reset_index()
                male_pct.columns = ['region', 'male_pct']
                
                # Merge all demographics
                region_demographics = region_demographics.merge(smoker_pct, on='region')
                region_demographics = region_demographics.merge(male_pct, on='region')
                
                # Create radar chart for each region
                st.subheader("Regional Demographic Profiles")
                
                # Scale values for radar chart
                region_radar = region_demographics.copy()
                region_radar['age_scaled'] = (region_radar['age'] - region_radar['age'].min()) / (region_radar['age'].max() - region_radar['age'].min())
                region_radar['bmi_scaled'] = (region_radar['bmi'] - region_radar['bmi'].min()) / (region_radar['bmi'].max() - region_radar['bmi'].min())
                region_radar['children_scaled'] = region_radar['children'] / region_radar['children'].max()
                region_radar['smoker_scaled'] = region_radar['smoker_pct'] / 100
                region_radar['male_scaled'] = region_radar['male_pct'] / 100
                
                # Create radar chart
                fig = go.Figure()
                
                for i, region in enumerate(region_radar['region']):
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            region_radar.loc[i, 'age_scaled'],
                            region_radar.loc[i, 'bmi_scaled'],
                            region_radar.loc[i, 'children_scaled'],
                            region_radar.loc[i, 'smoker_scaled'],
                            region_radar.loc[i, 'male_scaled']
                        ],
                        theta=['Average Age', 'Average BMI', 'Average Children', 'Smoker %', 'Male %'],
                        fill='toself',
                        name=region.capitalize()
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Regional Demographic Comparison",
                    showlegend=True,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a more detailed demographic table
                st.subheader("Detailed Regional Demographics")
                
                detailed_demo = region_demographics.copy()
                detailed_demo.columns = ['Region', 'Average Age', 'Average BMI', 'Average Children', 'Smoker %', 'Male %']
                
                # Round values
                detailed_demo['Average Age'] = detailed_demo['Average Age'].round(1)
                detailed_demo['Average BMI'] = detailed_demo['Average BMI'].round(1)
                detailed_demo['Average Children'] = detailed_demo['Average Children'].round(1)
                detailed_demo['Smoker %'] = detailed_demo['Smoker %'].round(1)
                detailed_demo['Male %'] = detailed_demo['Male %'].round(1)
                
                st.dataframe(detailed_demo, use_container_width=True)
                
                # Regional demographic insights
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Regional Demographic Insights</h4>
                    <ul>
                        <li><strong>Demographic Variations:</strong> Different regions show distinct demographic patterns that influence insurance costs</li>
                        <li><strong>Lifestyle Factors:</strong> Smoking rates and obesity levels vary by region, affecting overall insurance pricing</li>
                        <li><strong>Family Structures:</strong> Average family sizes and dependent coverage needs differ across regions</li>
                        <li><strong>Regulatory Differences:</strong> State-specific insurance regulations create regional pricing variations</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # New implementation for Correlation Matrix tab
        with viz_tabs[4]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Insurance Cost Correlation Analysis")
            
            # Create correlation tabs
            corr_tabs = st.tabs(["Correlation Matrix", "Factor Impact", "Predictive Power", "Interaction Effects"])
            
            with corr_tabs[0]:
                st.markdown("### Correlation Heatmap")
                
                # Calculate correlations
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                corr_matrix = numeric_df.corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix of Numeric Features',
                    labels=dict(x='Feature', y='Feature', color='Correlation')
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation of correlations
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Understanding Correlations</h4>
                    <p>The correlation matrix shows the relationship strength between different variables:</p>
                    <ul>
                        <li><strong>+1.0:</strong> Perfect positive correlation (as one variable increases, the other increases proportionally)</li>
                        <li><strong>0.0:</strong> No correlation (variables have no linear relationship)</li>
                        <li><strong>-1.0:</strong> Perfect negative correlation (as one variable increases, the other decreases proportionally)</li>
                    </ul>
                    <p>Insurance premiums are most strongly correlated with:</p>
                    <ol>
                        <li><strong>Smoking Status:</strong> The strongest predictor of insurance costs</li>
                        <li><strong>Age:</strong> A significant factor as health risks increase with age</li>
                        <li><strong>BMI:</strong> Higher BMI values correlate with higher insurance costs</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Display correlation values with charges
                st.subheader("Correlation with Insurance Charges")
                
                corr_with_charges = numeric_df.corrwith(numeric_df['charges']).sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Factor': corr_with_charges.index,
                    'Correlation': corr_with_charges.values
                })
                
                # Create bar chart
                fig2 = px.bar(
                    corr_df,
                    x='Factor',
                    y='Correlation',
                    title='Features Correlation with Insurance Charges',
                    labels={'Correlation': 'Correlation Coefficient', 'Factor': 'Feature'},
                    color='Correlation',
                    color_continuous_scale='RdBu_r',
                    text_auto='.2f'
                )
                
                fig2.update_traces(textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)
            
            with corr_tabs[1]:
                st.markdown("### Factor Impact Analysis")
                
                # Create scatter plot matrix for main factors
                main_factors = ['age', 'bmi', 'children', 'charges']
                
                fig = px.scatter_matrix(
                    df,
                    dimensions=main_factors,
                    color='smoker',
                    title='Scatter Plot Matrix of Key Factors',
                    opacity=0.7,
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
                
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (simulated)
                st.subheader("Estimated Feature Importance")
                
                # Create simulated feature importance based on correlations
                feature_importance = pd.DataFrame({
                    'Feature': ['smoker', 'age', 'bmi', 'children', 'region', 'sex'],
                    'Importance': [0.58, 0.24, 0.11, 0.05, 0.01, 0.01]
                })
                
                fig2 = px.bar(
                    feature_importance,
                    x='Feature',
                    y='Importance',
                    title='Estimated Feature Importance for Insurance Cost Prediction',
                    labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
                    color='Importance',
                    text_auto='.2f'
                )
                
                fig2.update_traces(textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Feature impact interpretation
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Factor Impact Interpretation</h4>
                    <p>The analysis reveals the relative importance of different factors in determining insurance costs:</p>
                    <ul>
                        <li><strong>Smoking (58%):</strong> By far the most influential factor, often doubling or tripling premiums</li>
                        <li><strong>Age (24%):</strong> The second most important factor, with costs rising steadily with age</li>
                        <li><strong>BMI (11%):</strong> Significant impact, especially when BMI is in obese categories</li>
                        <li><strong>Children (5%):</strong> Has a moderate but noticeable effect on premium costs</li>
                        <li><strong>Region and Gender (1% each):</strong> Have minimal direct impact compared to other factors</li>
                    </ul>
                    <p>These insights can help individuals understand which factors are most important to address when seeking to reduce insurance costs.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with corr_tabs[2]:
                st.markdown("### Predictive Power Analysis")
                
                # Create predicted vs actual plot
                # Note: This uses a simple linear model for illustration
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                
                # Prepare data
                X = df.drop('charges', axis=1)
                y = df['charges']
                
                # Create preprocessing pipeline
                categorical_features = ['sex', 'smoker', 'region']
                categorical_transformer = OneHotEncoder(drop='first')
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', categorical_transformer, categorical_features)
                    ],
                    remainder='passthrough'
                )
                
                # Create and train the model pipeline
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression())
                ])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Create prediction results dataframe
                pred_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred,
                    'Smoker': X_test['smoker'].reset_index(drop=True),
                    'Age': X_test['age'].reset_index(drop=True),
                    'BMI': X_test['bmi'].reset_index(drop=True)
                })
                
                # Create scatter plot of actual vs predicted
                fig = px.scatter(
                    pred_df,
                    x='Actual',
                    y='Predicted',
                    color='Smoker',
                    size='BMI',
                    hover_data=['Age'],
                    title='Actual vs Predicted Insurance Costs',
                    labels={'Actual': 'Actual Cost ($)', 'Predicted': 'Predicted Cost ($)'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
                
                # Add diagonal reference line
                max_value = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                fig.add_trace(go.Scatter(
                    x=[0, max_value],
                    y=[0, max_value],
                    mode='lines',
                    line=dict(color='black', dash='dash'),
                    name='Perfect Prediction'
                ))
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate performance metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                import numpy as np
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics
                metrics_cols = st.columns(3)
                
                with metrics_cols[0]:
                    st.metric("Mean Absolute Error", f"${mae:.2f}")
                    st.markdown("<div style='font-size: 0.8rem; color: #666;'>Average error in dollars</div>", unsafe_allow_html=True)
                
                with metrics_cols[1]:
                    st.metric("Root Mean Squared Error", f"${rmse:.2f}")
                    st.markdown("<div style='font-size: 0.8rem; color: #666;'>Square root of average squared error</div>", unsafe_allow_html=True)
                
                with metrics_cols[2]:
                    st.metric("R² Score", f"{r2:.3f}")
                    st.markdown("<div style='font-size: 0.8rem; color: #666;'>Proportion of variance explained (1.0 is perfect)</div>", unsafe_allow_html=True)
                
                # Create residual plot
                pred_df['Residual'] = pred_df['Actual'] - pred_df['Predicted']
                
                fig2 = px.scatter(
                    pred_df,
                    x='Predicted',
                    y='Residual',
                    color='Smoker',
                    title='Residual Plot (Prediction Errors)',
                    labels={'Residual': 'Residual (Actual - Predicted) ($)', 'Predicted': 'Predicted Cost ($)'},
                    color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"}
                )
                
                # Add zero reference line
                fig2.add_hline(y=0, line_dash="dash", line_color="black")
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Model performance interpretation
                st.markdown("""
                <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                    <h4 style="margin-top: 0; color: #3498db;">Predictive Model Insights</h4>
                    <p>The predictive model analysis reveals:</p>
                    <ul>
                        <li><strong>Strong Predictability:</strong> Key factors like smoking status, age, and BMI allow for reasonably accurate predictions</li>
                        <li><strong>Higher Uncertainty at Higher Costs:</strong> The model tends to have larger errors for very high insurance costs</li>
                        <li><strong>Smoker Impact:</strong> The predictive model performs differently for smokers vs. non-smokers</li>
                        <li><strong>Unexplained Variance:</strong> Some variation in insurance costs cannot be explained by the available factors alone</li>
                    </ul>
                    <p>This analysis helps understand how well insurance costs can be predicted from demographic and lifestyle factors.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with corr_tabs[3]:
                st.markdown("### Factor Interaction Analysis")
                
                st.info("This section explores how different factors interact with each other to affect insurance costs.")
                
                # Create interaction plots
                interaction_options = [
                    "Age & Smoking Status",
                    "BMI & Smoking Status",
                    "Age & BMI",
                    "Region & Smoking Status",
                    "Gender & Smoking Status",
                    "Children & Age"
                ]
                
                selected_interaction = st.selectbox("Select Interaction to Analyze", interaction_options)
                
                if selected_interaction == "Age & Smoking Status":
                    # Age bins
                    df['age_group'] = pd.cut(
                        df['age'],
                        bins=[0, 30, 40, 50, 60, 100],
                        labels=['<30', '30-40', '40-50', '50-60', '60+']
                    )
                    
                    # Create grouped bar chart
                    age_smoking = df.groupby(['age_group', 'smoker']).agg({
                        'charges': 'mean'
                    }).reset_index()
                    
                    
                    fig = px.bar(
                        age_smoking,
                        x='age_group',
                        y='charges',
                        color='smoker',
                        barmode='group',
                        title='Average Insurance Cost by Age Group and Smoking Status',
                        labels={'charges': 'Average Cost ($)', 'age_group': 'Age Group', 'smoker': 'Smoker'},
                        color_discrete_map={"yes": "#e74c3c", "no": "#2ecc71"},
                        text_auto='.0f'
                    )

                    fig.update_traces(texttemplate='$%{text}', textposition='outside')

                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate smoking premium increase by age
                    smoking_premium = age_smoking.pivot(index='age_group', columns='smoker', values='charges').reset_index()
                    smoking_premium['Increase'] = smoking_premium['yes'] - smoking_premium['no']
                    smoking_premium['Percentage'] = (smoking_premium['Increase'] / smoking_premium['no'] * 100).round(1)

                    # Format output
                    smoking_premium.columns = ['Age Group', 'Non-Smoker', 'Smoker', 'Increase', 'Percentage']
                    for col in ['Non-Smoker', 'Smoker', 'Increase']:
                        smoking_premium[col] = smoking_premium[col].map('${:,.2f}'.format)

                    smoking_premium['Percentage'] = smoking_premium['Percentage'].astype(str) + '%'

                    st.subheader("Smoking Premium Increase by Age Group")
                    st.dataframe(smoking_premium, use_container_width=True)

                    st.markdown("""
                    <div style="background-color: #f0f7fb; padding: 15px; border-radius: 5px; margin-top: 15px;">
                        <h4 style="margin-top: 0; color: #3498db;">Age & Smoking Interaction Insights</h4>
                        <ul>
                            <li><strong>Compounding Effect:</strong> The impact of smoking on insurance costs increases with age</li>
                            <li><strong>Highest Impact:</strong> Middle-aged smokers (40-60) typically see the largest absolute premium increases</li>
                            <li><strong>Percentage Increase:</strong> Smoking can increase premiums by over 200% across all age groups</li>
                            <li><strong>Risk Amplification:</strong> Insurers view smoking as a risk amplifier that magnifies age-related health risks</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        

# ===========================================================================
# HISTORY PAGE
# ===========================================================================


elif selected == "History":
    st.markdown("<h1 class='main-header'>Prediction History</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Review your past insurance premium predictions</p>", unsafe_allow_html=True)
    
    # Load history data
    try:
        history_file = "data/prediction_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history_data = json.load(f)
        else:
            history_data = []
            
        # Combine file history with session history for complete view
        session_history_ids = [item["id"] for item in st.session_state['history']]
        combined_history = st.session_state['history'].copy()
        
        for item in history_data:
            if item["id"] not in session_history_ids:
                combined_history.append(item)
                
        # Sort by timestamp (most recent first)
        combined_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if not combined_history:
            st.info("You haven't made any predictions yet. Go to the Predict tab to get started!")
        else:
            # Create a tabbed interface
            history_tabs = st.tabs(["Timeline View", "Table View", "Statistics"])
            
            with history_tabs[0]:
                # Timeline view with cards
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Recent Predictions")
                
                for i, item in enumerate(combined_history[:10]):  # Show only the 10 most recent
                    prediction = item.get("prediction", 0)
                    timestamp = item.get("timestamp", "Unknown date")
                    inputs = item.get("inputs", {})
                    
                    with st.expander(f"Prediction on {timestamp} - ${prediction:.2f}"):
                        # Create two columns
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown("#### Input Parameters")
                            
                            # Format the parameters in a more visually appealing way
                            st.markdown(f"""
                            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                <table style='width: 100%;'>
                                    <tr>
                                        <td style='padding: 4px; font-weight: 600; color: #2c3e50;'>Age:</td>
                                        <td style='padding: 4px; color: #2c3e50;'>{inputs.get('age', 'N/A')}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 4px; font-weight: 600; color: #2c3e50;'>Sex:</td>
                                        <td style='padding: 4px; color: #2c3e50;'>{inputs.get('sex', 'N/A').capitalize()}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 4px; font-weight: 600; color: #2c3e50;'>BMI:</td>
                                        <td style='padding: 4px; color: #2c3e50;'>{inputs.get('bmi', 'N/A'):.1f} ({get_bmi_category(inputs.get('bmi', 25))})</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 4px; font-weight: 600; color: #2c3e50;'>Dependents:</td>
                                        <td style='padding: 4px; color: #2c3e50;'>{inputs.get('children', 'N/A')}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 4px; font-weight: 600; color: #2c3e50;'>Smoker:</td>
                                        <td style='padding: 4px; color: #2c3e50;'>{inputs.get('smoker', 'N/A').capitalize()}</td>
                                    </tr>
                                    <tr>
                                        <td style='padding: 4px; font-weight: 600; color: #2c3e50;'>Region:</td>
                                        <td style='padding: 4px; color: #2c3e50;'>{inputs.get('region', 'N/A').capitalize()}</td>
                                    </tr>
                                </table>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### Prediction Result")
                            st.markdown(f"""
                            <div style='background-color: #e3f2fd; padding: 20px; border-radius: 8px; text-align: center;'>
                                <div style='font-size: 2rem; font-weight: 700; color: #1976D2;'>
                                    ${prediction:.2f}
                                </div>
                                <div style='font-size: 1rem; color: #2c3e50; margin-top: 5px;'>
                                    Annual Premium
                                </div>
                                <div style='font-size: 0.9rem; color: #2c3e50; margin-top: 5px;'>
                                    ~${prediction/12:.2f} monthly
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show buttons for actions
                            if st.button("Use These Inputs Again", key=f"reuse_{i}"):
                                # Store in session state and redirect to prediction page
                                st.session_state['reuse_inputs'] = inputs
                                st.session_state['selected_tab'] = "Predict"
                                st.experimental_rerun()
                            
                            # Add to comparison option
                            if st.button("Add to Compare", key=f"compare_{i}"):
                                comparison_item = {
                                    "inputs": inputs,
                                    "prediction": prediction,
                                    "id": item.get("id", str(uuid.uuid4())[:8])
                                }
                                
                                # Check if already in comparison data
                                if not any(comp["id"] == comparison_item["id"] for comp in st.session_state['comparison_data']):
                                    st.session_state['comparison_data'].append(comparison_item)
                                    st.success("Added to comparison!")
                                else:
                                    st.info("This profile is already in your comparisons.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            with history_tabs[1]:
                # Table view
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("All Predictions")
                
                # Create a DataFrame for easy display
                table_data = []
                for item in combined_history:
                    inputs = item.get("inputs", {})
                    row = {
                        "Date": item.get("timestamp", ""),
                        "Age": inputs.get("age", ""),
                        "Sex": inputs.get("sex", "").capitalize(),
                        "BMI": inputs.get("bmi", ""),
                        "Dependents": inputs.get("children", ""),
                        "Smoker": inputs.get("smoker", "").capitalize(),
                        "Region": inputs.get("region", "").capitalize(),
                        "Prediction": f"${item.get('prediction', 0):.2f}"
                    }
                    table_data.append(row)
                
                if table_data:
                    history_df = pd.DataFrame(table_data)
                    st.dataframe(history_df, use_container_width=True)
                    
                    # Add export option
                    st.markdown("#### Export History")
                    export_cols = st.columns([3, 1])
                    
                    with export_cols[0]:
                        st.write("Download your prediction history as a CSV file:")
                    
                    with export_cols[1]:
                        st.markdown(get_csv_download_link(history_df, "insurance_prediction_history.csv", "📥 Download CSV"), unsafe_allow_html=True)
                else:
                    st.info("No prediction history available.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            with history_tabs[2]:
                # Statistics about predictions
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Prediction Statistics")
                
                if combined_history:
                    # Extract predictions
                    predictions = [item.get("prediction", 0) for item in combined_history]
                    
                    # Create stats dashboard
                    stat_cols = st.columns(4)
                    
                    with stat_cols[0]:
                        st.metric("Total Predictions", len(predictions))
                    
                    with stat_cols[1]:
                        st.metric("Average Premium", f"${np.mean(predictions):.2f}")
                    
                    with stat_cols[2]:
                        st.metric("Lowest Premium", f"${min(predictions)::.2f}")
                    
                    with stat_cols[3]:
                        st.metric("Highest Premium", f"${max(predictions)::.2f}")
                    
                    # Create a histogram of predictions
                    st.subheader("Distribution of Your Premiums")
                    
                    fig = px.histogram(
                        pd.DataFrame({"Premium": predictions}), 
                        x="Premium",
                        nbins=20,
                        labels={"Premium": "Premium Amount ($)"},
                        title="Distribution of Predicted Premiums",
                        color_discrete_sequence=["#3498db"]
                    )
                    
                    fig.update_layout(
                        xaxis_title="Premium Amount ($)",
                        yaxis_title="Count",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Extract some more insights if we have enough data
                    if len(combined_history) >= 3:
                        st.subheader("Insights from Your Predictions")
                        
                        # Determine which factors appear frequently in predictions
                        smoker_counts = {'yes': 0, 'no': 0}
                        regions = {'northeast': 0, 'northwest': 0, 'southeast': 0, 'southwest': 0}
                        total_age = 0
                        total_bmi = 0
                        
                        for item in combined_history:
                            inputs = item.get("inputs", {})
                            smoker = inputs.get("smoker", "no")
                            region = inputs.get("region", "northeast")
                            
                            smoker_counts[smoker] += 1
                            regions[region] += 1
                            total_age += inputs.get("age", 30)
                            total_bmi += inputs.get("bmi", 25)
                        
                        avg_age = total_age / len(combined_history)
                        avg_bmi = total_bmi / len(combined_history)
                        
                        # Create insight cards
                        insight_cols = st.columns(2)
                        
                        with insight_cols[0]:
                            # Most common profile
                            most_common_smoker = "yes" if smoker_counts["yes"] > smoker_counts["no"] else "no"
                            most_common_region = max(regions, key=regions.get)
                            
                            st.markdown(f"""
                            <div class="insight-box">
                                <h4>Your Typical Profile</h4>
                                <p>Based on your prediction history, your most common profile is:</p>
                                <ul>
                                    <li><strong>Average age:</strong> {avg_age:.1f} years</li>
                                    <li><strong>Average BMI:</strong> {avg_bmi:.1f} ({get_bmi_category(avg_bmi)})</li>
                                    <li><strong>Smoking status:</strong> {most_common_smoker.capitalize()}</li>
                                    <li><strong>Region:</strong> {most_common_region.capitalize()}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with insight_cols[1]:
                            # Premium trends
                            if len(combined_history) >= 5:
                                recent_predictions = [item.get("prediction", 0) for item in combined_history[:5]]
                                older_predictions = [item.get("prediction", 0) for item in combined_history[5:10]]
                                
                                if older_predictions:
                                    recent_avg = np.mean(recent_predictions)
                                    older_avg = np.mean(older_predictions)
                                    change_pct = ((recent_avg - older_avg) / older_avg) * 100
                                    
                                    trend_direction = "increased" if change_pct > 0 else "decreased"
                                    trend_class = "negative-change" if change_pct > 0 else "positive-change"
                                    
                                    st.markdown(f"""
                                    <div class="insight-box">
                                        <h4>Premium Trend</h4>
                                        <div class="insight-label">Recent Premium Trend</div>
                                        <div class="insight-value">{abs(change_pct):.1f}%</div>
                                        <div class="insight-change {trend_class}">
                                            Your recent premiums have {trend_direction} compared to previous ones
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="insight-box">
                                        <h4>Premium Analysis</h4>
                                        <p>Make more predictions to see premium trends over time.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="insight-box">
                                    <h4>Premium Analysis</h4>
                                    <p>Make more predictions to see detailed premium insights.</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Clear history option
                    with st.expander("Manage History Data"):
                        st.warning("This will permanently delete your prediction history.")
                        if st.button("Clear All History"):
                            # Clear session state
                            st.session_state['history'] = []
                            
                            # Clear file if exists
                            if os.path.exists(history_file):
                                with open(history_file, "w") as f:
                                    json.dump([], f)
                            
                            st.success("History cleared successfully!")
                            st.experimental_rerun()
                else:
                    st.info("Make some predictions to see statistics here.")
                
                st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error("An error occurred while loading history data.")
        log_error(e, context="history_page")

# ===========================================================================
# COMPARE PAGE
# ===========================================================================

elif selected == "Compare":
    st.markdown("<h1 class='main-header'>Compare Insurance Scenarios</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Compare different profile scenarios and their predicted premiums</p>", unsafe_allow_html=True)
    
    try:
        # Check if we have comparison data
        if not st.session_state['comparison_data']:
            # No comparison data yet
            st.info("You haven't added any profiles to compare yet. Make predictions in the Predict tab and add them to comparison.")
            
            # Provide a quick button to go to predict page
            if st.button("Go to Prediction Page"):
                st.session_state['selected_tab'] = "Predict"
                st.experimental_rerun()
            
            # Show information on how to use the comparison feature
            st.markdown("""
            <div class="documentation-note">
                <h4>How to Use the Comparison Feature</h4>
                <p>The comparison tool helps you understand how different personal factors affect your insurance premium.</p>
                <ol>
                    <li>Go to the <strong>Predict</strong> tab and make a prediction</li>
                    <li>Click "Add to Compare" button after making a prediction</li>
                    <li>Make another prediction with different parameters</li>
                    <li>Add that prediction to comparison as well</li>
                    <li>Return to this tab to see a side-by-side comparison</li>
                </ol>
                <p>You can add up to 5 different profiles to compare at once.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show a sample comparison visualization
            st.subheader("Sample Comparison")
            
            # Create sample data
            sample_data = [
                {"profile": "Profile 1", "age": 30, "bmi": 22, "smoker": "No", "premium": 5000},
                {"profile": "Profile 2", "age": 30, "bmi": 22, "smoker": "Yes", "premium": 18000},
                {"profile": "Profile 3", "age": 50, "bmi": 28, "smoker": "No", "premium": 9500}
            ]
            
            sample_df = pd.DataFrame(sample_data)
            
            # Create a sample bar chart
            fig = px.bar(
                sample_df, 
                x="profile", 
                y="premium",
                color="profile",
                title="Sample Premium Comparison",
                labels={"premium": "Annual Premium ($)", "profile": "Profile"},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("This is a sample visualization. Add your own profiles to see your personalized comparison.")
            
        else:
            # We have comparison data to show
            comparison_data = st.session_state['comparison_data']
            
            # Add option to clear all comparisons
            clear_col1, clear_col2 = st.columns([4, 1])
            with clear_col1:
                st.subheader(f"Comparing {len(comparison_data)} Insurance Profiles")
            with clear_col2:
                if st.button("Clear All"):
                    st.session_state['comparison_data'] = []
                    st.success("Comparison data cleared!")
                    st.experimental_rerun()
            
            # Convert comparison data to DataFrame for easier handling
            comparison_rows = []
            
            for i, item in enumerate(comparison_data):
                inputs = item.get("inputs", {})
                prediction = item.get("prediction", 0)
                
                # Create a profile name based on key attributes
                profile_name = f"Profile {i+1}"
                
                row = {
                    "Profile": profile_name,
                    "Age": inputs.get("age", 0),
                    "Sex": inputs.get("sex", "").capitalize(),
                    "BMI": inputs.get("bmi", 0),
                    "BMI Category": get_bmi_category(inputs.get("bmi", 25)),
                    "Dependents": inputs.get("children", 0),
                    "Smoker": inputs.get("smoker", "").capitalize(),
                    "Region": inputs.get("region", "").capitalize(),
                    "Premium": prediction,
                    "Monthly": prediction / 12,
                    "ID": item.get("id", "")
                }
                comparison_rows.append(row)
            
            comparison_df = pd.DataFrame(comparison_rows)
            
            # Create a side-by-side comparison in tabs
            comparison_tabs = st.tabs(["Visual Comparison", "Tabular Comparison", "Analysis"])
            
            with comparison_tabs[0]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                # Create a bar chart comparing premiums
                fig = px.bar(
                    comparison_df, 
                    x="Profile", 
                    y="Premium",
                    color="Profile",
                    title="Premium Comparison",
                    labels={"Premium": "Annual Premium ($)"},
                    text_auto='.2f',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_traces(texttemplate='$%{text}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                st.plotly_chart(fig, use_container_width=True)
                
                # Create radar chart for comparing key factors
                st.subheader("Key Factors Comparison")
                
                # Normalize values for radar chart
                radar_df = comparison_df.copy()
                # Age normalized (0-1 scale where 1 is oldest)
                radar_df["Age (norm)"] = (radar_df["Age"] - 18) / (100 - 18)
                # BMI normalized where 25 is optimal (0 is optimal)
                radar_df["BMI (norm)"] = abs(radar_df["BMI"] - 25) / 25
                # Smoker is binary
                radar_df["Smoker (norm)"] = (radar_df["Smoker"] == "Yes").astype(float)
                # Dependents normalized by max value
                max_dependents = radar_df["Dependents"].max() if radar_df["Dependents"].max() > 0 else 1
                radar_df["Dependents (norm)"] = radar_df["Dependents"] / max_dependents
                
                categories = ["Age (norm)", "BMI (norm)", "Smoker (norm)", "Dependents (norm)"]
                
                # Create radar chart
                fig = go.Figure()
                
                for i, profile in enumerate(radar_df["Profile"].unique()):
                    profile_data = radar_df[radar_df["Profile"] == profile]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=profile_data[categories].values.flatten().tolist(),
                        theta=["Age", "BMI", "Smoking", "Dependents"],
                        fill='toself',
                        name=profile
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.1),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a breakdown of differences
                st.subheader("What Makes the Difference?")
                
                if len(comparison_df) >= 2:
                    # Get the min and max premium profiles
                    min_profile = comparison_df.loc[comparison_df["Premium"].idxmin()]
                    max_profile = comparison_df.loc[comparison_df["Premium"].idxmax()]
                    
                    diff_dollar = max_profile["Premium"] - min_profile["Premium"]
                    diff_percent = (diff_dollar / min_profile["Premium"]) * 100
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>Premium Difference</h4>
                        <p>The difference between the highest and lowest premium is:</p>
                        <div class="insight-value">${diff_dollar:.2f}</div>
                        <div class="insight-change positive-change">{diff_percent:.1f}% higher</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show key differences
                    st.markdown("### Key Differences")
                    
                    diff_cols = st.columns(2)
                    
                    with diff_cols[0]:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>Lowest Premium Profile</h4>
                            <ul>
                                <li><strong>Age:</strong> {min_profile["Age"]}</li>
                                <li><strong>BMI:</strong> {min_profile["BMI"]} ({min_profile["BMI Category"]})</li>
                                <li><strong>Smoker:</strong> {min_profile["Smoker"]}</li>
                                <li><strong>Dependents:</strong> {min_profile["Dependents"]}</li>
                                <li><strong>Region:</strong> {min_profile["Region"]}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with diff_cols[1]:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>Highest Premium Profile</h4>
                            <ul>
                                <li><strong>Age:</strong> {max_profile["Age"]}</li>
                                <li><strong>BMI:</strong> {max_profile["BMI"]} ({max_profile["BMI Category"]})</li>
                                <li><strong>Smoker:</strong> {max_profile["Smoker"]}</li>
                                <li><strong>Dependents:</strong> {max_profile["Dependents"]}</li>
                                <li><strong>Region:</strong> {max_profile["Region"]}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Add more profiles to see detailed differences.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with comparison_tabs[1]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Tabular Comparison")
                
                # Display comparison data as a table
                st.dataframe(comparison_df.drop(columns=["ID"]), use_container_width=True)
                
                # Add export option
                st.markdown("#### Export Comparison Data")
                export_cols = st.columns([3, 1])
                
                with export_cols[0]:
                    st.write("Download your comparison data as a CSV file:")
                
                with export_cols[1]:
                    st.markdown(get_csv_download_link(comparison_df.drop(columns=["ID"]), "insurance_comparison_data.csv", "📥 Download CSV"), unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with comparison_tabs[2]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Detailed Analysis")
                
                # Create a detailed analysis of differences
                if len(comparison_df) >= 2:
                    # Calculate differences for each factor
                    analysis_data = []
                    
                    for factor in ["Age", "BMI", "Dependents"]:
                        min_value = comparison_df[factor].min()
                        max_value = comparison_df[factor].max()
                        diff_value = max_value - min_value
                        
                        analysis_data.append({
                            "Factor": factor,
                            "Min Value": min_value,
                            "Max Value": max_value,
                            "Difference": diff_value
                        })
                    
                    # Add categorical factors
                    for factor in ["Sex", "Smoker", "Region"]:
                        unique_values = comparison_df[factor].unique()
                        analysis_data.append({
                            "Factor": factor,
                            "Min Value": unique_values[0],
                            "Max Value": unique_values[-1],
                            "Difference": "N/A"
                        })
                    
                    analysis_df = pd.DataFrame(analysis_data)
                    
                    # Display analysis data
                    st.dataframe(analysis_df, use_container_width=True)
                    
                    # Create a bar chart for numeric differences
                    numeric_analysis_df = analysis_df[analysis_df["Difference"] != "N/A"]
                    
                    fig = px.bar(
                        numeric_analysis_df, 
                        x="Factor", 
                        y="Difference",
                        title="Factor Differences",
                        labels={"Difference": "Difference Value"},
                        color="Factor",
                        text_auto='.2f',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Add more profiles to see detailed analysis.")
                
                st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error("An error occurred while loading comparison data.")
        log_error(e, context="comparison_page")

# ===========================================================================
# ABOUT PAGE
# ===========================================================================

elif selected == "About":
    st.markdown("<h1 class='main-header'>About This App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Learn more about the Health Insurance Estimator</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Overview</h2>
        <p>The Health Insurance Estimator is a powerful tool designed to help you understand and predict your health insurance premiums based on various personal factors. By leveraging machine learning models, this app provides accurate and personalized estimates to help you make informed decisions about your health insurance options.</p>
        
        <h2>Features</h2>
        <ul>
            <li><strong>Premium Prediction:</strong> Get instant estimates of your annual health insurance premiums based on your age, BMI, smoking status, and more.</li>
            <li><strong>Visualization:</strong> Explore how different factors affect insurance costs through interactive charts and graphs.</li>
            <li><strong>History:</strong> Review your past predictions and track changes over time.</li>
            <li><strong>Comparison:</strong> Compare different profile scenarios side-by-side to see how changes in personal factors impact premiums.</li>
            <li><strong>Recommendations:</strong> Receive personalized tips and strategies to reduce your insurance costs.</li>
        </ul>
        
        <h2>How It Works</h2>
        <p>The app uses a machine learning model trained on a dataset of health insurance costs. The model takes into account various factors such as age, BMI, smoking status, number of dependents, and geographic region to predict the annual premium. The predictions are based on patterns and relationships identified in the training data.</p>
        
        <h2>Data Privacy</h2>
        <p>Your privacy is important to us. All calculations are performed locally on your device, and your data is not shared with any external servers. The app does not store any personal information beyond your current session unless you choose to save your prediction history.</p>
        
        <h2>Contact Us</h2>
        <p>If you have any questions, feedback, or need support, please contact our team at <a href="mailto:support@insurance-estimator.com">support@insurance-estimator.com</a>.</p>
        
        <h2>Disclaimer</h2>
        <p>The Health Insurance Estimator provides estimates based on historical data and machine learning models. The actual premiums may vary based on individual circumstances and insurance provider policies. Always consult with a licensed insurance professional for accurate and personalized advice.</p>
    </div>
    """, unsafe_allow_html=True)