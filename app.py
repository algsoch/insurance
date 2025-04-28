# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
import time
import datetime
import uuid
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
        text-align: center;
    }
    .card {
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .chart-container {
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 15px;
        margin: 10px 0;
    }
    .tooltip {
        color: #616161;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

# Load saved model
@st.cache_resource
def load_model():
    return joblib.load('insurance_model.pkl')

model = load_model()

# Function to save prediction history
def save_prediction(user_inputs, prediction):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_id = str(uuid.uuid4())[:8]
    
    history_item = {
        "id": prediction_id,
        "timestamp": timestamp,
        "inputs": user_inputs,
        "prediction": prediction
    }
    
    st.session_state['history'].append(history_item)
    st.session_state['last_prediction'] = history_item

# Load sample data for visualizations
@st.cache_data
def load_sample_data():
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
        st.error(f"Error loading sample data: {e}")
        # Return minimal fallback data
        return pd.DataFrame({
            'age': [30, 40],
            'bmi': [25, 30],
            'charges': [5000, 10000],
            'smoker': ['no', 'yes']
        })

# Main sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health-insurance.png", width=80)
    st.title("Insurance Predictor")
    
    selected = option_menu(
        menu_title=None,
        options=["Predict", "Visualize", "History", "About"],
        icons=["calculator", "graph-up", "clock-history", "info-circle"],
        default_index=0,
    )
    
    st.markdown("---")
    st.caption("© 2025 Insurance Predictor App")

# Predict page
if selected == "Predict":
    st.markdown("<h1 class='main-header'>Insurance Cost Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Estimate your health insurance premium costs</p>", unsafe_allow_html=True)
    
    # Create columns for form layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Personal Information")
        
        age = st.slider("Age", min_value=18, max_value=100, value=30, help="Age of the primary beneficiary")
        sex = st.radio("Sex", options=["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, 
                             help="Body Mass Index, weight (kg) / (height (m))²")
        
        # Add BMI calculator expander
        with st.expander("BMI Calculator"):
            bmi_col1, bmi_col2 = st.columns(2)
            with bmi_col1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            with bmi_col2:
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
            
            if st.button("Calculate BMI"):
                calculated_bmi = weight / ((height/100) ** 2)
                st.success(f"Your BMI is: {calculated_bmi:.1f}")
                bmi = calculated_bmi
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Additional Factors")
        
        children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.select_slider("Smoking Status", options=["no", "yes"], value="no")
        
        # Add visual indicator for smoking risk
        if smoker == "yes":
            st.warning("⚠️ Smoking significantly increases insurance costs")
        
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
        # Add tooltip information about regions
        st.markdown("<p class='tooltip'>Different regions may have different average costs due to local regulations and healthcare prices.</p>", 
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
        predict_button = st.button("💰 Predict Insurance Cost", use_container_width=True)
    
    # Process prediction when button is clicked
    if predict_button:
        with st.spinner("Calculating your insurance costs..."):
            # Convert categorical variables to numeric
            sex_val = 1 if sex == "male" else 0
            smoker_val = 1 if smoker == "yes" else 0
            
            # Region encoding - assuming model expects one-hot encoding
            region_northeast = 1 if region == "northeast" else 0
            region_northwest = 1 if region == "northwest" else 0
            region_southeast = 1 if region == "southeast" else 0
            region_southwest = 1 if region == "southwest" else 0
            
            # Prepare input for model (adjust according to your model's expected input)
            input_data = np.array([[age, sex_val, bmi, children, smoker_val, 
                                   region_northeast, region_northwest, region_southeast, region_southwest]])
            
            # Show a progress bar to simulate calculation
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            prediction = model.predict(input_data)[0]
            save_prediction(user_inputs, float(prediction))
        
        # Display prediction result
        st.markdown("<div class='prediction-result' style='background-color: #E3F2FD;'>", unsafe_allow_html=True)
        st.subheader("Estimated Annual Insurance Cost:")
        st.markdown(f"<h1 style='color: #1976D2; text-align: center;'>${prediction:.2f}</h1>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add additional context
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Cost Breakdown Analysis")
            
            # Create a simplified explanation of what factors contributed most
            factors = []
            if smoker == "yes":
                factors.append(("Smoking Status", "Very High Impact"))
            if bmi > 30:
                factors.append(("BMI (Obesity)", "High Impact"))
            if age > 50:
                factors.append(("Age", "Moderate Impact"))
            if children > 2:
                factors.append(("Number of Children", "Moderate Impact"))
            
            if not factors:
                factors = [("Overall Profile", "Low Risk")]
            
            for factor, impact in factors:
                st.markdown(f"**{factor}**: {impact}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Recommendations")
            
            if smoker == "yes":
                st.error("Consider quitting smoking to significantly reduce costs")
            
            if bmi > 30:
                st.warning("Improving BMI could lower your insurance costs")
            
            if bmi <= 25 and smoker == "no":
                st.success("Your healthy lifestyle is helping keep costs down!")
            st.markdown("</div>", unsafe_allow_html=True)

# Visualization page
elif selected == "Visualize":
    st.markdown("<h1 class='main-header'>Insurance Cost Trends</h1>", unsafe_allow_html=True)
    
    # Load sample data for visualization
    df = load_sample_data()
    
    # Add tab-based navigation for different visualizations
    viz_tabs = st.tabs(["Age vs Cost", "BMI Impact", "Smoking Impact", "Regional Trends", "Correlation Matrix"])
    
    with viz_tabs[0]:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("How Age Affects Insurance Costs")
        
        fig = px.scatter(df, x="age", y="charges", color="smoker", 
                        size="bmi", hover_data=["sex", "children", "region"],
                        title="Insurance Charges by Age and Smoking Status",
                        color_discrete_map={"yes": "red", "no": "green"})
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("Age is a significant factor in determining insurance costs, with older individuals typically paying more. This visualization shows how age correlates with insurance charges, colored by smoking status.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with viz_tabs[1]:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("BMI Impact on Insurance Costs")
        
        fig = px.box(df, x="smoker", y="charges", color="smoker",
                    facet_col=pd.cut(df["bmi"], bins=[15, 25, 30, 35, 40, 50], 
                                    labels=["Underweight/Normal", "Overweight", "Obese I", "Obese II", "Extreme Obesity"]),
                    title="Insurance Charges by BMI Category and Smoking Status",
                    color_discrete_map={"yes": "red", "no": "green"})
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("BMI has a significant impact on insurance costs, especially in combination with smoking status. Higher BMI categories generally face higher insurance premiums.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with viz_tabs[2]:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Smoking Impact on Costs")
        
        fig = px.histogram(df, x="charges", color="smoker", 
                          nbins=50, opacity=0.7, barmode="overlay",
                          color_discrete_map={"yes": "red", "no": "green"},
                          title="Distribution of Insurance Charges by Smoking Status")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate average costs
        smoker_avg = df[df['smoker'] == 'yes']['charges'].mean()
        non_smoker_avg = df[df['smoker'] == 'no']['charges'].mean()
        difference = smoker_avg - non_smoker_avg
        percent_increase = (difference / non_smoker_avg) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Cost (Smokers)", f"${smoker_avg:.2f}")
        with col2:
            st.metric("Average Cost (Non-Smokers)", f"${non_smoker_avg:.2f}", 
                     delta=f"-${difference:.2f} (-{percent_increase:.1f}%)")
        
        st.markdown("Smoking has one of the most dramatic impacts on insurance costs, often more than doubling premiums compared to non-smokers with similar profiles.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with viz_tabs[3]:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Regional Cost Variations")
        
        fig = px.box(df, x="region", y="charges", color="region",
                    title="Insurance Charges by Region")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show average costs by region
        region_avgs = df.groupby('region')['charges'].mean().reset_index()
        region_avgs = region_avgs.sort_values('charges', ascending=False)
        
        fig2 = px.bar(region_avgs, x='region', y='charges', color='region',
                     title="Average Insurance Cost by Region")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("Insurance costs can vary significantly by region due to differences in healthcare costs, state regulations, and other local factors.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with viz_tabs[4]:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("Correlation Matrix of Factors")
        
        # Create numeric version of the dataframe for correlation
        df_numeric = df.copy()
        df_numeric['sex'] = df_numeric['sex'].map({'male': 1, 'female': 0})
        df_numeric['smoker'] = df_numeric['smoker'].map({'yes': 1, 'no': 0})
        df_numeric = pd.get_dummies(df_numeric, columns=['region'], drop_first=True)
        
        corr = df_numeric.corr()
        
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                       title="Correlation Matrix of Insurance Factors",
                       color_continuous_scale='RdBu_r')
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("This correlation matrix shows how different factors relate to insurance charges. Positive values (closer to 1) indicate that as one factor increases, costs tend to increase as well. Negative values suggest the opposite relationship.")
        st.markdown("</div>", unsafe_allow_html=True)

# History page
elif selected == "History":
    st.markdown("<h1 class='main-header'>Prediction History</h1>", unsafe_allow_html=True)
    
    if not st.session_state['history']:
        st.info("You haven't made any predictions yet. Try making a prediction first!")
    else:
        # Add option to clear history
        if st.button("Clear History"):
            st.session_state['history'] = []
            st.success("History cleared!")
            st.stop()
        
        st.markdown(f"### You have made {len(st.session_state['history'])} predictions")
        
        # Create a dataframe from history for better display
        history_data = []
        for item in st.session_state['history']:
            history_item = {
                "ID": item["id"],
                "Date": item["timestamp"],
                "Age": item["inputs"]["age"],
                "Sex": item["inputs"]["sex"],
                "BMI": item["inputs"]["bmi"],
                "Children": item["inputs"]["children"],
                "Smoker": item["inputs"]["smoker"],
                "Region": item["inputs"]["region"],
                "Predicted Cost": f"${item['prediction']:.2f}"
            }
            history_data.append(history_item)
        
        history_df = pd.DataFrame(history_data)
        
        # Display history as an interactive table
        st.dataframe(history_df.style.highlight_max(subset=['Predicted Cost'], color='#ffb3b3')
                                     .highlight_min(subset=['Predicted Cost'], color='#b3ffb3'),
                     use_container_width=True,
                     column_config={
                         "Date": st.column_config.DatetimeColumn("Date & Time", format="MMM DD, YYYY, hh:mm"),
                         "Predicted Cost": st.column_config.NumberColumn("Predicted Cost", format="$%d")
                     })
        
        # Add visualization of prediction history
        if len(history_data) > 1:
            st.subheader("Your Prediction Trends")
            
            # Extract costs and timestamps for trend visualization
            costs = [item["prediction"] for item in st.session_state['history']]
            times = [item["timestamp"] for item in st.session_state['history']]
            
            trend_data = pd.DataFrame({
                "Timestamp": times,
                "Cost": costs
            })
            
            fig = px.line(trend_data, x="Timestamp", y="Cost", 
                         title="Your Insurance Cost Predictions Over Time",
                         markers=True)
            
            st.plotly_chart(fig, use_container_width=True)

# About page
elif selected == "About":
    st.markdown("<h1 class='main-header'>About This App</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Insurance Cost Prediction Tool
    
    This application uses machine learning to predict health insurance premium costs based on several personal factors:
    
    - **Age**: The primary beneficiary's age
    - **Sex**: The beneficiary's gender
    - **BMI**: Body Mass Index, a measure of body fat based on height and weight
    - **Children**: Number of dependents covered by the insurance
    - **Smoker**: Whether the beneficiary is a smoker
    - **Region**: The beneficiary's residential area
    
    #### How It Works
    
    The prediction model is trained on a dataset containing thousands of insurance records. It learns the relationships between personal factors and the resulting insurance charges, then applies these patterns to predict costs for new individuals.
    
    #### Model Performance
    
    The model provides estimates based on historical data and identified patterns. While it achieves good accuracy for general predictions, actual insurance prices may vary based on additional factors not included in this model.
    
    #### Data Privacy
    
    All predictions are calculated locally in your browser. No personal information is stored or transmitted to external servers. Your prediction history is only saved in your current browser session.
    
    #### Disclaimer
    
    This tool is for educational purposes only and should not be considered as financial advice. Actual insurance costs are determined by insurance providers based on their own criteria.
    """)
    
    # Add sample prediction button
    st.subheader("Try a Sample Prediction")
    
    if st.button("Run Sample Prediction"):
        with st.spinner("Calculating sample prediction..."):
            sample_inputs = {
                "age": 42,
                "sex": "male",
                "bmi": 28.6,
                "children": 2,
                "smoker": "no",
                "region": "northeast"
            }
            
            # Make prediction using sample data
            sex_val = 1 if sample_inputs["sex"] == "male" else 0
            smoker_val = 1 if sample_inputs["smoker"] == "yes" else 0
            
            region_northeast = 1 if sample_inputs["region"] == "northeast" else 0
            region_northwest = 1 if sample_inputs["region"] == "northwest" else 0
            region_southeast = 1 if sample_inputs["region"] == "southeast" else 0
            region_southwest = 1 if sample_inputs["region"] == "southwest" else 0
            
            input_data = np.array([[
                sample_inputs["age"], 
                sex_val, 
                sample_inputs["bmi"], 
                sample_inputs["children"], 
                smoker_val,
                region_northeast, 
                region_northwest, 
                region_southeast, 
                region_southwest
            ]])
            
            time.sleep(1)  # Simulate calculation time
            prediction = model.predict(input_data)[0]
        
        st.success(f"Sample prediction for a 42-year-old male with BMI 28.6, 2 children, non-smoker from Northeast: ${prediction:.2f}")

# Custom footer
st.markdown("---")
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center; padding: 10px;'>
        <p style='font-size: 0.8rem; color: #666;'>Powered by Streamlit</p>
        <p style='font-size: 0.8rem; color: #666;'>Version 2.1.0</p>
        <p style='font-size: 0.8rem; color: #666;'>© 2025</p>
    </div>
    """, 
    unsafe_allow_html=True
)