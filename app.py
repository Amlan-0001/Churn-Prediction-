import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import time  # added for delay effects

# ------------------ Page Configuration ------------------ #
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------ Load Model & Tools ------------------ #
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------ Header Section ------------------ #
st.image("https://user-images.githubusercontent.com/58620359/174948746-5dc3418a-8296-4cc8-9561-f8f12ca9a0a4.png", use_container_width=True)
st.title(" Customer Churn Prediction App")

st.markdown("""
Welcome to the **Churn Prediction App**!  
This tool helps banks predict whether a customer is likely to leave (churn) based on their profile and account activity.

**👈 Enter customer details in the sidebar to make a prediction.**
""")

# ------------------ Sidebar Inputs ------------------ #
st.sidebar.header("🧾 Customer Details")

geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age', 18, 92)
balance = st.sidebar.number_input('💰 Balance')
credit_score = st.sidebar.number_input('📊 Credit Score')
estimated_salary = st.sidebar.number_input('💼 Estimated Salary')
tenure = st.sidebar.slider('📅 Tenure (years)', 0, 10)
num_of_products = st.sidebar.slider('📦 Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card [Yes=1, No=0]', [0, 1])
is_active_member = st.sidebar.selectbox('✅ Is Active Member[Yes=1 , No =0]', [0, 1])

# ------------------ Main Predict Button ------------------ #
st.markdown("## 🔍 Click below to predict churn")
if st.button("🚀 Predict Now"):
    
    # Suspense animation
    st.markdown("🎲 Predicting...")
    time.sleep(1)
    st.markdown("🧠 Analyzing Data...")
    time.sleep(1)
    st.markdown("✅ Result Ready!")
    time.sleep(0.5)

    # Prepare input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Merge with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0][0]

    st.markdown("---")
    st.subheader("📌 Prediction Result")
    st.write(f"**Churn Probability:** `{prediction:.2f}`")

    if prediction > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **likely to stay**.")

    st.markdown("""
    ---
    **Note:** This prediction is based on a trained machine learning model. It provides a probability based on patterns in historical data.
    """)

# ------------------ Footer ------------------ #
st.markdown("     Created by **Amlan** | Powered by Streamlit & TensorFlow")
