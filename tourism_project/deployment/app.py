import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


TYPE_OF_CONTACT_MAP = {'Self Enquiry': 1, 'Company Invited': 0}
OCCUPATION_MAP = {'Salaried': 2, 'Free Lancer': 0, 'Small Business': 3, 'Large Business': 1}
GENDER_MAP = {'Female': 1, 'Male': 2, 'Fe Male': 0}
MARITAL_STATUS_MAP = {'Single': 2, 'Divorced': 0, 'Married': 1, 'Unmarried': 3}
PRODUCT_PITCHED_MAP = {'Deluxe': 1, 'Basic': 0, 'Standard': 3, 'Super Deluxe': 4, 'King': 2}
DESIGNATION_MAP = {
    'Manager': 2, 'Executive': 1, 'Senior Manager': 3, 'AVP': 0, 'VP': 4
}


# Download and load the model
try:
    model_path = hf_hub_download(repo_id="Pvt-Pixel/tourism_package_model", filename="best_tourism_package_model_v1.joblib")
    model = joblib.load(model_path)
    st.success("Model loaded successfully from Hugging Face Hub!")
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure the model exists in the specified Hugging Face repository and has the correct filename.")
    st.stop() # Stop the app if model can't be loaded


# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Please enter customer details and interaction data below to get a prediction.
""")

# User input for Customer Details
st.header("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Age of the customer.")
typeofcontact = st.selectbox("Type of Contact", list(TYPE_OF_CONTACT_MAP.keys()), help="Method by which the customer was contacted.")
citytier = st.selectbox("City Tier", [1, 2, 3], help="City category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).")
occupation = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()), help="Customer's occupation.")
gender = st.selectbox("Gender", list(GENDER_MAP.keys()), help="Gender of the customer.")
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, help="Total number of people accompanying the customer on the trip.")
preferredpropertystar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3, help="Preferred hotel rating by the customer (1-5 stars).")
maritalstatus = st.selectbox("Marital Status", list(MARITAL_STATUS_MAP.keys()), help="Marital status of the customer.")
numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5, help="Average number of trips the customer takes annually.")
passport = st.selectbox("Passport Holder?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Whether the customer holds a valid passport (0: No, 1: Yes).")
owncar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Whether the customer owns a car (0: No, 1: Yes).")
numberofchildrenvisiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0, help="Number of children below age 5 accompanying the customer.")
designation_input = st.selectbox("Designation", list(DESIGNATION_MAP.keys()), help="Customer's designation in their current organization. Please select from the list or ensure your custom input maps correctly.")
monthlyincome = st.number_input("Monthly Income", min_value=0, max_value=200000, value=50000, help="Gross monthly income of the customer.")

# User input for Customer Interaction Data
st.header("Customer Interaction Data")
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3, help="Score indicating the customer's satisfaction with the sales pitch.")
productpitched = st.selectbox("Product Pitched", list(PRODUCT_PITCHED_MAP.keys()), help="The type of product pitched to the customer.")
numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2, help="Total number of follow-ups by the salesperson after the sales pitch.")
durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15, help="Duration of the sales pitch delivered to the customer.")

# Assemble input into DataFrame, ensuring column order matches training data
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': TYPE_OF_CONTACT_MAP[typeofcontact],
    'CityTier': citytier,
    'Occupation': OCCUPATION_MAP[occupation],
    'Gender': GENDER_MAP[gender],
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': MARITAL_STATUS_MAP[maritalstatus],
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': DESIGNATION_MAP[designation_input],
    'MonthlyIncome': monthlyincome,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'ProductPitched': PRODUCT_PITCHED_MAP[productpitched],
    'NumberOfFollowups': numberoffollowups,
    'DurationOfPitch': durationofpitch
}])

# Add a prediction button
if st.button("Predict Purchase"):
    # Make prediction
    # The model expects probability, and we used a threshold of 0.45 during training
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = (prediction_proba >= 0.45).astype(int)[0]

    result = "Will Purchase Wellness Tourism Package" if prediction == 1 else "Will NOT Purchase Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.markdown(f"**{result}**")
    st.info(f"Probability of Purchase: {prediction_proba[0]:.2f}")
