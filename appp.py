import streamlit as st
import pandas as pd
import joblib

# Load the trained Decision Tree model
model = joblib.load("credit.joblib")

# Streamlit app
def main():
    # Setting up some background colors
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to bottom, #7F7FD5, #86A8E7, #91EAE4);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Fraud Detection App")

    # Form for user input
    st.subheader("Input Form:")
    trans_date_trans_time = st.text_input("Transaction Date/Time:")
    cc_num = st.text_input("Credit Card Number:")
    merchant = st.text_input("Merchant:")
    category = st.text_input("Category:")
    amt = st.number_input("Amount:")
    first = st.text_input("First Name:")
    last = st.text_input("Last Name:")
    gender = st.selectbox("Gender", ["Male", "Female"])
    street = st.text_input("Street:")
    city = st.text_input("City:")
    state = st.text_input("State:")
    zip_code = st.text_input("ZIP Code:")
    city_pop = st.number_input("City Population:")
    job = st.text_input("Job:")
    dob = st.text_input("Date of Birth:")
    trans_num = st.text_input("Transaction Number:")

    # Make predictions
    if st.button("Predict Fraud Status"):
        input_data = pd.DataFrame({
            'trans_date_trans_time': [trans_date_trans_time],
            'cc_num': [cc_num],
            'merchant': [merchant],
            'category': [category],
            'amt': [amt],
            'first': [first],
            'last': [last],
            'gender': [gender],
            'street': [street],
            'city': [city],
            'state': [state],
            'zip': [zip_code],
            'city_pop': [city_pop],
            'job': [job],
            'dob': [dob],
            'trans_num': [trans_num]
        })

        # Factorize categorical columns
        for column in input_data.columns:
            input_data[column], _ = pd.factorize(input_data[column])

        # Make predictions
        prediction = model.predict(input_data)

        # Display results
        st.subheader("Prediction:")
        st.write(f"Predicted Fraud Status: {prediction[0]}")

if __name__ == "__main__":
    main()
