from tensorflow.keras.models import load_model
import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st

with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = pickle.load(file_1)

with open("list_num_cols.txt", "r") as file_2:
    list_num_cols = json.load(file_2)

with open("list_catn_cols.txt", "r") as file_3:
    list_catn_cols = json.load(file_3)

model_ann = load_model('churn_predict_model.h5')

def run():
    with st.form(key="Customer churn risk prediction"):
        name = st.text_input("Name", value="")
        age = st.number_input("Age", min_value=0, max_value=150, value=30, help="age")
        Gender = st.selectbox("Gender", ("Male", "Female"))
        selected_date = st.date_input("Joining Date")
        st.markdown("---")
        membership_category = st.selectbox("Membership", ('No Membership', 'Basic Membership', 'Silver Membership', 'Premium Membership', 'Gold Membership', 'Platinum Membership'))
        region_category = st.selectbox("Region", ("City", "Village", 'Town'))
        internet_option = st.selectbox("Internet", ("Wi-Fi", "Fiber_Optic", 'Mobile_Data'))
        preferred_offer_types = st.selectbox("Preferred Offer Types", ('Without Offers', 'Credit/Debit Card Offers','Gift Vouchers/Coupons'))
        medium_of_operation = st.selectbox("Medium Of Operation",('Desktop', 'Smartphone', 'Both'))
        complaint_status = st.selectbox("Complaint Status", ('No Information Available', 'Not Applicable', 'Unsolved', 'Solved','Solved in Follow-up'))
        feedback = st.selectbox("Feedback", ('Poor Website', 'Poor Customer Service', 'Too many ads','Poor Product Quality', 'No reason specified','Products always in Stock', 'Reasonable Price','Quality Customer Care', 'User Friendly Website'))
        st.markdown("---")
        joined_through_referral = st.selectbox("Joined Through Referral", ("Yes", "No"))
        used_special_discount = st.selectbox("Used Special Discount", ("Yes", "No"))
        offer_application_preference = st.selectbox("Offer Application Preference", ("Yes", "No"))
        past_complaint = st.selectbox("Past Complaint", ("Yes", "No"))
        st.markdown("---")
        days_since_last_login = st.number_input("Days Since Last Login", min_value=0, max_value=60, value=10, help="Days Since Last Login")
        avg_time_spent = st.number_input("Average Time Spent", min_value=0, max_value=10000, value=100, help="Average Time Spent")
        avg_transaction_value = st.number_input("Average Transaction Value", min_value=0, max_value=1000000, value=2500, help="Average Transaction Value")
        avg_frequency_login_days = st.number_input("Average Frequency Login Days", min_value=0, max_value=200, value=15, help="Average Frequency Login Days")
        points_in_wallet = st.number_input("Points In Wallet", min_value=0, max_value=100000, value=500, help="Points In Wallet")

        submitted = st.form_submit_button("Predict")

    data_inf = {
        'membership_category': membership_category,
        'feedback': feedback,
        'days_since_last_login': days_since_last_login,
        'avg_time_spent': avg_time_spent,
        'avg_transaction_value': avg_transaction_value,
        'avg_frequency_login_days':avg_frequency_login_days,
        'points_in_wallet': points_in_wallet
        }

    data_inf = pd.DataFrame([data_inf])

    if submitted:
        num_cols = list_num_cols
        catn_cols = list_catn_cols

        data_inf_transform = model_pipeline.transform(data_inf)

        y_pred_inf = model_ann.predict(data_inf_transform)
        y_pred_inf = np.where(y_pred_inf>=0.5,"Risk Of Churn","Not Risk Of Churn")
        st.write(y_pred_inf)

if __name__ == "__app__":
    run()