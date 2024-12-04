import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Column name mappings (same as before)
COLUMN_NAMES = {
    # Existing mapped columns
    'NETMONTHLYINCOME': 'Net Monthly Income',
    'Time_With_Curr_Empr': 'Employment Tenure',
    'CC_Flag': 'Credit Card Holder',
    'PL_Flag': 'Personal Loan Holder',
    'HL_Flag': 'Home Loan Holder',
    'GL_Flag': 'Gold Loan Holder',
    'EDUCATION': 'Education Level',
    'MARITALSTATUS_Married': 'Married Status',
    'MARITALSTATUS_Single': 'Single Status',
    'GENDER_F': 'Female',
    'GENDER_M': 'Male',

    # Additional columns to map
    'pct_tl_open_L6M': 'Percent Total Lines Open (Last 6M)',
    'pct_tl_closed_L6M': 'Percent Total Lines Closed (Last 6M)',
    'Tot_TL_closed_L12M': 'Total Lines Closed (Last 12M)',
    'pct_tl_closed_L12M': 'Percent Total Lines Closed (Last 12M)',
    'Tot_Missed_Pmnt': 'Total Missed Payments',
    'CC_TL': 'Credit Card Total Lines',
    'Home_TL': 'Home Loan Total Lines',
    'PL_TL': 'Personal Loan Total Lines',
    'Secured_TL': 'Secured Loan Total Lines',
    'Unsecured_TL': 'Unsecured Loan Total Lines',
    'Other_TL': 'Other Loan Total Lines',
    'Age_Oldest_TL': 'Age of Oldest Trade Line',
    'Age_Newest_TL': 'Age of Newest Trade Line',
    'time_since_recent_payment': 'Time Since Recent Payment',
    'max_recent_level_of_deliq': 'Max Recent Delinquency Level',
    'num_deliq_6_12mts': 'Delinquencies in 6-12 Months',
    'num_times_60p_dpd': 'Times 60+ Days Past Due',
    'num_std_12mts': 'Standard Accounts in 12 Months',
    
    # Additional columns
    'num_sub': 'Number of Substandard Accounts',
    'num_sub_6mts': 'Number of Substandard Accounts (6 Months)',
    'num_sub_12mts': 'Number of Substandard Accounts (12 Months)',
    'num_dbt': 'Number of Doubtful Accounts',
    'num_dbt_12mts': 'Number of Doubtful Accounts (12 Months)',
    'num_lss': 'Number of Loss Accounts',
    'recent_level_of_deliq': 'Recent Delinquency Level',
    'CC_enq_L12m': 'Credit Card Inquiries (Last 12M)',
    'PL_enq_L12m': 'Personal Loan Inquiries (Last 12M)',
    'time_since_recent_enq': 'Time Since Recent Inquiry',
    'enq_L3m': 'Inquiries in Last 3 Months',
    'pct_PL_enq_L6m_of_ever': 'Percent Personal Loan Inquiries (Last 6M)',
    'pct_CC_enq_L6m_of_ever': 'Percent Credit Card Inquiries (Last 6M)',
    
    # Product Inquiry Columns
    'last_prod_enq2_AL': 'Last Product Inquiry - Auto Loan',
    'last_prod_enq2_CC': 'Last Product Inquiry - Credit Card',
    'last_prod_enq2_ConsumerLoan': 'Last Product Inquiry - Consumer Loan',
    'last_prod_enq2_HL': 'Last Product Inquiry - Home Loan',
    'last_prod_enq2_PL': 'Last Product Inquiry - Personal Loan',
    'last_prod_enq2_others': 'Last Product Inquiry - Others',
    
    # First Product Inquiry Columns
    'first_prod_enq2_AL': 'First Product Inquiry - Auto Loan',
    'first_prod_enq2_CC': 'First Product Inquiry - Credit Card',
    'first_prod_enq2_ConsumerLoan': 'First Product Inquiry - Consumer Loan',
    'first_prod_enq2_HL': 'First Product Inquiry - Home Loan',
    'first_prod_enq2_PL': 'First Product Inquiry - Personal Loan',
    'first_prod_enq2_others': 'First Product Inquiry - Others'
}

def load_model():
    with open('xgb_classifier.pkl', 'rb') as f:
        return pickle.load(f)

def predict_credit_approval(model, input_data):
    probabilities = model.predict_proba(input_data)
    return probabilities

def main():
    st.set_page_config(page_title="Credit Approval Predictor", page_icon="💳")
    st.title("Credit Approval Prediction")

    # Load the model
    model = load_model()

    # Sidebar for input
    st.sidebar.header("Input Customer Details")
    
    # Create input fields dynamically
    input_data = {}
    renamed_input_data = {}
    for column in model.feature_names_in_:
        renamed_column = COLUMN_NAMES.get(column, column)
        
        if column in ['GENDER_F', 'GENDER_M', 'MARITALSTATUS_Married', 'MARITALSTATUS_Single', 
                      'CC_Flag', 'PL_Flag', 'HL_Flag', 'GL_Flag']:
            # Boolean/Binary columns
            input_data[column] = st.sidebar.selectbox(renamed_column, [0, 1], key=column)
        elif column == 'EDUCATION':
            # Categorical column
            input_data[column] = st.sidebar.selectbox(renamed_column, [0, 1, 2, 3], key=column)
        else:
            # Numerical columns
            input_data[column] = st.sidebar.number_input(renamed_column, key=column)
        
        # Store the renamed input for display
        renamed_input_data[renamed_column] = input_data[column]

    # Predict button
    if st.sidebar.button("Predict Credit Approval"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Get probabilities
        probabilities = predict_credit_approval(model, input_df)[0]
        
        # Display results
        st.header("Prediction Results")
        
        # Create a bar chart of probabilities
        st.bar_chart({
            'Low Risk (P1)': probabilities[0],
            'Med Risk(P2)': probabilities[1],
            'High Risk (P3)': probabilities[2],
            'Near to Default (P4)': probabilities[3]
        })
        
        # Detailed probability display
        st.subheader("Detailed Probability Breakdown")
        cols = st.columns(4)
        categories = ['Not Approved (P1)', 'Low Approval (P2)', 'Medium Approval (P3)', 'High Approval (P4)']
        
        for i, (cat, prob) in enumerate(zip(categories, probabilities)):
            with cols[i]:
                st.metric(cat, f"{prob*100:.2f}%")
        
        # Display input data with renamed columns
        st.subheader("Input Data")
        st.write(pd.DataFrame([renamed_input_data]).T)

if __name__ == "__main__":
    main()