import streamlit as st
import joblib
import pandas as pd


st.set_page_config(

page_title="Loan Risk Predictor",
page_icon="🏦",
layout="wide"

)

try:
    model = joblib.load(
        "models/loan_default_model (1).pkl"
    )

except Exception as e:
    st.error(e)


st.markdown(

"""
<style>

.stButton button{

width:100%;
height:45px;
font-size:18px;

}

</style>

""",

unsafe_allow_html=True

)



st.title(
"🏦 Bank Loan Default Risk Predictor"
)


st.write(
"AI powered loan risk assessment system"
)



col1,col2,col3 = st.columns(3)



with col1:


    age = st.number_input(
        "Age",
        21,
        70
    )


    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )


    education = st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )


    employment = st.selectbox(
        "Employment Type",
        [
        "Salaried",
        "Self Employed",
        "Business"
        ]
    )



with col2:


    income = st.number_input(
        "Annual Income",
        100000,
        10000000
    )


    credit = st.slider(
        "Credit Score",
        300,
        900,
        650
    )


    loan_type = st.selectbox(
        "Loan Type",
        [
        "Home Loan",
        "Personal Loan",
        "Auto Loan",
        "Education Loan"
        ]
    )


    loan_amount = st.number_input(
        "Loan Amount",
        100000,
        50000000
    )



with col3:


    interest = st.number_input(
        "Interest Rate",
        5.0,
        20.0
    )


    debt = st.slider(
        "Debt To Income Ratio",
        0.0,
        1.0,
        0.3
    )


    collateral = st.number_input(
        "Collateral Value",
        0,
        100000000
    )


    existing = st.number_input(
        "Existing Loans",
        0,
        10
    )



if st.button(
"🔍 Predict Risk"
):


    input_data = pd.DataFrame({

        "Age":[age],

        "Gender":[gender],

        "Marital_Status":["Married"],

        "Dependents":[1],

        "Education":[education],

        "Employment_Type":[employment],

        "Annual_Income":[income],

        "Monthly_Income":[income/12],

        "Credit_Score":[credit],

        "Existing_Loan_Count":[existing],

        "Loan_Type":[loan_type],

        "Loan_Amount":[loan_amount],

        "Loan_Term":[60],

        "Interest_Rate":[interest],

        "Debt_to_Income_Ratio":[debt],

        "Collateral_Value":[collateral],

        "Loan_Region":["East"],

        "Disbursement_Status":["Disbursed"],

        "Repayment_Status":["On Time"]

    })



    result = model.predict(
        input_data
    )


    if result[0]==1:

        st.error(
        "⚠️ High Default Risk"
        )


    else:

        st.success(
        "✅ Low Default Risk"
        )
