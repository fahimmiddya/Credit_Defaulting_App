import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

st.title("Credit Default Prediction App")

st.write("""
### _Makes predictions on which customers are likely to **default** in the following month & whether the account should be considered for **credit counseling** or not._
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
        EDUCATION  = st.sidebar.selectbox('Education',('School Graduate' , 'University' , 'High School' , 'Others'))
        MARRIAGE = st.sidebar.selectbox('Marital Status',('Married' , 'Single' , 'Others'))
        AGE = st.sidebar.slider('Age', 21,79,20)
        PAY_1 = st.sidebar.slider('Previous Month Payment', -2.00,8.00,0.00)
        LIMIT_BAL = st.sidebar.slider('Balance Limit', 10000.00,800000.00,0.00)
        BILL_AMT1 = st.sidebar.slider('Bill Amount 1', -165580.00,746814.00,0.00)
        BILL_AMT2 = st.sidebar.slider('Bill Amount 2', -165580.00,746814.00,0.00)
        BILL_AMT3 = st.sidebar.slider('Bill Amount 3', -165580.00,746814.00,0.00)
        BILL_AMT4 = st.sidebar.slider('Bill Amount 4', -165580.00,746814.00,0.00)
        BILL_AMT5 = st.sidebar.slider('Bill Amount 5', -165580.00,746814.00,0.00)
        BILL_AMT6 = st.sidebar.slider('Bill Amount 6', -165580.00,746814.00,0.00)
        PAY_AMT1 = st.sidebar.slider('Pay Amount 1', 0.00,873552.00,0.00)
        PAY_AMT2 = st.sidebar.slider('Pay Amount 2', 0.00,873552.00,0.00)
        PAY_AMT3 = st.sidebar.slider('Pay Amount 3', 0.00,873552.00,0.00)
        PAY_AMT4 = st.sidebar.slider('Pay Amount 4', 0.00,873552.00,0.00)
        PAY_AMT5 = st.sidebar.slider('Pay Amount 5', 0.00,873552.00,0.00)
        PAY_AMT6 = st.sidebar.slider('Pay Amount 6', 0.00,873552.00,0.00)
        if EDUCATION=="School Graduate":
           educ = 1
        elif EDUCATION == "University":
           educ =2
        elif EDUCATION == "High School":
           educ = 3
        else:
           educ = 4
        if MARRIAGE == "Married":
           marry =1
        elif MARRIAGE == "Single":
           marry =2
        else:
           marry = 3
        data = {'EDUCATION': educ,
                'MARRIAGE': marry,
                'AGE': AGE,
                'PAY_1': PAY_1,
                'LIMIT_BAL': LIMIT_BAL,
                'BILL_AMT1': BILL_AMT1,
                'BILL_AMT2': BILL_AMT2,
                'BILL_AMT3': BILL_AMT3,
                'BILL_AMT4': BILL_AMT4,
                'BILL_AMT5': BILL_AMT5,
                'BILL_AMT6': BILL_AMT6,
                'PAY_AMT1': PAY_AMT1,
                'PAY_AMT2': PAY_AMT2,
                'PAY_AMT3': PAY_AMT3,
                'PAY_AMT4': PAY_AMT4,
                'PAY_AMT5': PAY_AMT5,
                'PAY_AMT6': PAY_AMT6,
                }
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
credit_raw = pd.read_csv('cleaned_appdata.csv')
credit = credit_raw.drop(columns = ['DEFAULT'])
df = pd.concat([input_df,credit],axis=0)

# Encoding of ordinal features
#encode = ['EDUCATION','MARRIAGE']
#for col in encode:
#    dummy = pd.get_dummies(df[col], prefix=col)
#    df = pd.concat([df,dummy], axis=1)
#    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features:')
st.write(df)
st.write('---')


# Reads in saved classification model
load_rf = joblib.load(open('credit_rf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_rf.predict(df)
prediction_proba = load_rf.predict_proba(df)

# Prediction
st.subheader('Prediction :')
status = np.array(['Not Default',' Default'])
st.write(status[prediction])
st.write('---')
st.subheader('Prediction Probability:')
st.write(prediction_proba)
st.write('---')
st.subheader("Verdict :")
if all(prediction_proba[0]>=0.25):
    if all(prediction==0):
        st.write('The account should be considered for **CREDIT COUNSELING**.')
    else:
        st.write("The account shouldn't be considered for **CREDIT COUNSELING**.")    
