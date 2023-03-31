import numpy as np 
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

diabetes_model = pickle.load(open("trained models/diabetes_model.sav",'rb'))
heart_model = pickle.load(open("trained models/heart_model.sav",'rb'))

selected =''
with st.sidebar:
    
    selected = option_menu('Multiple Disease prediction system',['Diabetes prediction','Heart Disease prediction'],icons=['activity','heart'],default_index =0)


def diabetes_pred(input_data):
    
    inp_np_arr = np.asarray(input_data)
    inp_data_reshaped = inp_np_arr.reshape(1,-1)
    pred = diabetes_model.predict(inp_data_reshaped)
    if pred[0] == 0:
        return 'You are not diabetic'
    else:
        return 'You are diabetic'

    
def heart_disease_pred(input_data):
    
    inp_np_arr = np.asarray(input_data)
    inp_data_reshaped = inp_np_arr.reshape(1,-1)
    pred = heart_model.predict(inp_data_reshaped)
    if pred[0] == 0:
        return 'You have heart disease'
    else:
        return 'You do not have heart disease'
    
if (selected=='Diabetes prediction'):
    st.title('Diabetes Prediction System')
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.text_input('Number of pregnancies')
        glucose = st.text_input('Blood Glucose level')
        bloodPressure = st.text_input('Blood Pressure value')
    with col2:
        skinThickness = st.text_input('Skin Thickness value')
        insulin = st.text_input('Insulin level')
        bmi = st.text_input('BMI value')
    with col3:
        diabetespedigreeFunction = st.text_input('Diabetes pedigree function value')
        age = st.text_input('Age')
    
    diagnosis= ''
    
    if st.button('Diabetes test result'):
        diagnosis = diabetes_pred([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,diabetespedigreeFunction,age])
        
    st.success(diagnosis)
    

if (selected=='Heart Disease prediction'):
    st.title('Heart Disease prediction')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
        sex = st.text_input('Sex')
        cp = st.text_input('Chest pain type')
        trestbps = st.text_input('Resting blood pressure value')
        chol = st.text_input('Serum Cholestrol value in mg/dl')
    with col2:
        fbs = st.text_input('Fasting blood sugar value')
        resrecg = st.text_input('Rest ecg result (0,1,2)')
        thalach = st.text_input('Maximum heart rate achived')
        exang = st.text_input('Excercise induced angima 1 for yes 0 for no')
        oldpeak =  st.text_input('ST depression induced by exercise relative to rest')
    with col3:
        slope = st.text_input('the slope of the peak exercise ST segment')
        ca = st.text_input('No of major vessels (0-3)')
        thal = st.text_input('0-normal 1- fixed defect 2- reverseable defect')
    diagnosis= ''
    
    if st.button('Heart disease test result'):
        diagnosis = heart_disease_pred([age,sex,cp,trestbps,chol,fbs,resrecg,thalach,exang,oldpeak,slope,ca,thal])
        
    st.success(diagnosis)

    
    

