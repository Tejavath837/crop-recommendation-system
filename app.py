import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop recommendation system",layout="wide")

model=joblib.load('crop_model.pkl')
scaler=joblib.load('scaler.pkl')
le=joblib.load('label_encoder.pkl')

with st.sidebar:
    st.title('Crop recommendation system')
    st.write("Model: Random Forest")
    st.write("Accuracy: 99.32%")
    st.write("Dataset: 2200 samples, 22 crops")

st.title('Crop recommendation model')
st.write('Enter the SOIL  nand CLIMATE values ')
st.divider()
st.subheader('Enter the following required values')
col1,col2=st.columns(2)
with col1:
    N=st.slider('Nitrogen(N)',0,140,50)
    P=st.slider('Phosphorous(P)',0,145,50)
    K=st.slider('Potassium(K)',0,205,50)
    temperature=st.slider('Temperature(T)',0,44,25)
with col2:
    humidity=st.slider('Humidity(H)',14,100,60)
    ph=st.slider('Ph value(ph)',3.0,10.0,6.5)
    rainfall=st.slider('Rainfall(mm)',20,300,60)
st.divider()

if st.button('Predict'):
    warnings_list=[]
    if ph<4.0:
        warnings_list.append('ph is less than 4 most of the crops doesnot survive this much aicidity')
    elif ph>9.0:
        warnings_list.append('ph is more than 9 very few grow in this alkaline nature')
    if(temperature <10):
        warnings_list.append('most crops in dataset prefer warmer condition')
    elif(temperature >42):
        warnings_list.append('extreme heat many predict good result ')
    if humidity<15:
        warnings_list.append('very dry conditions')
    if rainfall<25:
        warnings_list.append('predctions may be incorrect due to extreme dry condition')
    for w in warnings_list:
        st.warning(w)
    
    
    input_data=pd.DataFrame([[N,P,K,temperature,humidity,ph,rainfall]],columns=['N','P','K','temperature','humidity','ph','rainfall'])
    prediction=model.predict(input_data)
    crop_name=le.inverse_transform(prediction)[0]
    probabilities=model.predict_proba(input_data)[0]
    confidence=probabilities.max()*100
    st.success(f" Recommended crop :- {crop_name}")
    st.info(f"Model confidence :- {confidence:.2f}")
    if confidence<50:
        st.warning('the inputs are unusual for this dataset treat with caution')
    st.subheader('top 3 crops')
    top3_indies=probabilities.argsort()[-3:][::-1]
    for i,idx in enumerate(top3_indies):
        crop=le.classes_[idx]
        prob=probabilities[idx]*100
        medal = ["🥇", "🥈", "🥉"][i]
        st.write(f"{medal} **{crop.capitalize()}** — {prob:.1f}% confidence")


