import streamlit as st
import numpy as np
import pickle
import pandas as pd

if __name__=='__main__':
    model = pickle.load(open('cal_housing_model.pkl','rb'))
    scaler = pickle.load(open('cal_housing_preprocessing_scaler.pkl','rb'))

    st.set_page_config(page_title="House Price")
    st.header("California House Price Prediction")
    st.markdown("""
                The target variable is the median house value for California districts.
                
                All attributes are collected from the block.
                
                Attributes: 
                - MedInc        median income in block group (measured in $10,000) 
                - HouseAge      median house age in block group 
                - AveRooms      average number of rooms per household 
                - AveBedrms     average number of bedrooms per household 
                - Population    block group population 
                - AveOccup      average number of household members 
                - Latitude      block group latitude 
                - Longitude     block group longitude 
                """)
    with st.form("Inputs"):
        medinc = st.number_input("Median Income in Block Group (in $10,000)",value=4.0,step=0.01)
        
        houseage = st.number_input("Median House Age in Block Group",value=25,step=1)
        
        averooms = st.number_input("Average Number of Rooms per Household",value=6,step=1)
        
        avebedrms = st.number_input("Average Number of Bedrooms per Household",value=2,step=1)
        
        population = st.number_input("Block Group Population",value=1500,step=10)
        
        aveoccup = st.number_input("Average Number of Household Members",value=4,step=1)
        
        latitude = st.slider("Latitude (limited to California)", min_value=32.54,max_value=41.95,step=0.01,value=37.76)
        longitude = st.slider("Longitude (limited to California)", min_value=-124.35,max_value=-114.31,step=0.01,value=-121.51)
        
        submit_form = st.form_submit_button("Submit")
    
    if submit_form:
        input_vector = [medinc,houseage,averooms,avebedrms,population,aveoccup,latitude,longitude]
        scaled_vector=scaler.transform(np.array(input_vector).reshape(1,-1))
        
        predicted_price=int(model.predict(scaled_vector)[0]*100000)
        st.write("The predicted median house price is **$ {:,}**".format(predicted_price))
        
        location = pd.DataFrame([[latitude,longitude]],columns=['lat','lon'])
        st.map(location)
        
        st.markdown("-----")
        st.write("This model was created by training a CatBoostClassifier on the California Housing Price dataset.")
        st.write("Check out the source code [here](https://github.com/spikspiks/cali_housepricing)")