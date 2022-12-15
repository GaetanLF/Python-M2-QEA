#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 20:21:58 2022

@author: Julia SCHMIDT, Gaëtan LE FLOCH

/!\ WARNING: Two absolute pathfiles need to be modified.
"""

# import the streamlit library
import streamlit as st
import pandas as pd
import numpy as np
import tweepy
import pickle
from datetime import datetime


def encodingLast(df, col_name,add):
    '''Function takes the dataframe, the column name that is encoded and outputs the dataframe with the encoded catgecorical variables'''
    serie = pd.concat((df[col_name],pd.Series(add)))
    class_name = serie.unique()
    serie = pd.Categorical(serie, categories = class_name).codes
    return serie.tolist()[-1]

#%% Settle the Twitter API

key = "ZTy6KNHKikYvhrkrNjwGlE4Oh"
key_secret = "gjGUR8ilD1mHSjYwaEFhdl7Iqc9EBI3mvyUA9wawCFlwCsxXU0"
token = "1594776332226990081-OGcGAqhvS811ny4AdbgwdHfrJypjsv"
token_secret = "tE7IeKQyODarRrqPsae1LhLfhLKc3pmIzB3mEO3r0yCGm"

auth = tweepy.OAuthHandler(key, key_secret)
auth.set_access_token(token, token_secret)
api = tweepy.API(auth)

#%% Import the data

df = pd.read_csv('/Users/gaetanlefloch/Documents/Master2/Python/Final_Project/data/metabolic_syndrome.csv')
# Absolute pathfile to be modified.
# give a title to our app
st.title('Welcome to this auto-test for Metabolic syndrome')
st.text("By Julia SCHMIDT and Gaëtan LE FLOCH")

st.text("This algorithm is made by the Hospital to offer a first diagnosis on your condition.")
st.text("Please contact your M.D. if it says so.")
st.text("You can decide to share the pre-diagnosis on Twitter to raise awareness on Metabolic Syndrome.")
 
st.markdown("### General informations")

sex = st.radio(('Indicate your sex :'), ('Male','Female'))


age = st.number_input("Fill your age",min_value=18,max_value=100,step=1)

Race = st.radio('Select your ethnical group: ',
                  ('White', 'Black', 'Hispanic','MexAmerican',
                   'Asian','Other'))

salary = st.number_input("Indicate your monthly income: ",min_value=0,max_value=20000,step=100)

Marital = st.radio('Select your marital status: ',
                         ('Single','Widowed','Divorced','Married',
                          'Separated'))


st.markdown("### Medical informations")
st.text("Please refer to the numerous analyses you had to fill the following questions.")

waistCircle = st.number_input('Fill your waist circle: ',min_value=64,max_value=300)

albuminuria = st.radio('Indicate your albuminuria stage :', ('No albuminuria',
                                                             'Microalbuminuria',
                                                             'Macroalbuminuria'))

# TAKE WEIGHT INPUT in kgs
weight = st.number_input("Enter your weight (in kgs)",min_value=30,step=1)
 
# TAKE HEIGHT INPUT
height = st.number_input('Select your height format in cms: ',min_value=100)

bmi = weight/((0.01*height)**2)
if(st.button('Calculate BMI')):
    # print the BMI INDEX
    st.text(f"Your BMI Index is {bmi}.")
 
   # give the interpretation of BMI index
    if(bmi < 16):
        st.error("You are Extremely Underweight")
    elif (bmi >= 16 and bmi < 18.5):
        st.warning("You are Underweight")
    elif(bmi >= 18.5 and bmi < 25):
        st.success("Healthy")
    elif(bmi >= 25 and bmi < 30):
        st.warning("Overweight")
    elif(bmi >= 30):
        st.error("Extremely Overweight")

uricAcid = st.number_input('Enter your uric acid level (in mG.L): ')

ACRatio = st.number_input('Enter your albumin on creatinine ratio :')

bloodGlucose = st.number_input('Enter your blood glucose: ')

triglycerides = st.number_input('Enter your triglycerides level :')

HDL = st.number_input('Enter your HDL level :')

shareTwitter = st.checkbox("YES, share my pre-diagnosis on Twitter if I have the syndrome",
                           value=True)

if (st.button('Do I have metabolic syndrome ?')):
    
    loaded_model = pickle.load(open('/Users/gaetanlefloch/Documents/Master2/Python/Final_Project/data/dtc.sav', 'rb'))
    # Absolute pathfile to be modified
    sex = encodingLast(df,"Sex",sex)
    Marital = encodingLast(df, 'Marital',Marital)
    Race = encodingLast(df, 'Race',Race)
    albuminuria = encodingLast(df,'Albuminuria',albuminuria)
    y = np.array([age,sex,Marital,salary,Race,waistCircle,bmi,
                           albuminuria,ACRatio,uricAcid,bloodGlucose,
                           HDL,triglycerides]).reshape(-1,1)
    y = y.transpose()
    prediction = int(loaded_model.predict(y))
    if not prediction: # 0 case
        st.success(f"{prediction} Based on the declarations, you might not have metabolic syndrome.")
    else:
        st.error("Based on the declarations, you may have metabolic syndrome. Please contact your M.D.")  
        if shareTwitter:
            now = datetime.now()
            api.update_status(status = f"{now.strftime('%H:%M:%S')} : a patient aged {age} has been pre-diagnosed with MetSyn.")
