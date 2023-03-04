#Import Libs
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

#--------------
# GUI
st.markdown("<h1 style='text-align: center; color: grey;'>Data Science Project 3</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: blue;'>Segmentation Customers</h1>", unsafe_allow_html=True)
 
Recency = st.text_input(label="Input Recency of Customer:")
Frequency = st.number_input(label="Input Frequency of Customer:")
Monetary = st.number_input(label="Input Monetary of Customer:")
if (Recency!="")&(Frequency!=0)&(Monetary!=0):
    data_input = pd.DataFrame({'Recency': [int(Recency)],
                                       'Frequency': [Frequency],
                                       'Monetary': [Monetary]})
    st.dataframe(data_input)

    

