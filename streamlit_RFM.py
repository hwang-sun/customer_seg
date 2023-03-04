#Import Libs
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from sklearn.decomposition import PCA
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load file csv
@st.cache_data(max_entries=1000)
def load_csv(file):
    csv=pd.read_csv(file)
    return csv

# Load file txt
@st.cache_data(max_entries=1000)
def load_txt(file):
    txt=pd.read_csv(file, delimiter ='\s+',header=None,names=["customer_ID","date", "quantity", "price"])
    return txt

# Convert df
@st.cache_data(max_entries=1000)
def convert_df(df):
    return df.to_csv("data.csv",index = False).encode('utf-8')

# Read data
@st.cache_data(max_entries=1000)
def read_data(df):
    # Convert date to datetime format
    df[['date']] = df[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[6:],s[4:6], s[0:4]))
    string_to_date = lambda x : datetime.strptime(x, "%d/%m/%Y").date()
    df['date']= df['date'].apply(string_to_date)
    df['date'] = df['date'].astype('datetime64[ns]')
    #drop null
    df.dropna(inplace = True)
    #drop duplicate
    df.drop_duplicates(inplace = True)
    #remove the orders have price = 0
    df = df[df["price"] > 0]
    
    return df

# Create RFM
@st.cache_data(max_entries=1000)
def create_RFM(df):
    # Convert string to date, get max date of dataframe
    max_date = df['date'].max().date()
    #RFM
    Recency = lambda x : (max_date - x.max().date()).days
    Frequency  = lambda x: x.count()
    Monetary = lambda x : round(sum(x), 2)
    df_RFM = df.groupby('customer_ID').agg({'date': Recency,
                                        'quantity': Frequency,  
                                        'price': Monetary })
    # Rename the columns of DataFrame
    df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
    # Descending Sorting 
    df_RFM = df_RFM.sort_values('Monetary', ascending=False)
    #Outliers
    new_df = df_RFM[['Recency','Frequency','Monetary']].copy()
    z_scores = stats.zscore(new_df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_df = new_df[filtered_entries]
    
    return new_df

# Data pre-processing

#Scaling data
@st.cache_data(max_entries=1000)
def preprocess(df):
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    norm = scaler.transform(df_log)
    return norm

#PCA
@st.cache_resource
def PCA_model(df):
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(df)
    PCA_components = pd.DataFrame(principalComponents)
    return PCA_components

# Load pickle
@st.cache_resource
def load_pickle(pkl_filename):
    with open(pkl_filename, 'rb') as file:  
        model = pickle.load(file)
    return model

def save_pickle(pkl_filename,model):
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(model,file)

#Calculate average RFM values and size for each cluster:
@st.cache_data(max_entries=1000)
def Label_model(new_df,_model):
    rfm_rfm = new_df.assign(K_Cluster = _model.labels_)
    rfm_rfm["Segment"]=rfm_rfm['K_Cluster'].map(lambda x:"Loyal" if x ==1
                                              else "Promising" if x == 2 
                                              else "Hibernating")
    return rfm_rfm

@st.cache_data(max_entries=1000)
def average_RFM(rfm_rfm):
    rfm_agg2 = rfm_rfm.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']}).round(1)
    rfm_agg2.columns = rfm_agg2.columns.droplevel()
    rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
    rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)
    # Reset the index
    rfm_agg2 = rfm_agg2.reset_index()
    return rfm_agg2

#File upload
@st.cache_data(max_entries=1000)
def upload_file(uploaded_file_1,uploaded_file_2):
    new_df1=''
    if uploaded_file_1 is not None:
        new_df1=load_txt(uploaded_file_1)
        st.write("Thanks for your RAW file")
        new_df1 = read_data(new_df1)
        new_df1= create_RFM(new_df1)
        return new_df1
    if uploaded_file_2 is not None:
        new_df1=load_csv(uploaded_file_2)
        st.write("Thanks for your RFM file")
        return new_df1
    return new_df1

#load file result
Customer = load_csv("result_KMeans.csv")

# Load & Train data
df=load_txt("CDNOW_master.txt")
df=read_data(df)
new_df=create_RFM(df)
df_scale=preprocess(new_df)
PCA_components=PCA_model(df_scale)
#model = KMeans(n_clusters= 3, random_state=42,n_init=10)
model=load_pickle('model_Kmeans.pkl')
model.fit(PCA_components.iloc[:,:1])
silhouette=silhouette_score(PCA_components.iloc[:,:1], model.labels_, metric='euclidean')
rfm_rfm=Label_model(new_df,model)
rfm_agg2=average_RFM(rfm_rfm)

#--------------
# GUI
st.markdown("<h1 style='text-align: center; color: grey;'>Data Science Project 3</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: blue;'>Segmentation Customers</h1>", unsafe_allow_html=True)
menu = ["Business Objective", "Build Project","New Prediction"]

choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':   
    st.subheader("Business Objective")
    st.image("new1.png")
    st.write("""
    An e-commerce company wants to segment its customers and determine marketing strategies according to these segments. For example, it is desirable to organize different campaigns to retain customers who are very profitable for the company, and different campaigns for new customers.
    """)
    st.write("""
    Customer segmentation is the process of dividing customers into groups based on common characteristics so companies can market to each group effectively and appropriately. In this project, we have applied RFM & Machine Learning algorithms to cluster customers from file CDNOW_master.txt.
    """)
    st.image("RFM.png")
    st.write("""
    RFM analysis is a technique used to categorize customers according to their purchasing behaviour. RFM stands for the three dimensions:
    1. Recency: This is the date when the customer made the last purchase. It is calculated by subtracting the customer's last shopping date from the analysis date.
    2. Frequency: This is the total number of purchases of the customer. In a different way, it gives the frequency of purchases made by the customer.
    3. Monetary: It is the total monetary value spent by the customer.
    """)
    st.write("""##### Our Task:""")
    st.write("""=> Problem/ Requirement: Use RFM & Machine Learning algorithms in Python for Customers segmentation.""")
    st.image("2.png")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.image("new.jpg")
    st.write("##### Dataset")
    st.write("""
    The file CDNOW_master.txt contains the entire purchase history up to the end of June 1998 of the cohort of 23,570 individuals who made their first-ever purchase at CDNOW in the first quarter of 1997. This CDNOW dataset was first used by Fader and Hardie (2001).
    Each record in this file, 69,659 in total, comprises four fields: the customer's ID, the date of the transaction, the number of CDs purchased, and the dollar value of the transaction.
    """)
    st.write("##### 1. Some data")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))

    st.write("##### 2. Visualize data")
    st.write("###### Sales in different Months")
    st.image("month_year.png")
    st.write("###### Pareto chart")
    st.image("pareto.png")
    st.write("###### Cohor chart")
    st.image("cohor.png")

    st.write("##### 3. Build model...")

    st.write("##### 4. Evaluation")
    st.dataframe(rfm_agg2)
    st.code("Silhouette score: "+str(round(silhouette,2)))
    st.write("###### Tree plot")
    st.image("chart.png")
    colors_dict = {'Loyal':'gold','Hibernating':'red','Promising':'green'}
    st.write("###### 2D Scatter")
    fig2 = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Segment",
           hover_name="Segment", size_max=100)
    st.plotly_chart(fig2, use_container_width=True)
    st.write("###### 3D Scatter")
    fig = px.scatter_3d(rfm_rfm, x="Recency", y="Monetary", z="Frequency",
                    color = 'Segment', opacity=0.5,
                    color_discrete_map = colors_dict)     
    st.plotly_chart(fig, use_container_width=True)

    st.write("##### 5. Summary: This model is good enough for Customers Segmentation")


elif choice == 'New Prediction':
    st.image("1.jpg")
    st.subheader("Select data")
    lines = None
    type = st.radio("Upload data (txt/csv) or Input new customer or Input customer ID?", options=("Upload", "Input_new_customer","Customer_ID"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a RAW txt/csv file", type=['txt','csv'])
        uploaded_file_2 = st.file_uploader("OR choose a RFM csv file", type=['csv'])
        if uploaded_file_1 is not None:
            new_df1=upload_file(uploaded_file_1,uploaded_file_2)
            lines = preprocess(new_df1)
            PCA_components = PCA_model(lines)
            lines=PCA_components.iloc[:,:1]  
            st.write("Content:")
            if len(lines)>0:      
                y_pred_new = model.predict(lines)    
                st.code("New predictions(Loyal: 1; Promising: 2; Hibernating: 0): " + str(y_pred_new))
                #Calculate average RFM values and size for each cluster:
                rfm_rfm_k3 = Label_model(new_df1,model)
                st.dataframe(rfm_rfm_k3)
                #csv
                csv=rfm_rfm_k3.to_csv("data.csv")
                with open('data.csv', 'r') as f:
	                st.download_button('Download result file', f,file_name='data.csv', mime='text/csv')
        
    if type=="Input_new_customer":
        Recency = st.text_input(label="Input Recency of Customer:")
        Frequency = st.number_input(label="Input Frequency of Customer:")
        Monetary = st.number_input(label="Input Monetary of Customer:")
        if (Recency!="")&(Frequency!=0)&(Monetary!=0):
            data_input = pd.DataFrame({'Recency': [int(Recency)],
                                       'Frequency': [Frequency],
                                       'Monetary': [Monetary]})
            new_df1=new_df.copy()
            new_df1=new_df1.append(data_input)
            lines = preprocess(new_df1)
            PCA_components = PCA_model(lines)
            lines=PCA_components.iloc[:,:1] 
            st.write("Content:")
            if len(lines)>0: 
                model.fit(lines)
                y_pred_new = Label_model(new_df1,model)
                y_pred_new=pd.DataFrame({'New_customer':y_pred_new.iloc[-1]})
                st.write("New predictions: ")
                st.dataframe(y_pred_new,use_container_width=True)
            
    if type=="Customer_ID":
        email = st.number_input(label="Input your customer_ID:",format='%i',step=1)
        if email!=0:
            if int(email) in Customer['customer_ID']:
                email=Customer[Customer['customer_ID']==int(email)]
                st.dataframe(email,use_container_width=True)
            else:
                st.write("Not found customer_ID!")
    

    

