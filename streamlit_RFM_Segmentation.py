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

# 1. Read data
@st.cache_data(max_entries=1000)
def read_data(filepath):
    #read data
    df = pd.read_csv(filepath, delimiter ='\s+',header=None,names=["customer_ID","date", "quantity", "price"])
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

# 2. Data pre-processing

#Scaling data
@st.cache_data(max_entries=1000)
def preprocess(df):
    """Preprocess data for model clustering"""
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    norm = scaler.transform(df_log)
    return norm

#PCA
@st.cache_data(max_entries=1000)
def PCA_model(df):
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(df)
    PCA_components = pd.DataFrame(principalComponents)
    return PCA_components

# Upload file result
@st.cache_data(max_entries=1000)
def load_csv(file):
    csv=pd.read_csv(file)
    return csv

Customer = load_csv("result_KMeans.csv")

#4.Load & Train data
df=read_data("CDNOW_master.txt")
new_df=create_RFM(df)
df_scale=preprocess(new_df)
PCA_components=PCA_model(df_scale)
model = KMeans(n_clusters= 3, random_state=42)
model.fit(PCA_components.iloc[:,:1])
silhouette=silhouette_score(PCA_components.iloc[:,:1], model.labels_, metric='euclidean')

#Calculate average RFM values and size for each cluster:
rfm_rfm = new_df.assign(K_Cluster = model.labels_)
rfm_rfm["Segment"]=rfm_rfm['K_Cluster'].map(lambda x:"Loyal" if x ==1
                                              else "Promising" if x == 2 
                                              else "Hibernating")

rfm_agg2 = rfm_rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(1)
rfm_agg2.columns = rfm_agg2.columns.droplevel()
rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)
# Reset the index
rfm_agg2 = rfm_agg2.reset_index()


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
    type = st.radio("Upload new data (.txt) or Search by customer ID?", options=("Upload","Customer_ID"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt'])
        if uploaded_file_1 is not None:
            new_df1 = read_data(uploaded_file_1)
            new_df1= create_RFM(new_df1)
            lines = preprocess(new_df1)
            PCA_components = PCA_model(lines)
            lines=PCA_components.iloc[:,:1]  
            st.write("Content:")
            if len(lines)>0:      
                y_pred_new = model.predict(lines)    
                st.code("New predictions(Loyal: 1; Promising: 2; Hibernating: 0): " + str(y_pred_new))

                #Calculate average RFM values and size for each cluster:
                rfm_rfm_k3 = new_df1.assign(K_Cluster = y_pred_new)
                rfm_rfm_k3["Segment"]=rfm_rfm_k3['K_Cluster'].map(lambda x:"Loyal" if x ==1
                                              else "Promising" if x == 2 
                                              else "Hibernating")
                st.dataframe(rfm_rfm_k3)
                #csv
                csv=rfm_rfm_k3.to_csv("data.csv")
                with open('data.csv', 'r') as f:
	                st.download_button('Download result file', f,file_name='data.csv', mime='text/csv')

    if type=="Customer_ID":
        email = st.number_input(label="Input your customer_ID:",format='%i',step=1)
        if email!=0:
            if int(email) in Customer['customer_ID']:
                email=Customer[Customer['customer_ID']==int(email)]
                st.code(email)
            else:
                st.write("Not found customer_ID!")
    

    

