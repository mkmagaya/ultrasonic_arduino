import streamlit as st
# import serial
import time
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import plotly.express as px
# from generator import *
from scipy.stats import pearsonr
from sklearn import linear_model, metrics
from sklearn.metrics import r2_score
from scipy import stats

# import subprocess

# subprocess.call("Script1.py", shell=True)


# st.set_page_config(layout = "wide")

# loading data captured from the sensor for processing
st.write("""
        # Tracking Project
        ### Collecting** Distance** and **Time** of a** Mine** ###
        #""")
 
st.header('Visualising relationship between Distance and Time variables')
st.subheader('Data analysis')

# def gather_data():
#     number = st.number_input('enter the number of readings to capture into the dataset csv')
#     generate_dataset(number)

uploaded_file= st.file_uploader("Choose a file", type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")

if uploaded_file is not None:

    #read csv
    df1=pd.read_csv(uploaded_file)
    data = pd.DataFrame(df1,columns=['Distance', 'Time'])
    st.dataframe(data)
else:
    st.warning('Please select a data source👆')
    # from generator import generate_dataset
    
    # st.subheader('Create a Dataset')


    # if st.button("Generate Dataset"):

    #     driver=GenerateDataset()
    #     # records=st.number_input('enter the number of readings to capture into the dataset csv')
    #     records=int(input(num))
    #     driver.main(records)
    #         # exec(open("generator.py").read())
    #         # gather_data()

def plot_graph(dataset):
    fig = plt.figure(figsize = (10, 5))
    g = sns.pairplot(dataset, dropna = True, diag_kind="kde")
    fig3 = px.scatter(dataset, x=dataset.Time, y=dataset.Distance)
    g.map_lower(sns.regplot)
    plt.plot(dataset.Time, dataset.Distance)
    plt.scatter(dataset.Time, dataset.Distance)
    # plt.px.scatter(dataset.Time, dataset.Distance, color="species")
    plt.ylabel('Distance')
    plt.title('Distance vs. Time')
    # plt.show()
    # st.pyplot(fig)
    # st.subheader('Pairplot')
    # st.pyplot(g)
    st.subheader('Distance time Graph')
    h = st.plotly_chart(fig3)
    gradient=np.gradient(dataset)
    st.balloons()


#Correlation calculations (Pearson)
def calc_corr1(Distance,Time):
    corr1, p_val1 = stats.pearsonr(Time, Distance)
    return corr1, p_val1

#Correlation calculations (Spearman)
def calc_corr2(Distance, Time):
    corr2, p_val2 = stats.spearmanr(Time, Distance)
    return corr2, p_val2

# st.subheader('Scatterplot analysis')
# selected_x_var = st.selectbox('What do you want the x variable to be?', df.columns)
# selected_y_var = st.selectbox('What about the y?', df.columns)
# fig = px.scatter(df, x = df[selected_x_var], y = df[selected_y_var], color="species")
# st.plotly_chart(fig)
    
st.subheader('Data Visualizations')
if st.button('Plot Graph'):
    plot_graph(data)
    filename=uploaded_file.name
    st.write('Graph for: ', filename)
if st.button('Evaluation and Recommendation'):
    x1 = data.Distance.to_numpy()
    y1 = data.Time.to_numpy()
    x2 = data.Distance.to_numpy()
    y2 = data.Time.to_numpy()
    correlation1, corr_p_val1 = calc_corr1(x1,y1)
    correlation2, corr_p_val2 = calc_corr2(x2,y2)
    st.subheader("Pearson Correlation")
    st.write('Pearson correlation coefficient: %.3f' % correlation1)
    st.write('p value: %.3f' % corr_p_val1)
    # st.subheader("Spearman Correlation")
    # st.write('Spearman correlation coefficient: %.3f' % correlation2)
    # st.write('p value: %.3f' % corr_p_val2)
    st.subheader("Suggested Recommendation")
    if correlation1>0:
        st.write('condition 1: strong relationship between Distance and Time')
    elif correlation1<0:
        st.write('condition 2')
    elif correlation1==-1:
        st.write('condition 3')
    elif correlation1==1:
        st.write('condition 4')
    elif correlation1==0:
        st.write('condition 4')


#in case we want to generate new data and create a dataset saved with date

# if st.button('Recommend'):
#     if correlation1>0 & correlation2 >0:
#         st.write('condition 1: strong relationship between Distance and Time')
#     elif correlation1<0 & correlation2<0:
#         st.write('condition 2')
#     elif correlation1==-1 & correlation2==-1:
#         st.write('condition 3')
#     elif correlation1==1 & correlation2==1:
#         st.write('condition 4')
#     elif correlation1==0 & correlation2==0:
#         st.write('condition 4')
# if st.button('Recommend'):
