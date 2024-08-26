import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

st.title('Automatic EDA Dashboard :bar_chart: :computer:')
st.caption("Upload CSV file to see various Charts related to EDA. Please upload file that has both continuous columns and categorical columns. Once you upload file, various charts, widgets and basic stats will be displayed. As a sample example, you can upload famous <a href='https://www.kaggle.com/competitions/titanic/data?select=train.csv'>Titanic Dataset</a> available from Kaggle.", unsafe_allow_html=True)
upload = st.file_uploader('Upload File Here',type=['csv'])

# def create_correlation_chart(corr_df): ## Create Correlation Chart using Matplotlib
#     fig = plt.figure(figsize=(10,10))
#     plt.imshow(corr_df.values, cmap="Blues")
#     plt.xticks(range(corr_df.shape[0]), corr_df.columns, rotation=90, fontsize=15)
#     plt.yticks(range(corr_df.shape[0]), corr_df.columns, fontsize=15)
#     plt.colorbar()

#     for i in range(corr_df.shape[0]):
#         for j in range(corr_df.shape[0]):
#             plt.text(i,j, "{:.2f}".format(corr_df.values[i, j]), color="red", ha="center", fontsize=14, fontweight="bold")

#     return fig

def create_missing_values_bar(df):
    missing_fig = plt.figure(figsize=(10,5))
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

    ax = missing_fig.add_subplot(111)
    ax.barh(missing_values.index, missing_values.values, color='skyblue')
    ax.set_xlabel("Number of Missing Values", fontsize=12)
    ax.set_title("Missing Values Distribution", fontsize=14)
    plt.tight_layout()

    return missing_fig

def find_cat_cont_columns(df): # Separate Continuous & Categorical Columns
    cont_columns, cat_columns = [], []
    for col in df.columns:
        if len(df[col].unique()) <= 25 or df[col].dtype == np.object_:
            cat_columns.append(col.strip())
        else:
            cont_columns.append(col.strip())
    return cont_columns, cat_columns

titanic_path = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

if upload is None:  # File as Bytes
    upload = titanic_path

df = pd.read_csv(upload)

st.write('\n')
     
tab1, tab2, tab3 = st.tabs(["Dataset Overview :clipboard:", "Individual Column Stats :bar_chart:", "Explore Relation Between Features :chart:"])

with tab1: # Dataset Overview
    st.subheader('1. Dataset')
    st.write(df)

    cont_columns, cat_columns = find_cat_cont_columns(df)

    st.subheader('2. Dataset Overview')
    st.write('Rows:', df.shape[0])
    st.write('Duplicates:', df.duplicated().sum())
    st.write('Features:', df.shape[1])
    st.write('Categorical Columns:', len(cat_columns))
    st.write(cat_columns)
    st.write('Continuous Columns:', len(cont_columns))
    st.write(cont_columns)

    st.subheader('3. Correlation Chart')

    numeric_df = df.select_dtypes(include=[np.number])

    # corr_df = df[cont_columns].corr() 
    # corr_fig = create_correlation_chart(corr_df)
    # st.pyplot(corr_fig, use_container_width=True)

    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10,8))
    
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f")
    st.pyplot(plt)
    
    st.subheader('4. Missing Values Distribution')
    missing_fig = create_missing_values_bar(df)
    st.pyplot(missing_fig, use_container_width=True)

with tab2:
    df_descr = df.describe()
    st.subheader("Analyze Individual Feature Distribution")
    st.write('\n')
    st.write(df_descr)

    st.markdown("#### 1. Understand Continuous Feature")      
    feature = st.selectbox(label="Select Continuous Feature", options=cont_columns, index=0)
    
    na_cnt = df[feature].isna().sum()

    st.write('Count: ', df_descr[feature]['count'])
    missing_percent = (na_cnt / df.shape[0])*100
    st.write('Missing Count', na_cnt, f"({missing_percent:.2f}%)")
    st.write('Mean: ', f"{df_descr[feature]['mean']:.2f}")
    st.write('Standard Deviation: ', f"{df_descr[feature]['std']:.2f}")
    st.write('Minimum:', df_descr[feature]['min'])
    st.write('Maximun:', df_descr[feature]['max'])
    st.write('Quantiles: ', df_descr[[feature]].T[['25%', "50%", "75%"]])

    # Histogram
    hist_fig = plt.figure(figsize=(10,6))
    plt.hist(df[feature].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    st.pyplot(hist_fig, use_container_width=True)

    st.markdown("#### 2. Understand Categorical Feature")
    feature = st.selectbox(label="Select Categorical Feature", options=cat_columns, index=0)

    ### Categorical Columns Distribution
    values_cnt = df[feature].value_counts()
    cat_fig = plt.figure(figsize=(10,6))
    plt.bar(values_cnt.index, values_cnt.values, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(cat_fig, use_container_width=True)