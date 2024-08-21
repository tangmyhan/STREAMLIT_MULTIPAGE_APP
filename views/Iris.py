import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


st.title("Iris Flower Prediction 🪻")
st.write("This app predict the Iris flower typer")

st.sidebar.header('User Input Parameter')


def user_input_feature():
    sepal_length = st.sidebar.slider('sepal_length', 4.3, 7.9, 4.5)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 2.5)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 2.5)	
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 1.0)
    
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    feature = pd.DataFrame(data, index=[0])
    return feature

df = user_input_feature()
model_name = st.sidebar.selectbox('Pick a model', ['Random Forest','KNN','Logistic Regression','SVM','Decision Tree'])


st.subheader('User Input Parameter')
st.write(df)

iris = datasets.load_iris()

# iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
# st.write(iris_df.describe())

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)



if model_name == 'Random Forest':
    clf = RandomForestClassifier()
elif model_name == 'KNN':
    clf = KNeighborsClassifier()
elif model_name == 'Logistic Regression':
    clf = LogisticRegression()
elif model_name == 'SVM':
    clf = SVC(gamma='auto',probability=True)
elif model_name == 'Decision Tree':
    clf = DecisionTreeClassifier()

clf.fit(X_train,Y_train)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Model Accuracy')
st.write(f"Accuracy of {model_name}: {accuracy:.2f}")
