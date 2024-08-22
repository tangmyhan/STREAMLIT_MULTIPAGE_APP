import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

st.write("## Hyperparameter Tuning with Scikit-Learn!")
st.write("Tune **parameters** with Scikit-Learn's *GridSearchCV*")

col1, col2 = st.columns(2)
about_expander = col1.expander("About",expanded=False)
about_expander.info('''
                    This web application is a simple demonstration of Hyperparameter tuning with 
                    **GridSearchCV**. The parameters customizable in this app are only limited 
                    and the algorithms and datasets used are from Scikit learn. There may be other combinations 
                    of parameters and algorithms that can yield a better accuracy score for a given dataset.
                    ''')

info_expander = col2.expander("What is Hyperparameter Tuning?",expanded=False)
info_expander.info('''
                    **Hyperparameters** are the parameters that describe the model architecture and 
                    **hyperparameter tuning** is the method of looking for the optimal model architecture
                   ''')

st.sidebar.header('Select Dataset')

datasets_name = st.sidebar.selectbox('Select Dataset',['Iris Flower','Wine Recognition'])
st.write('')
st.write(f"### **{datasets_name} Dataset**")

model_name = st.sidebar.selectbox('Pick a model', ['Random Forest','KNN','Logistic Regression','SVM'])

cv_count = st.sidebar.slider('Cross-validation count', 2, 5, 3)

st.sidebar.write('---')
st.sidebar.header('User Input Parameters')
st.sidebar.write('')

def get_dataset(name):
    df = None
    if name == 'Iris Flower':
        df = datasets.load_iris()
    elif name == 'Wine Recognition':
        df = datasets.load_wine()
    X = df.data
    Y = df.target
    return X, Y

X, Y = get_dataset(datasets_name)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=5)

st.write('Shape of dataset:', X.shape)
st.write('Number of classes:', len(np.unique(Y)))

def get_model(clf_name):
    clf = None
    parameters = None
    if clf_name == 'Random Forest':
        st.sidebar.subheader('Number of Estimators')
        st.sidebar.write('The number of trees in the forest.')
        n1 = st.sidebar.slider('n_estimators 1', 1, 40, 5)
        n2 = st.sidebar.slider('n_estimators 2', 41, 80, 50)
        n3 = st.sidebar.slider('n_estimators 3', 81, 120, 100)
        st.sidebar.write('\n')

        st.sidebar.subheader('Max depth')
        st.sidebar.write('The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.')
        m1 = st.sidebar.slider('max_depth 1', 1, 7, 2)
        m2 = st.sidebar.slider('max_depth 2', 8, 14, 10)
        m3 = st.sidebar.slider('max_depth 3', 15, 20, 20)

        parameters = {'n_estimators': [n1, n2, n3], 
                      'max_depth': [m1, m2, m3]}
        clf = RandomForestClassifier()

    elif clf_name == 'SVM':
        st.sidebar.subheader('Kernel Type')
        st.sidebar.write('Specifies the kernel type to be used in the algorithm.')
        kernel_type = st.sidebar.multiselect('',options=['linear', 'rbf', 'poly', 'sigmoid'], 
                                             default=['linear', 'rbf', 'poly'])
        st.sidebar.write('\n')

        st.sidebar.subheader('Regularization Parameter')
        st.sidebar.write('The strength of the regularization is inversely proportional to C.')
        c1 = st.sidebar.slider('C1', 1, 7, 1)
        c2 = st.sidebar.slider('C2', 8, 14, 10)
        c3 = st.sidebar.slider('C3', 15, 20, 20)
        st.sidebar.write('\n')

        st.sidebar.subheader('Gamma')
        st.sidebar.write("Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.")
        gamma1 = st.sidebar.slider('Gamma 1', 0.001, 0.01, 0.001)
        gamma2 = st.sidebar.slider('Gamma 2', 0.01, 0.1, 0.01)
        gamma3 = st.sidebar.slider('Gamma 3', 0.1, 1.0, 1.0)

        parameters = {'kernel': kernel_type,
                      'C': [c1, c2, c3],
                      'gamma': [gamma1, gamma2, gamma3]}
        clf = SVC()

    elif clf_name == 'Logistic Regression':
        st.sidebar.subheader('Penalty')
        st.sidebar.write('Used to specify the norm used in the penalization.')
        penalty_type = st.sidebar.multiselect('',options=['l1', 'l2', 'elasticnet'],
                                              default=['l1','l2'])
        st.sidebar.write('\n')
        st.sidebar.subheader('Regularization Parameter')
        st.sidebar.write('Inverse of regularization strength; must be a positive float.')
        c1 = st.sidebar.slider('C1', 0.01, 1.0, 0.1)
        c2 = st.sidebar.slider('C2', 2, 19, 5)
        c3 = st.sidebar.slider('C3', 20, 100, 80, 10)

        parameters = {'penalty': penalty_type,
                      'C': [c1, c2, c3]}
        clf = LogisticRegression()

    else:
        st.sidebar.subheader('Number of neighbors')
        st.sidebar.write('Number of neighbors to use by default for `kneighbors` queries.')
        k1 = st.sidebar.slider('n_neighbors 1', 1, 5, 2)
        k2 = st.sidebar.slider('n_neighbors 2', 6, 10, 7)
        k3 = st.sidebar.slider('n_neighbors 3', 11, 15, 13)
        parameters = {'n_neighbors': [k1, k2, k3]}
        clf = KNeighborsClassifier()

    return clf, parameters

clf, parameters = get_model(model_name)

grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=cv_count,return_train_score=False)
grid_search.fit(X,Y)

df = pd.DataFrame(grid_search.cv_results_)

st.write("### **Tuning Results**")
results_df = st.multiselect('', options=['params','mean_fit_time', 'std_fit_time', 'mean_score_time',
                                         'std_score_time', 'split0_test_score', 'split1_test_score', 
                                         'split2_test_score', 'std_test_score','mean_test_score', 'rank_test_score'],
                                default=['mean_score_time', 'std_score_time',
                                        'split0_test_score', 'split1_test_score', 
                                        'split2_test_score'])
df_results = df[results_df]
st.write(df_results)

st.write("### **Parameters and Mean test score**")

st.write(df[['params', 'mean_test_score']])

st.write('Best score:', grid_search.best_score_)

st.write('Best parameters:', grid_search.best_params_)
