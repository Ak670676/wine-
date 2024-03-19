#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image


st.title('Wine Quality Prediction Random forest')

# st.markdown('* **1-5 = Bad**')
# st.markdown('* **6-10 = Good**')
image = Image.open('C:/Users/Ankit/Desktop/wine/wine_project/wine_image.png')
st.image(image, caption='wine Quality',use_column_width=True)
st.subheader('**Quality Index:**')
dataset = st.selectbox('Select Wine type', ('Red Wine', 'White Wine'))

def get_data(dataset):
    data_red = pd.read_csv('C:/Users/Ankit/Desktop/wine/wine_project/Data/winequality-red.csv',  delimiter = ';')
    data_white = pd.read_csv('C:/Users/Ankit/Desktop/wine/wine_project/Data/winequality-white.csv',  delimiter = ';')
    if dataset == 'Red Wine':
        data = data_red
    else:
        data = data_white
    return data

data_heatmap = get_data(dataset)
data = get_data(dataset)

def get_dataset(dataset):
    # bins = (1, 5, 10)
    # groups = ['1', '2']
    # data['quality'] = pd.cut(data['quality'], bins=bins, labels=groups)
    # x= data['quality'].apply(lambda x:0 if x<=5 else 1)
    x = data.drop(columns=['quality'])
    y = data['quality']#.apply(lambda y_value:1 if y_value>=7 else 0)
    return x, y

x, y = get_dataset(data)
st.write('Shape of dataset:', data.shape)

with st.expander('Data Visualisation'):
    plot = st.selectbox('Select Plot type', ('Histogram', 'Box Plot', 'Heat Map'))

    if plot == 'Heat Map':
        fig1 = plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(data_heatmap.corr()[['quality']].sort_values(by='quality', ascending=False), vmin=-1,
                              vmax=1, annot=True)
        heatmap.set_title('Features Correlating with quality', fontdict={'fontsize': 18}, pad=16)
        st.pyplot(fig1)
    else:
        feature = st.selectbox('Select Feature', ('fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                                  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                                  'pH', 'sulphates', 'alcohol'))
        if plot == 'Histogram':
            fig2 = plt.figure(figsize=(7, 5))
            plt.xlabel(feature)
            sns.distplot(x[feature])
            st.pyplot(fig2)
        else:
            fig3 = plt.figure(figsize=(3, 3))
            plt.xlabel(feature)
            plt.boxplot(x=x[feature])
            st.pyplot(fig3)

with st.expander('Prediction'):
    n_estimators = st.slider('n_estimators', 100, 1000)
    max_depth = st.slider('max_depth', 1, 15)

    params = {'n_estimators': n_estimators, 'max_depth': max_depth}

    model = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])

    sc = StandardScaler()
    x = sc.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    accuracy1 = accuracy_score(y_test, y_predict)
    accuracy1 = accuracy1 * 100
    accuracy1 = round(accuracy1, 2)
    st.write(f'Accuracy is {accuracy1}%')

    st.write('**If you want to make quality prediction for custom input values, enter the values below**')

    def user_input_features():
        fixed_acidity = st.slider('Fixed Acidity', min_value=0.0, max_value=100.0, step=0.1)
        volatile_acidity = st.slider('Volatile Acidity', min_value=0.0, max_value=2.0, step=0.01)
        citric_acid = st.slider('Citric Acid', min_value=0.0, max_value=1.0, step=0.01)
        residual_sugar = st.slider('Residual Sugar', min_value=0.0, max_value=100.0, step=0.1)
        chlorides = st.slider('Chlorides', min_value=0.0, max_value=1.0, step=0.01)
        free_sulfur_dioxide = st.slider('Free sulfur dioxide', min_value=0, max_value=100, step=1)
        total_sulfur_dioxide = st.slider('Total sulfur dioxide', min_value=0, max_value=300, step=1)
        density = st.slider('Density', min_value=0.0, max_value=2.0, step=0.001)
        pH = st.slider('pH', min_value=0.0, max_value=14.0, step=0.01)
        sulphates = st.slider('Sulphates', min_value=0.0, max_value=2.0, step=0.01)
        alcohol = st.slider('Alcohol', min_value=8.0, max_value=16.0, step=0.1)

        to_predict = {'fixed_acidity': fixed_acidity, 'volatile_acidity': volatile_acidity,
                      'citric_acid': citric_acid, 'residual_sugar': residual_sugar,
                      'chlorides': chlorides, 'free_sulfur_dioxide': free_sulfur_dioxide,
                      'total_sulfur_dioxide': total_sulfur_dioxide, 'density': density, 'pH': pH,
                      'sulphates': sulphates, 'alcohol': alcohol}
        df = pd.DataFrame(to_predict, index=[0])
        return df

    df = user_input_features()

    def quality_prediction():
        y_custom_predict = model.predict(df)
        st.write(f'prediction  is {y_custom_predict}')
        if y_custom_predict >5:
            text = '**Your Wine is of Good quality**'
        else:
            text = '**Your Wine is of Bad quality**'
        return text

    text = quality_prediction()

    if st.button('Predict'):
        st.write(text)
