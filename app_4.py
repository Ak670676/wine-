import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Function to load data based on selected wine type
def get_data(dataset):
    data_red = pd.read_csv('C:/Users/Ankit/Desktop/wine/wine_project/Data/winequality-red.csv',  delimiter = ';')
    data_white = pd.read_csv('C:/Users/Ankit/Desktop/wine/wine_project/Data/winequality-white.csv',  delimiter = ';')
    if dataset == 'Red Wine':
        data = data_red
    else:
        data = data_white
    return data

# Function to get dataset based on selected wine type
def get_dataset(dataset):
    x = dataset.drop(columns=['quality'])
    y = dataset['quality']
    return x, y

# Function to make predictions based on user input features
def quality_prediction(model, input_features):
    y_custom_predict = model.predict(input_features)
    st.write(f'prediction  is {y_custom_predict}')
    if y_custom_predict > 5:
        text = '**Your Wine is of Good quality**'
    else:
        text = '**Your Wine is of Bad quality**'
    return text

# Load image
image = Image.open('C:/Users/Ankit/Desktop/wine/wine_project/wine_image.png')

# Set title and image
st.title('Wine Quality Prediction Random forest')
st.write("Ankit kumar (23D1604) ")
st.image(image, caption='wine Quality', use_column_width=True)

# Select wine type
dataset = st.sidebar.selectbox('Select Wine type', ('Red Wine', 'White Wine'))

# Load data based on selected wine type
data = get_data(dataset)
x, y = get_dataset(data)

# Sidebar for prediction
st.sidebar.subheader('**Quality Prediction:**')
n_estimators = st.sidebar.slider('n_estimators', 100, 1000)
max_depth = st.sidebar.slider('max_depth', 1, 15)
params = {'n_estimators': n_estimators, 'max_depth': max_depth}

model = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
sc = StandardScaler()
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
model.fit(x_train, y_train)

# Main content area
col1, col2 = st.columns(2)

# Visualization section
with col1:
    st.header('Data Visualisation')
    plot = st.selectbox('Select Plot type', ('Histogram', 'Box Plot', 'Heat Map'))

    if plot == 'Heat Map':
        fig1 = plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(data.corr()[['quality']].sort_values(by='quality', ascending=False), vmin=-1,
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
            sns.distplot(data[feature])
            st.pyplot(fig2)
        else:
            fig3 = plt.figure(figsize=(3, 3))
            plt.xlabel(feature)
            plt.boxplot(x=data[feature])
            st.pyplot(fig3)

# Prediction section
with col2:
    st.header('Prediction')
    st.write('**If you want to make quality prediction for custom input values, enter the values below**')

    input_features = {}
    for feature in ('fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'):
        input_features[feature] = st.slider(f'{feature}', min_value=0.0, max_value=100.0, step=0.1)

    input_features_df = pd.DataFrame(input_features, index=[0])

    # Button to trigger prediction
    if st.button('Predict'):
        prediction_text = quality_prediction(model, input_features_df)
        st.write(prediction_text)

# Apply padding to increase the gap between columns
col1.empty()
col2.empty()
col1.markdown("&nbsp;")
col2.markdown("&nbsp;")
