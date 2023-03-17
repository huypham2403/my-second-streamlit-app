import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_csv('https://raw.githubusercontent.com/huypham2403/my-second-streamlit-app/main/IceCreamData.csv')
df.head()
x = df['Temperature'].values
y = df['Revenue'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=50)
model = LinearRegression()
x = x.reshape(-1,1)
model.fit(x, y)
st.title('Revenue Prediction')
a = st.number_input('Input Temperature')
if st.button('Predict'):
    a = np.array(a).reshape(-1,1)
    y_pred = model.predict(a)
    st.caption('Revenue Prediction')
    st.success(y_pred)
filename = 'my-second-streamlit-app-main'
pickle.dump(model, open(filename, "wb"))
model = pickle.load(open(filename, "rb"))
