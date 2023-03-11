import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
st.title('Revenue Prediction')
a = st.number_input('Input Temperature')
if st.button('Predict'):
    a = np.array(a).reshape(-1,1)
    df = pd.read_csv(r'C:\Users\Admin\Downloads\emhuy\IceCreamData.csv')
    df.head()
    x = df['Temperature'].values
    y = df['Revenue'].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=50)
    model = LinearRegression()
    x = x.reshape(-1,1)
    model.fit(x, y)
    y_pred = model.predict(a)
    st.success(y_pred)