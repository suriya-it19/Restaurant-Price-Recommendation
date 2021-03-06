import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import cv2

st.title('Analytics')

"""
# Business Analysis of Restautant's profit
Using People Counter and Machine learning
"""

if st.button('Start'):
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FPS, 25)

    image_placeholder = st.empty()

    while True:
        success, image = video.read()
        if not success:
            break
        image_placeholder.image(image, channels="BGR")
        time.sleep(0.01)
        


df = pd.DataFrame({
  'Date': '08/02/2021',
  'Price': 'prediction',
  'Popular Dishes' : 'prediction',
  'Quantity of the dishes' : 'prediction',
  'Discount value' : 'prediction'
}, index=[0])

df

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

    option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected:', option

left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")