import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
st.header('Image Classification Model')
model = load_model(r'C:\Users\asus\OneDrive - SRM Institute of Science & Technology\Desktop\Fruits_vegetable_cnn\Image_Classify.keras')
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

image_length = 180
image_width = 180
image = st.text_input('Enter Image name','banana.jpg') 
image_load = tf.keras.utils.load_img(image, target_size=(image_length,image_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)
st.image(image,width=300)
score = tf.nn.softmax(predict)
st.write('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))
