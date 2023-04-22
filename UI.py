import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Define a function to preprocess the user input
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28, 28))
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to a numpy array
    image = np.array(image)
    # Reshape the image to a 4D tensor
    image = image.reshape(-1, 28, 28, 1)
    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    return image

# Create the Streamlit app
def app():
    st.title('Handwritten Digit Recognition')
    st.write('Upload an image to see the prediction!')

    # Add a file uploader widget for the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # When the user clicks the "Predict" button, preprocess the input and make a prediction
    if st.button('Predict'):
        if uploaded_file is not None:
            # If the user uploaded an image, preprocess it and make a prediction
            image = Image.open(uploaded_file)
            image = np.array(image)
            x = preprocess_image(image)
            prediction = model.predict(x)
            digit = np.argmax(prediction)
            st.write('Prediction: ', digit)
        else:
            st.write('Please upload an image')

# Run the Streamlit app
if __name__ == '__main__':
    app()
