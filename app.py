import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
import mysql.connector
import requests

st.title('Emotion Detection and Music Recommendation')

# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return None

    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        st.error("Error: Could not capture an image from the webcam.")
        return None

# Function to predict emotion
def predict_emotion(image):
    # Load the emotion detection model
    emotion_model = tf.keras.models.load_model('best_model.h5')

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Make prediction
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    emotion_prediction = emotion_model.predict(image)
    predicted_emotion = emotions[np.argmax(emotion_prediction)]
    return predicted_emotion

# Function to recommend a song based on emotion
def recommend_song(emotion):
    # Replace this with your song recommendation logic using MySQL or an API
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Godwin12love$',  # Replace with your MySQL root user's password
        database='music_db_revised_setup'  # Replace with the name of your actual database
    )
    cursor = connection.cursor()

    query = """
    SELECT file_path FROM track_emotions 
    JOIN tracks ON track_emotions.track_id = tracks.track_id 
    JOIN emotions ON track_emotions.emotion_id = emotions.emotion_id 
    WHERE emotion_name = %s ORDER BY RAND() LIMIT 1
    """
    cursor.execute(query, (emotion,))
    track_url = cursor.fetchone()

    cursor.close()
    connection.close()

    if not track_url:
        return "No song found for this emotion."
    
    return track_url[0]

# Capture an image from the webcam
if st.button('Capture Image'):
    captured_image = capture_image()
    if captured_image is not None:
        st.image(captured_image, caption='Captured Image', use_column_width=True)

        # Predict emotion
        predicted_emotion = predict_emotion(captured_image)
        st.write(f'Predicted Emotion: {predicted_emotion}')

        # Recommend a song
        recommended_song = recommend_song(predicted_emotion)
        st.write(f'Recommended Song: {recommended_song}')

        # Play the recommended song
        audio_url = recommended_song
        st.audio(audio_url, format='audio/mp3')

        # Option to download the recommended song
        if st.button('Download Song'):
            st.audio(audio_url, format='audio/mp3', key='audio')
            st.write(f'Download the recommended song [here]({audio_url})')
