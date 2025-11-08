# 🎭 Face Emotion Detection Web App
A Flask + TensorFlow web application that detects and classifies human emotions from facial images. Users can upload a photo, and the system predicts the emotion displayed on the face — such as Happy, Sad, Angry, Surprise, Neutral, Fear, or Disgust.

🚀 Features

Detects emotions from face images

Built with TensorFlow (CNN model)

Simple Flask web interface

Upload and analyze your own images

Deployed online using Render

🧠 Model Overview

The model was trained using a Convolutional Neural Network (CNN) on facial expression data.
Output classes include:

Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral

🗂️ Project Structure
FACE_DETECTION/
│
├── app.py                 # Flask app entry point
├── face_emotionModel.h5   # Trained emotion detection model
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── static/
│   └── (CSS, JS, or images)
└── runtime.txt            # Python version for Render
