from flask import Flask, render_template, request
import sqlite3, os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Load trained model once at startup
MODEL_PATH = 'face_emotionModel.h5'
model = load_model(MODEL_PATH)

# Class labels used during training
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
#  Database Setup
# -------------------------------
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT,
                    image_path TEXT,
                    predicted_emotion TEXT,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# -------------------------------
#  Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    file = request.files['image']

    if not file:
        return "No file uploaded.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, color_mode='grayscale', target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict emotion
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]

    # Custom message
    emotion_messages = {
        'happy': "You look happy today!",
        'sad': "You are frowning. Why are you sad?",
        'angry': "You look a bit angry. Take a deep breath.",
        'surprise': "You seem surprised!",
        'neutral': "You look calm and neutral.",
        'fear': "You seem scared. Everything will be fine.",
        'disgust': "Hmm... you look disgusted!"
    }

    message = emotion_messages.get(predicted_label, "Emotion detected.")

    # Save to database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO students (name, email, image_path, predicted_emotion, timestamp) VALUES (?, ?, ?, ?, ?)",
              (name, email, filepath, predicted_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return render_template('index.html', 
                           name=name, 
                           email=email, 
                           emotion=predicted_label, 
                           message=message, 
                           image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
