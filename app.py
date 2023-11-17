from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('./vgg_spill_detect_model.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')  
    img = img.resize((224, 224))
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        image_array = preprocess_image(file)
        prediction = model.predict(np.array([image_array]))
        print(prediction)
        result = {'prediction': float(prediction[0][1])}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/about')
def about():
    return 'About'