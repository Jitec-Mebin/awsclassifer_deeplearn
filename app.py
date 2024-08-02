
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('custom_cnn_model.h5')

def prepare_image(img):
    img = img.resize((150, 150))  # Resize image to match model input
    img = img.convert('RGB')      # Ensure image is in RGB format
    img = np.array(img)           # Convert to numpy array
    img = img / 255.0             # Normalize
    img = img.reshape(1, 150, 150, 3)  # Reshape for model prediction
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = Image.open(file.stream)
    img = prepare_image(img)
    predictions = model.predict(img)
   # predicted_class = np.argmax(predictions, axis=1)[0]

    predictions = model.predict(img)
    predicted_prob = predictions[0][0]  # Probability of the positive class
    predicted_class = 'DME' if predicted_prob > 0.5 else 'NORMAL'  # Convert to class label


    return jsonify({'label': (predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
