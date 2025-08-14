import tensorflow as tf
from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__, static_folder='static')
model = tf.keras.models.load_model('my_model.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/predict', methods=['POST'])
def pimage():
    if 'pc_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['pc_image']
    
    # Use static/uploads folder for serving images
    UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    img_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(img_path) 
    
    print(f"Image saved at: {img_path}")  # Debugging line

    if not os.path.exists(img_path):
        print("ERROR: Image not saved correctly!")
    
    # Preprocess the image
    img = load_img(img_path, target_size=(224, 224))
    image_array = img_to_array(img) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    pred = np.argmax(model.predict(image_array), axis=1)[0]
    index = ['Biodegradable Images(0)', 'Recyclable Images(1)', 'Trash Images(2)']
    prediction = index[pred]

    # Generate URL for the image to be served statically
    image_url = url_for('static', filename=f'uploads/{f.filename}')
    return render_template("predict.html", predict=prediction, image_url=image_url)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/service-details')
def service_details():
    return render_template("service-details.html")

@app.route('/starter-page')
def starter_page():
    return render_template("starter-page.html")

@app.route('/predict', methods=['GET'])
def show_predict():
    return render_template("predict.html")

if __name__ == '__main__':
    print("starting flask app")
    app.run(debug=True, port=5001)