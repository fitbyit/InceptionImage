from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'  # Folder to save uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained InceptionV3 model with weights downloaded from TensorFlow Hub
def load_model():
    model = InceptionV3(weights='imagenet')  # Download weights from TensorFlow Hub
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_image = request.form.get('image')
        if selected_image:
            img_path = os.path.join('static', 'images', selected_image)
            prediction = predict_image(img_path)
            return render_template('result.html', prediction=prediction, image=selected_image)
    # List of image files in the static/images directory
    image_files = os.listdir('static/images')
    return render_template('index.html', image_files=image_files)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('result.html', prediction=prediction, image=filename)
    return render_template('upload.html')

@app.route('/delete/<filename>', methods=['POST'])
def delete(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
    return redirect(url_for('index'))
    
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

if __name__ == '__main__':
    app.run(debug=True)
