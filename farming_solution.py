from flask import Flask, request, jsonify, render_template
import os
import joblib
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog

app = Flask(__name__)

model = joblib.load('random_forest_model_best.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def extract_features(image_path):
    image = imread(image_path)
    image_resized = resize(image, (64, 64), anti_aliasing=True)
    features_hog, _ = hog(image_resized, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return features_hog

@app.route('/')
def index():
    return render_template('interface.html')  # Update to refer to interface.html

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    features = extract_features(file_path)
    prediction = model.predict([features])
    nutrient_status = label_encoder.inverse_transform(prediction)[0]
    
    return jsonify({'result': nutrient_status})

if __name__ == '__main__':
    app.run(debug=True)
