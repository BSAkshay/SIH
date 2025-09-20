from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load models and metadata
def load_models():
    try:
        embedding_model = tf.keras.models.load_model('exported_model/embedding_model.h5')
        classifier_model = tf.keras.models.load_model('exported_model/breed_classifier.h5')
        
        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            
        return embedding_model, classifier_model, metadata
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

embedding_model, classifier_model, metadata = load_models()

def preprocess_img_for_mobilenet(path, size=(224, 224)):
    img = Image.open(path).convert('RGB').resize(size)
    arr = np.array(img).astype('float32')
    return preprocess_input(arr)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img_array = preprocess_img_for_mobilenet(filepath)
            
            emb = embedding_model.predict(img_array[None, ...])
            probs = classifier_model.predict(img_array[None, ...])[0]
            
            top2_idx = np.argsort(probs)[-2:][::-1]
            top2_breeds = [metadata['idx_to_label'][idx] for idx in top2_idx]
            top2_confidences = [float(probs[idx]) for idx in top2_idx]
            
            centroid = metadata['centroid_arr'][top2_idx[0]]
            health_score = cosine_similarity(emb, centroid[None, ...])[0][0]
            
            result = {
                'top_breed': top2_breeds[0],
                'top_confidence': top2_confidences[0],
                'second_breed': top2_breeds[1],
                'second_confidence': top2_confidences[1],
                'health_score': float(health_score)
            }
            
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Prediction error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
