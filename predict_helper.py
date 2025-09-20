
# predictor helper (use in backend)
import tensorflow as tf, numpy as np, pickle
from PIL import Image
def preprocess_img_for_mobilenet(path, size=(224,224)):
    img = Image.open(path).convert('RGB').resize(size)
    arr = np.array(img).astype('float32')
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input(arr)
# Load models:
# embedding_model = tf.keras.models.load_model('exported_model_package/embedding_model_saved.keras') # Corrected extension
# classifier_model = tf.keras.models.load_model('exported_model_package/breed_classifier_saved.keras') # Corrected extension
# metadata = pickle.load(open('exported_model_package/metadata.pkl', 'rb'))
# Example usage:
# emb = embedding_model.predict(arr[None,...])
# probs = classifier_model.predict(arr[None,...])[0]
# ... compute top2 and health as cosine(emb, centroid)
