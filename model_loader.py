import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Define the class labels in the same order used during training
class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# English translations for reference:
# cane = dog
# cavallo = horse
# elefante = elephant
# farfalla = butterfly
# gallina = chicken
# gatto = cat
# mucca = cow
# pecora = sheep
# ragno = spider
# scoiattolo = squirrel

def get_available_models():
    """Return a list of available model files"""
    return ["prj_best_cnn_model.keras", "prj_best_resnet_model.keras"]

def load_model(model_name):
    """Load a specific model by name"""
    model_path = os.path.join("models", model_name)
    print(f"Loading model from: {model_path}")
    
    if model_name not in get_available_models():
        raise ValueError("Invalid model name. Must be either 'prj_best_cnn_model.keras' or 'prj_best_resnet_model.keras'")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    return model

def predict_label(img_path, model_name="prj_best_cnn_model.keras", confidence_threshold=0.70):
    """Predict the label for an image using the specified model"""
    print(f"\nMaking prediction for image: {img_path}")
    print(f"Using model: {model_name}")
    
    model = load_model(model_name)
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)[0]
    print(f"Raw predictions: {predictions}")
    
    # Get top 3 predictions and their confidences
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_confidences = predictions[top_indices]
    
    print("\nTop 3 predictions:")
    for idx, conf in zip(top_indices, top_confidences):
        print(f"{class_names[idx]}: {conf:.4f}")
    
    # Return the highest confidence prediction if it meets the threshold
    top_idx = top_indices[0]
    confidence = float(predictions[top_idx])
    
    # If confidence is below threshold, return None to indicate no confident prediction
    if confidence < confidence_threshold:
        print(f"\nConfidence {confidence:.4f} below threshold {confidence_threshold}")
        return None, confidence
    
    print(f"\nFinal prediction: {class_names[top_idx]} with confidence {confidence:.4f}")
    return class_names[top_idx], confidence
