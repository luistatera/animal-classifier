import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model (you can swap with 'best_resnet_model.keras' if preferred)
model = tf.keras.models.load_model("best_cnn_model.keras")

# Define the class labels in the same order used during training
class_names = ['cat', 'dog', 'horse', 'spider', 'butterfly', 'chicken', 'sheep', 'cow', 'squirrel', 'elephant']

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # adjust if your model used another size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)[0]
    top_idx = np.argmax(predictions)
    return class_names[top_idx], float(predictions[top_idx])
