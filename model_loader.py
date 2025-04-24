import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

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
    return ["prj_best_cnn_model.keras", "best_resnet_model.pth"]

def load_model(model_name):
    """Load a specific model by name"""
    model_path = os.path.join("models", model_name)
    print(f"Loading model from: {model_path}")
    
    if model_name not in get_available_models():
        raise ValueError("Invalid model name. Must be either 'prj_best_cnn_model.keras' or 'best_resnet_model.pth'")
    
    if model_name.endswith('.keras'):
        model = tf.keras.models.load_model(model_path)
        print("Loaded TensorFlow model successfully")
        return model, 'tf'
    else:
        # Load PyTorch model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load ResNet model with modified fc layer to match saved architecture
        model = models.resnet18(pretrained=False)
        # Replace the fc layer with a sequential module to match the saved state dict
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, len(class_names))
        )
        
        # Load the saved state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Loaded PyTorch model successfully")
        return model, 'torch'

def predict_label(img_path, model_name="prj_best_cnn_model.keras", confidence_threshold=None):
    """Predict the label for an image using the specified model"""
    print(f"\nMaking prediction for image: {img_path}")
    print(f"Using model: {model_name}")
    
    # Set model-specific confidence thresholds
    if confidence_threshold is None:
        confidence_threshold = 0.60 if model_name.endswith('.keras') else 0.70
    
    model, framework = load_model(model_name)
    
    if framework == 'tf':
        # TensorFlow prediction path
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)[0]
    else:
        # PyTorch prediction path
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        with torch.no_grad():
            outputs = model(img_tensor)
            predictions = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
    
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
