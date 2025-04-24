import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

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
    """Return a list of available model files in the models directory"""
    models_dir = "models"
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith(('.keras', '.pth')):
            model_files.append(file)
    return model_files

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def load_model(model_name):
    """Load a specific model by name"""
    model_path = os.path.join("models", model_name)
    print(f"Loading model from: {model_path}")  # Debug print
    
    if model_name.endswith('.keras'):
        print("Loading Keras model")  # Debug print
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")  # Debug print
        return model
    else:
        print("Loading PyTorch model")  # Debug print
        # Load PyTorch model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")  # Debug print
        
        # Determine model architecture based on filename
        if 'resnet' in model_name.lower():
            print("Creating ResNet18 architecture")  # Debug print
            # For project ResNet model, load the saved model directly
            model = models.resnet18(weights=None)
            # Replace the fc layer with a sequential module to match the saved state dict
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, len(class_names))
            )
        else:
            print("Creating SimpleCNN architecture")  # Debug print
            model = SimpleCNN(num_classes=len(class_names))
        
        # Load the saved state dict
        try:
            state_dict = torch.load(model_path, map_location=device)
            print(f"State dict keys: {state_dict.keys()}")  # Debug print
            model.load_state_dict(state_dict)
            print("State dict loaded successfully")  # Debug print
        except Exception as e:
            print(f"Error loading state dict: {str(e)}")  # Debug print
            raise
            
        model = model.to(device)
        model.eval()
        return model

def predict_label(img_path, model_name="best_cnn_model.keras", confidence_threshold=0.30):
    """Predict the label for an image using the specified model"""
    print(f"\nMaking prediction for image: {img_path}")  # Debug print
    print(f"Using model: {model_name}")  # Debug print
    
    model = load_model(model_name)
    
    if isinstance(model, tf.keras.Model):
        print("Using TensorFlow prediction path")  # Debug print
        # TensorFlow prediction
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)[0]
        print(f"Raw predictions: {predictions}")  # Debug print
    else:
        print("Using PyTorch prediction path")  # Debug print
        # PyTorch prediction
        device = next(model.parameters()).device
        print(f"Model is on device: {device}")  # Debug print
        
        # Use the appropriate transforms based on the model
        if 'resnet' in model_name.lower():
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            predictions = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            print(f"Raw predictions: {predictions}")  # Debug print
            print(f"Predictions per class:")  # Debug print
            for i, (pred, class_name) in enumerate(zip(predictions, class_names)):
                print(f"{class_name}: {pred:.4f}")
    
    # Get top 3 predictions and their confidences
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_confidences = predictions[top_indices]
    
    print("\nTop 3 predictions:")  # Debug print
    for idx, conf in zip(top_indices, top_confidences):
        print(f"{class_names[idx]}: {conf:.4f}")
    
    # Return the highest confidence prediction if it meets the threshold
    top_idx = top_indices[0]
    confidence = float(predictions[top_idx])
    
    # If confidence is below threshold, return None to indicate no confident prediction
    if confidence < confidence_threshold:
        print(f"\nConfidence {confidence:.4f} below threshold {confidence_threshold}")  # Debug print
        return None, confidence
    
    print(f"\nFinal prediction: {class_names[top_idx]} with confidence {confidence:.4f}")  # Debug print
    return class_names[top_idx], confidence
