import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Define the class labels in the same order used during training
CLASS_NAMES = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

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

def preprocess_image_tf(image_path, target_size=(224, 224)):
    """
    Preprocess image for TensorFlow model
    """
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize to [-1, 1] range instead of just [0, 1]
        img_array = (img_array / 127.5) - 1
        
        print(f"\nPreprocessing Debug (TensorFlow):")
        print(f"Image shape: {img_array.shape}")
        print(f"Value range: [{np.min(img_array)}, {np.max(img_array)}]")
        return img_array
    except Exception as e:
        print(f"Error in TensorFlow preprocessing: {str(e)}")
        raise

def preprocess_image_torch(image_path, target_size=(128, 128)):
    """
    Preprocess image for PyTorch model
    """
    try:
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        print(f"\nPreprocessing Debug (PyTorch):")
        print(f"Image shape: {img_tensor.shape}")
        print(f"Value range: [{torch.min(img_tensor).item()}, {torch.max(img_tensor).item()}]")
        return img_tensor
    except Exception as e:
        print(f"Error in PyTorch preprocessing: {str(e)}")
        raise

def get_available_models():
    """Return a list of available model files"""
    return ["best_cnn_model_luis.pth", "best_resnet_model.pth"]

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Second convolutional block
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Third convolutional block
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._initialize_size()
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(self._to_linear, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        
    def _initialize_size(self):
        # Create a sample input to calculate the size after convolutions
        x = torch.randn(1, 3, 128, 128)  # Changed from 224x224 to 128x128
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        self._to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        print(f"Flattened feature size: {self._to_linear}")
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

def load_model(model_name):
    """Load a specific model by name"""
    model_path = os.path.join("models", model_name)
    print(f"Loading model from: {model_path}")
    
    if model_name not in get_available_models():
        raise ValueError("Invalid model name. Must be either 'best_cnn_model_luis.pth' or 'best_resnet_model.pth'")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if model_name == "best_cnn_model_luis.pth":
        # Use the custom CNN architecture
        model = SimpleCNN(num_classes=len(CLASS_NAMES))
    else:  # ResNet18 model
        # Load ResNet model with modified fc layer to match saved architecture
        model = models.resnet18(pretrained=False)
        # Replace the fc layer with a sequential module to match the saved state dict
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, len(CLASS_NAMES))
        )
    
    # Load the saved state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} successfully")
    return model, 'torch'

def predict_label(model, image_path, model_type, confidence_threshold=0.3):
    """
    Predict the label for a given image using the specified model.
    
    Args:
        model: The loaded model (TensorFlow or PyTorch)
        image_path: Path to the image file
        model_type: Type of model ('tf' or 'torch')
        confidence_threshold: Minimum confidence threshold for accepting predictions
        
    Returns:
        tuple: (predicted_class_name, confidence) or (None, confidence) if below threshold
    """
    try:
        if model_type == 'tf':
            # Preprocess image for TensorFlow
            processed_image = preprocess_image_tf(image_path)
            
            # Get predictions
            predictions = model.predict(processed_image, verbose=0)
            
            # Apply softmax to get probabilities
            probabilities = tf.nn.softmax(predictions).numpy()
            
            # Get top 3 predictions for debugging
            top_3_idx = np.argsort(probabilities[0])[-3:][::-1]
            print("\nTop 3 Predictions (TensorFlow):")
            for idx in top_3_idx:
                print(f"{CLASS_NAMES[idx]}: {probabilities[0][idx]*100:.2f}%")
            
            # Get the predicted class index and confidence
            predicted_class_idx = np.argmax(probabilities[0])
            confidence = float(probabilities[0][predicted_class_idx])
            
        else:  # PyTorch model
            # Preprocess image for PyTorch
            device = next(model.parameters()).device
            processed_image = preprocess_image_torch(image_path).to(device)
            
            # Get predictions
            with torch.no_grad():
                predictions = model(processed_image)
                probabilities = torch.nn.functional.softmax(predictions, dim=1)
                
                # Get top 3 predictions for debugging
                top_3_prob, top_3_idx = torch.topk(probabilities[0], 3)
                print("\nTop 3 Predictions (PyTorch):")
                for prob, idx in zip(top_3_prob, top_3_idx):
                    print(f"{CLASS_NAMES[idx.item()]}: {prob.item()*100:.2f}%")
                
                # Get the predicted class index and confidence
                predicted_class_idx = torch.argmax(probabilities[0]).item()
                confidence = float(probabilities[0][predicted_class_idx])
        
        # Debug information
        print("\nFinal Prediction:")
        print(f"Predicted class: {CLASS_NAMES[predicted_class_idx]}")
        print(f"Confidence: {confidence*100:.2f}%")
        
        # Return None if confidence is below threshold
        if confidence < confidence_threshold:
            print(f"Confidence below threshold ({confidence_threshold*100:.0f}%)")
            return None, confidence
        
        return CLASS_NAMES[predicted_class_idx], confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
