import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from io import BytesIO

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
        self.dropout = torch.nn.Dropout(p=0.5)
        
    def _initialize_size(self):
        # Create a sample input to calculate the size after convolutions
        x = torch.randn(1, 3, 128, 128)  # Input size is 128x128
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
        x = self.dropout(x)
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

def predict_label(model, image_data, model_type):
    """
    Predict the label for an image using the specified model
    image_data can be either a file path or a BytesIO object
    """
    try:
        # Load and preprocess the image
        if isinstance(image_data, str):
            # If image_data is a file path
            image = Image.open(image_data)
        else:
            # If image_data is a BytesIO object
            image = Image.open(image_data)
        
        print(f"Image opened successfully. Mode: {image.mode}, Size: {image.size}")
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("Converted image to RGB mode")
        
        # Define the transformation pipeline based on model type
        if isinstance(model, SimpleCNN):
            # SimpleCNN expects 128x128 input
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("Using SimpleCNN preprocessing (128x128)")
        else:
            # ResNet expects 224x224 input
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("Using ResNet preprocessing (224x224)")
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)
        print(f"Transformed image tensor shape: {image_tensor.shape}")
        
        # Move to the same device as the model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        print(f"Using device: {device}")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Print all class probabilities for debugging
            probs = probabilities[0].cpu().numpy()
            print("\nClass probabilities:")
            for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, probs)):
                print(f"{class_name}: {prob*100:.2f}%")
            
        # Get the predicted class and confidence
        confidence = confidence.item()
        predicted_class = predicted.item()
        
        print(f"\nPredicted class: {CLASS_NAMES[predicted_class]}")
        print(f"Confidence: {confidence*100:.2f}%")
        
        # Lower the threshold to 0.3
        if confidence > 0.3:  # Adjusted threshold
            return CLASS_NAMES[predicted_class], confidence
        else:
            return None, confidence
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, 0.0

def get_class_name(idx):
    """Get class name from index"""
    return CLASS_NAMES[idx]
