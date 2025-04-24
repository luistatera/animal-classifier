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
class_names = ['cat', 'dog', 'horse', 'spider', 'butterfly', 'chicken', 'sheep', 'cow', 'squirrel', 'elephant']

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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_name):
    """Load a specific model by name"""
    model_path = os.path.join("models", model_name)
    if model_name.endswith('.keras'):
        return tf.keras.models.load_model(model_path)
    else:
        # Load PyTorch model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine model architecture based on filename
        if 'vgg16' in model_name.lower():
            # Load VGG16 with ImageNet weights
            model = models.vgg16(weights='DEFAULT')
            # Freeze all convolutional layers
            for param in model.features.parameters():
                param.requires_grad = False
            # Replace the classifier head
            model.classifier[6] = nn.Linear(4096, len(class_names))
        elif 'resnet' in model_name.lower():
            # Load ResNet18 with ImageNet weights
            model = models.resnet18(weights='DEFAULT')
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze the last two layers
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
            # Replace the final layer
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        else:
            model = SimpleCNN(num_classes=len(class_names))
        
        # Load the saved state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model

def predict_label(img_path, model_name="best_cnn_model.keras"):
    """Predict the label for an image using the specified model"""
    model = load_model(model_name)
    
    if isinstance(model, tf.keras.Model):
        # TensorFlow prediction
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)[0]
    else:
        # PyTorch prediction
        device = next(model.parameters()).device
        
        # Use the appropriate transforms based on the model
        if 'vgg16' in model_name.lower() or 'resnet' in model_name.lower():
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
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
    
    top_idx = np.argmax(predictions)
    return class_names[top_idx], float(predictions[top_idx])
