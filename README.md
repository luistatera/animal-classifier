# 🧠 Animal Image Classifier (Ironhack AI/ML Bootcamp - Project 2)

This project showcases an image classification system using deep learning. We applied Convolutional Neural Networks (CNNs) with **Transfer Learning (ResNet18)** to classify animal images into 10 categories. The project includes model training, evaluation, and deployment via a Flask web app.

## 📁 Dataset

We used the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data), which contains ~28,000 images in 10 categories:

- `cat`, `dog`, `horse`, `elephant`, `butterfly`, `chicken`, `cow`, `sheep`, `squirrel`, `spider`

## 🧪 Model Development

### 🔧 Preprocessing

- Resized to 224x224
- Normalization to ImageNet mean/std
- Data augmentation: RandomHorizontalFlip, RandomRotation, RandomCrop
- Split: 80% training / 20% validation

### 🧠 Model Architecture

- **Transfer Learning with ResNet18**
  - Frozen early layers
  - Custom fully connected classifier for our 10 animal classes

### ⚙️ Training

- Optimizer: `Adam`
- Loss function: `CrossEntropyLoss`
- Techniques: `EarlyStopping`, `lr_scheduler`
- Device: GPU/CPU fallback using PyTorch

### 📊 Evaluation

- Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Best accuracy: **≈91%** on validation set

## 🌐 Deployment

We deployed the model using a Flask web app:

- Users can upload images
- The app returns:
  - Predicted class
  - Top 3 class probabilities
- Simple and responsive UI using HTML templates


## 📂 Project Structure

project2-animal-classifier/
├── data/              # Sample images for testing
├── models/            # Saved models (CNN, VGG16, etc.) -- ignored because of the size
├── app/               # Flask app files
│   └── app.py
├── notebooks/         # Development notebooks
├── utils/             # Data loading, preprocessing, model utils
├── requirements.txt
├── report.pdf         # Final project report
└── README.md


## 🚀 Setup Instructions

1. Clone the repo  
2. Install dependencies  
```bash
pip install -r requirements.txt


## 📚 Key Learnings

- Gained hands-on experience with CNNs using PyTorch
- Applied real-world ML practices: data prep, model tuning, evaluation
- Learned how to integrate AI models into web apps
- Practiced critical thinking and experimentation with ML pipelines

## 🏁 Final Thoughts

This project marks the completion of our second major milestone in the Ironhack AI/ML bootcamp. It demonstrates our ability to build, evaluate, and deploy deep learning models in a production-like scenario.

![image](https://github.com/user-attachments/assets/8b6c2b3f-7ce2-4165-ab34-da0238aafb43)


---

**Made with ❤️ by Group 1 Proejct 2 @ Ironhack 2025**

