# 🧠 Animal Image Classifier with CNN (Ironhack AI Engineer Bootcamp - Group 1 - Project 2)

This project is part of the Ironhack AI/ML Bootcamp and demonstrates a deep learning application using Convolutional Neural Networks (CNNs) to classify images of animals into predefined categories.

## 🗂 Project Overview

In this project, we:

- Built and trained a CNN model from scratch using PyTorch
- Preprocessed a custom dataset of ~28,000 animal images across 10 categories
- Evaluated model performance with standard classification metrics
- Performed Transfer Learning using pretrained models (e.g., VGG16 or Inception)
- Deployed the final model using Flask for image classification via a web interface

## 📁 Dataset

We used the **Animals-10 dataset** from Kaggle:  
👉 [Download here](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data)

**Categories:**  
`dog, cat, horse, spider, butterfly, chicken, sheep, cow, squirrel, elephant`

## 🧪 Model Development

### 🔧 Preprocessing

- Image resizing, normalization
- Data augmentation (e.g., flipping, rotation)
- Train-validation split

### 🏗 CNN Architecture

The base CNN includes:

- Convolutional and pooling layers
- Batch normalization and dropout
- Fully connected layers with softmax activation

We also experimented with **transfer learning**, using models like **VGG16**, fine-tuned for our dataset.

### 🧠 Training

- Optimizer: Adam
- Loss: CrossEntropyLoss
- Techniques: Early stopping, learning rate scheduler

### 📊 Evaluation

- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Best model achieved XX% accuracy on the validation set

## 🌐 Deployment

We deployed the best-performing model using Flask:

- Users can upload one or more images
- The app returns predictions and probabilities per class

_(Optional: Hosted on Google Cloud Platform / AWS / Render / etc.)_

## 📄 Project Structure

project2-animal-classifier/
├── data/              # Sample images for testing
├── models/            # Saved models (CNN, VGG16, etc.)
├── app/               # Flask app files
│   └── app.py
├── notebooks/         # Development notebooks
├── utils/             # Data loading, preprocessing, model utils
├── requirements.txt
├── report.pdf         # Final project report
└── README.md


## 📚 Key Learnings

- Gained hands-on experience with CNNs using PyTorch
- Applied real-world ML practices: data prep, model tuning, evaluation
- Learned how to integrate AI models into web apps
- Practiced critical thinking and experimentation with ML pipelines

## 🏁 Final Thoughts

This project marks the completion of our second major milestone in the Ironhack AI/ML bootcamp. It demonstrates our ability to build, evaluate, and deploy deep learning models in a production-like scenario.

---

**Made with ❤️ by Group 1 Proejct 2 @ Ironhack 2025**

