# 🧠 Animal Image Classifier  
*Ironhack AI/ML Bootcamp – Project 2*

This project showcases an image classification system using deep learning. We applied **Convolutional Neural Networks (CNNs)** with **Transfer Learning (ResNet18)** to classify animal images into 10 categories. It covers model development, evaluation, and deployment via a Flask web app.

---

## 📁 Dataset

We used the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data), which contains ~28,000 images across 10 categories:

`cat`, `dog`, `horse`, `elephant`, `butterfly`, `chicken`, `cow`, `sheep`, `squirrel`, `spider`

---

## 🧪 Model Development

### 🔧 Preprocessing
- Resized images to 224×224
- Normalized using ImageNet mean and std
- Applied data augmentation:
  - `RandomHorizontalFlip`, `RandomRotation`, `RandomCrop`
- Dataset split: 80% training / 20% validation

### 🧠 Model Architecture
- **ResNet18 (Transfer Learning)**
  - Early layers frozen
  - Custom classifier added for 10 animal classes

### ⚙️ Training
- Optimizer: `Adam`
- Loss function: `CrossEntropyLoss`
- Techniques: `EarlyStopping`, `lr_scheduler`
- Runs on GPU (if available), otherwise CPU

### 📊 Evaluation
- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion Matrix for class-wise analysis
- Best accuracy: **≈91%** on the validation set

---

## 🌐 Deployment

We deployed the model using a **Flask web app**, where users can:
- Upload an image
- Receive:
  - Predicted class
  - Top 3 class probabilities

Includes a simple, responsive UI built with HTML templates.

---

## 📂 Project Structure

```
project2-animal-classifier/
├── app/                # Flask app files
│   └── app.py
├── data/               # Sample test images
├── models/             # Trained models (ignored in repo due to size)
├── notebooks/          # Development notebooks
├── utils/              # Preprocessing and model helper functions
├── requirements.txt    # Dependency list
├── report.pdf          # Final project report
└── README.md
```

---

## 🚀 Setup Instructions

1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app/app.py
   ```

---

## 📚 Key Learnings

- Hands-on experience with CNNs in PyTorch
- Applied real-world ML practices: data prep, tuning, and evaluation
- Learned how to deploy ML models via web applications
- Gained practical insights into experimentation and critical thinking in ML

---

## UI



## 🏁 Final Thoughts

This project represents the completion of our second major milestone in the Ironhack AI/ML Bootcamp. It demonstrates our capability to build, evaluate, and deploy deep learning models in a production-like environment.

![model diagram](https://github.com/user-attachments/assets/8b6c2b3f-7ce2-4165-ab34-da0238aafb43)

---

**Made with ❤️ by Group 1 – Project 2 @ Ironhack 2025**

