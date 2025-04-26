# 🧠 Animal Image Classifier  
*Ironhack AI/ML Bootcamp – Project 2*

![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project showcases an image classification system using deep learning. We applied **Convolutional Neural Networks (CNNs)** with **Transfer Learning (ResNet18)** to classify animal images into 10 categories. It covers model development and evaluation.

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

## 📂 Project Structure

```
animal-classifier/
├── app.py              # Main Flask application
├── model_loader.py     # Model loading and prediction functions
├── requirements.txt    # Project dependencies
├── .gitignore         # Git ignore rules
├── .gcloudignore      # Google Cloud ignore rules
├── docs/              # Project documentation
├── img2test/          # Directory for test images
├── models/            # Trained model files
│   ├── best_cnn_model_luis.pth
│   └── best_resnet_model.pth
├── notebooks/         # Jupyter notebooks for development
├── static/            # Static files (CSS, JS, images)
│   ├── uploads/       # Directory for uploaded images
│   ├── style.css
│   └── upload.js
├── templates/         # HTML templates
│   ├── index.html
│   └── project.html
├── utils/             # Utility functions
└── temp/              # Temporary files
```

---

## 🚀 Setup Instructions

1. Clone the repository  
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask app:
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

---

## 📚 Key Learnings

- Hands-on experience with CNNs and ResNet
- Applied real-world ML practices: data prep, tuning, and evaluation
- Gained practical insights into experimentation and critical thinking in ML

---

## UI



## 🏁 Final Thoughts

This project represents the completion of our second major milestone in the Ironhack AI/ML Bootcamp. It demonstrates our capability to build and evaluate deep learning models in a production-like environment.

![model diagram](https://github.com/user-attachments/assets/8b6c2b3f-7ce2-4165-ab34-da0238aafb43)

---

**Made with ❤️ by Group 1 – Project 2 @ Ironhack 2025**

<footer class="footer">
    <div class="container">
        <p>© 2025 Group 1 - Project 2 | Ironhack Bootcamp</p>
    </div>
</footer>

