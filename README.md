# PRODIGY_ML_03
Dogs vs Cats Image Classification (SVM + HOG)  This project builds a machine learning model to classify cat and dog images using HOG feature extraction and an SVM classifier. Includes preprocessing, feature engineering, model training, evaluation, and a custom prediction function for testing new images.


🐶🐱 Dogs vs Cats Image Classification

A Machine Learning project using HOG feature extraction + SVM classifier

---
 📌 Project Overview

This project focuses on building a **binary image classification model** that distinguishes between **cats and dogs** using the popular Kaggle dataset.
Instead of deep learning, this model uses a traditional ML pipeline with:

* **Image preprocessing**
* **HOG (Histogram of Oriented Gradients) feature extraction**
* **SVM (Support Vector Machine) classification**

This makes it lightweight, interpretable, and easy to run even without a high-end GPU.

---

🚀 Features

✔ Preprocessing: resize → grayscale → flatten
✔ Feature extraction using **HOG**
✔ Model training using **SVM (linear kernel)**
✔ Validation accuracy and classification report
✔ Custom prediction function for testing any image
✔ Google Colab–friendly code

---

📂 Dataset

Dataset used: **Dogs vs Cats — Kaggle Competition**

* Training images: 25,000+
* Two classes: `cat`, `dog`

After downloading, the dataset was uploaded to **Google Drive** and connected to Colab.

---

🧠 Model Workflow

1. Load dataset paths
2. Convert images to grayscale
3. Resize to fixed dimension (64×64)
4. Flatten image → numeric array
5. Extract HOG features
6. Train SVM classifier
7. Evaluate on test set

---

 📊 Results

| Metric                  | Score                        |
| ----------------------- | ---------------------------- |
| **Accuracy**            | ~52%                         |
| **Precision/Recall/F1** | Balanced across both classes |

> Note: This is a baseline traditional ML model. Accuracy can be significantly improved using CNNs like VGG16 or MobileNet.

---

## 🖼 Prediction on New Images

A custom `predict_image()` function allows you to upload any image to Colab and run:

```python
print(predict_image("your_image.jpg"))
```

The displayed output includes both the prediction and the visual image.

---
 🛠 Technologies Used

* **Python**
* **OpenCV** (image processing)
* **scikit-learn** (SVM, scaling)
* **NumPy**
* **Matplotlib**
* **Google Colab**

---

 📚 How to Run

1. Clone the repository
2. Open the notebook in Google Colab
3. Upload dataset or mount Google Drive
4. Run all cells in sequence
5. Upload any image and test the model

---

 💡 Future Improvements

* Switch to **deep learning (VGG16 / CNN)** for higher accuracy
* Add GUI / web app (Streamlit or Flask)
* Perform hyperparameter tuning
* Use augmentation to improve generalization

---

📬 Contact

For feedback or collaboration:
Himanshu vishwakarma
9167340705


