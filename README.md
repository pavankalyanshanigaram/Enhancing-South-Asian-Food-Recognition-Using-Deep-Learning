# Enhancing-South-Asian-Food-Recognition-Using-Deep-Learning
Built a deep CNN with SE blocks, attention mechanisms, and EfficientNet transfer learning for dish classification. Optimized the model with ONNX (graph optimizations + quantization) for faster inference and deployed via Flask API + Docker. Developed a Streamlit UI for real-time predictions, enabling a complete end-to-end ML solution.
# 🍽️ Dish Classification using Deep Learning

## 📌 Overview

This project implements an **end-to-end deep learning pipeline for dish classification**, leveraging **EfficientNet with transfer learning**, **Squeeze-and-Excitation (SE) blocks**, and **attention mechanisms** to achieve high accuracy. The model is optimized for real-time inference and deployed as a **Flask REST API** containerized with **Docker**, along with a **Streamlit web UI** for user interaction.

---

## 📊 Dataset

* Dataset Source: **[Kaggle - Dish/Food Classification Dataset](https://www.kaggle.com/)**
* Preprocessing: Applied **image resizing, normalization, and augmentation** (rotation, flipping, cropping, color jittering) to improve model generalization.
* Dataset split into **train / validation / test** sets.

---

## 🧠 Model Architecture

* **Base Model**: EfficientNet (transfer learning from ImageNet).
* **Enhancements**:

  * Squeeze-and-Excitation (SE) blocks for adaptive channel recalibration.
  * Attention mechanisms for improved feature extraction.
* **Training**:

  * Data augmentation using Albumentations.
  * Regularization via dropout, weight decay, label smoothing.
  * Optimized with AdamW + learning rate scheduling.
* **Optimization**: Converted to **ONNX format**, with graph optimization + quantization for faster inference.

---

## 🚀 Deployment

* Exposed trained model through a **Flask REST API**.
* Containerized using **Docker** for portability.
* Tested API endpoints with **Postman**.
* Built a **Streamlit UI** for real-time predictions (upload an image → get dish prediction + confidence score).

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/dish-classification.git
cd dish-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the Model

```bash
python train.py
```

### 2. Run Flask API

```bash
python app.py
```

### 3. Run Streamlit UI

```bash
streamlit run ui.py
```

### 4. Docker Deployment

```bash
docker build -t dish-classification .
docker run -p 5000:5000 dish-classification
```

---

## 📈 Results

* Achieved **high accuracy** with top-1 and top-5 metrics outperforming baseline CNN models.
* Inference optimized with ONNX Runtime → **~40% faster on CPU** and **~60% faster on GPU**.

---

## 📦 Tech Stack

* **Frameworks**: PyTorch, ONNX Runtime
* **API & Deployment**: Flask, Docker, Streamlit, Postman
* **Data Processing**: OpenCV, Albumentations, NumPy, Pandas
* **Monitoring**: TensorBoard, Matplotlib, Seaborn

---

## 👨‍💻 Author

**Pavan Kalyan Shanigarm**

---

🚀 This project demonstrates expertise in **computer vision, model optimization, and MLOps**, and can be adapted to other domains like medical imaging, retail product classification, and food recognition systems.


