# Enhancing-South-Asian-Food-Recognition-Using-Deep-Learning
Built a deep CNN with SE blocks, attention mechanisms, and EfficientNet transfer learning for dish classification. Optimized the model with ONNX (graph optimizations + quantization) for faster inference and deployed via Flask API + Docker. Developed a Streamlit UI for real-time predictions, enabling a complete end-to-end ML solution.
### Project Overview

This project focuses on building an **end-to-end deep learning system for dish classification**, combining cutting-edge computer vision architectures with optimized deployment pipelines. The solution leverages CNNs enhanced with **Squeeze-and-Excitation (SE) blocks**, **attention mechanisms**, and **EfficientNet transfer learning**, delivering high accuracy while ensuring scalability and real-time inference capabilities.

---

### Data Collection & Preprocessing

* Collected and curated a **large-scale food image dataset**, containing multiple categories of dishes.
* Performed **data cleaning and normalization**, ensuring consistent input dimensions and balanced class distributions.
* Applied extensive **data augmentation techniques** (random rotations, flips, crops, brightness/contrast adjustments, color jittering) to improve model generalization and reduce overfitting.
* Split data into **training, validation, and test sets** to monitor performance across all stages.

---

### Model Development

* Designed a **deep Convolutional Neural Network (CNN)** architecture with **Squeeze-and-Excitation (SE) blocks** to adaptively recalibrate channel-wise feature responses.
* Incorporated **attention mechanisms** to enhance feature extraction by allowing the model to focus on the most informative regions of dish images.
* Leveraged **EfficientNet (B3/B4 variants) with transfer learning**, fine-tuning pre-trained ImageNet weights to achieve state-of-the-art performance with fewer parameters and reduced training cost.
* Implemented **regularization techniques** such as dropout, label smoothing, and weight decay, ensuring robust performance and preventing overfitting.
* Used **learning rate schedulers (Cosine Annealing / Step Decay)** and optimizers like AdamW/SGD with momentum for stable convergence.

---

### Training & Monitoring

* Trained models using **GPU acceleration**, significantly reducing training time.
* Integrated **TensorBoard** for real-time monitoring of loss curves, accuracy metrics, and confusion matrices.
* Applied **early stopping** based on validation performance to prevent overfitting and optimize training duration.
* Achieved strong performance with **high top-1 and top-5 accuracy**, outperforming baseline CNN models.

---

### Model Optimization

* Converted the trained PyTorch model into **ONNX format** for cross-platform compatibility.
* Applied **ONNX Graph Optimizations** (constant folding, operator fusion, elimination of redundant nodes).
* Experimented with **dynamic quantization** and **FP16 precision** to reduce model size and speed up inference without significant accuracy loss.
* Benchmarked **GPU vs. CPU inference times**, ensuring performance portability across cloud and edge devices.

---

### Deployment

* Exposed the model through a **Flask REST API**, enabling inference requests via JSON/image uploads.
* Containerized the application using **Docker**, ensuring portability and reproducibility across environments.
* Conducted extensive **API testing with Postman** to validate endpoint behavior, response times, and prediction accuracy.
* Built a user-friendly **Streamlit web UI** that allows users to upload dish images and receive real-time classification results with confidence scores.
* Implemented **batch inference and caching mechanisms** to handle multiple concurrent requests efficiently.

---

### Results & Impact

* Delivered a **production-ready end-to-end ML pipeline**, covering all stages from data preprocessing and training to optimization and deployment.
* Achieved **real-time inference speeds** with optimized ONNX deployment, making the system suitable for cloud, web, and edge applications.
* Created a **scalable and modular architecture**, enabling easy adaptation to other domains such as plant disease detection, medical imaging, or retail product classification.

---

### Tech Stack

* **Deep Learning Frameworks**: PyTorch, TensorFlow (for ONNX validation)
* **Optimization**: ONNX Runtime, Quantization, FP16 Precision
* **Deployment**: Flask API, Docker, Streamlit, Postman
* **Data Processing**: OpenCV, NumPy, Pandas, Albumentations
* **Monitoring**: TensorBoard, Matplotlib, Seaborn

---

### Key Highlights

* Designed and trained a **state-of-the-art dish classification model** with SE blocks + attention + EfficientNet transfer learning.
* Achieved **high classification accuracy** through extensive augmentation, fine-tuning, and optimization.
* Successfully deployed the model as an **end-to-end application**, accessible via API and an interactive UI.
* Ensured scalability and real-world usability with **ONNX optimization, Docker deployment, and inference benchmarking**.

---

This project demonstrates expertise in **deep learning, model optimization, MLOps, and full-stack deployment**, making it directly applicable to real-world AI systems in domains like healthcare, retail, and food tech.

