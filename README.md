# Morphological-Classification-Of-Extra-Galactic-Radio-Sources-Using-Gradient-Boosting-Methods
Project based work - analysis.


# Morphological Classification of Extragalactic Radio Sources

**Author:** Zaighum Jawad Aslam  
**Reg. No:** D03000118  
**Project Type:** Data Mining & Machine Learning  
**Score:** 28/30  

---

## 🧠 Overview

This project focuses on the **automated classification of extragalactic radio galaxies** into three Fanaroff–Riley classes — **FR0**, **FRI**, and **FRII** — using deep learning (CNN) and machine learning (XGBoost) techniques. With data augmentation, feature extraction (ResNet50), dimensionality reduction (PCA), and interpretability tools (SHAP), this project demonstrates an end-to-end pipeline for morphological classification using astronomical image data.

---

## 📂 Dataset

- **Source:** Best-Heckman sample
- **Size:** 13,140 grayscale images (300×300)
- **Classes:**
  - **FR0:** Compact, no prominent jets
  - **FRI:** Low-brightness jets, asymmetric
  - **FRII:** Bright, well-separated radio lobes
- **Preprocessing:**
  - Cropped center 60×60 patches
  - Normalized to [0, 1]
  - Filtered low-content/blank images

---

## ⚙️ Project Structure

- `data/`: Raw and augmented image dataset
- `models/`: Saved CNN model (`.keras` format)
- `notebooks/`: Jupyter notebook with full code
- `visuals/`: All plots and diagrams (EDA, PCA, SHAP, CNN layers, etc.)
- `README.md`: Project documentation
- `requirements.txt`: Required libraries

---

## 🔍 Key Steps & Techniques

### 📊 1. Exploratory Data Analysis (EDA)
- Class distribution visualized before & after augmentation
- Mean & median intensity plots
- Edge detection (Sobel)
- Symmetry analysis (SSIM)
- Correlation heatmaps (FR0, FRI, FRII)

### 🔧 2. Data Augmentation
- Used `ImageDataGenerator` (Keras)
- Applied on FRI and FRII to fix class imbalance
- Techniques: rotation, flip, zoom, shear, translation

### 🧪 3. Feature Engineering
- **ResNet50** (pretrained on ImageNet)
  - Extracted 2048-dimensional feature vectors
  - Custom transform pipeline (resize, grayscale → RGB, normalize)
- **PCA**
  - Reduced to 50 components for model input
  - 2D scatter plot shows partial separability of classes

### 📉 4. Pixel-Level Analysis
- Flattened grayscale images
- Plotted pixel-wise correlation matrices
- Used for verifying structural consistency

---

## 🤖 Modeling & Training

### 🧠 CNN Model (Keras/TensorFlow)
- Input shape: (60, 60, 1)
- Layers: 7 Conv layers → BatchNorm → ReLU → MaxPooling
- Output: Softmax layer with 3 classes
- Optimizer: Adam (lr=9e-6)
- Class Weights: FR0=1.0, FRI=2.5, FRII=1.5
- EarlyStopping callback (patience=15)

### 📈 Training Results
- **Train Accuracy:** 88.3%
- **Test Accuracy:** 81.0%
- Loss curves and accuracy plots included

### ✅ SHAP (Explainability)
- DeepExplainer used on CNN predictions
- Visual heatmaps showed pixel-level contributions
- Confirmed model focuses on jet/lobe regions

---

## 📦 Traditional ML: XGBoost Classifier
- Input: Flattened 60x60 grayscale (3600 features)
- Test Accuracy: **81.3%**
- Strong for FR0, more confusion between FRI/FRII
- Confusion matrix and classification report included

---

## 🔍 Real-time Image Prediction
- Deployed a function to:
  - Load image from user’s path
  - Resize → Normalize → CNN model inference
  - Output class label + probability

---

## 📌 Key Libraries Used

| Library     | Purpose                            |
|-------------|-------------------------------------|
| TensorFlow  | CNN, training, data augmentation    |
| PyTorch     | ResNet50 feature extraction         |
| scikit-learn| PCA, metrics, train-test split      |
| XGBoost     | Traditional ML classification       |
| SHAP        | Model explainability (CNN)          |
| Matplotlib  | Plotting and visualizations         |
| OpenCV      | Edge detection, symmetry (SSIM)     |
| NumPy/Pandas| Data handling and manipulation      |

---

## 🧠 Learnings & Takeaways

- Deep learning outperformed traditional models in multi-class classification.
- Class balancing (augmentation + class weights) significantly improved fairness.
- SHAP helped build model trust and interpretability.
- PCA helped in visualizing class separation and reducing model input size.

---

## 📌 Future Work

- Deploy CNN model as a web app for real-time predictions.
- Explore transformer-based vision models.
- Train on larger datasets (e.g., from SKA).
- Perform anomaly detection on radio galaxy structures.

---

## 📞 Contact

For any questions or collaboration:

**Zaighum Jawad Aslam**  
📧 Email: *[add your email]*  
📁 LinkedIn / GitHub: *[add your links]*

---

## ⭐️ Acknowledgements

- Special thanks to [Professor’s Name] for guidance and evaluation.
- Dataset sourced from Best-Heckman catalog.
