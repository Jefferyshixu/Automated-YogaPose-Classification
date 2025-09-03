# Automated Yoga Pose Classification 🧘‍♀️

## 📌 Overview
This project focuses on **automatic yoga pose classification** using **computer vision** and **deep learning** techniques. Yoga pose recognition plays a crucial role in domains such as **fitness monitoring**, **physical therapy**, and **sports training**.  
The system aims to **accurately and rapidly identify yoga poses** from images, overcoming the challenges of manual classification which is time-consuming, error-prone, and highly dependent on subtle differences between poses.

## 🎯 Problem Statement
- Manual inspection of yoga poses is slow and error-prone.  
- Wide variety of poses and subtle differences make classification challenging.  
- Goal: Develop a **neural network–based automated recognition system** with:  
  - ✅ High accuracy  
  - ✅ Robustness to varying appearances and angles  
  - ✅ Real-time deployment capabilities  

## 📂 Dataset
We use the **Yoga Pose Dataset** (`shrutisaxena/yoga-pose-image-classification-dataset`) [^1], which contains **107 different yoga poses** categorized into folders.  
- Each folder corresponds to a yoga pose label.  
- Image sizes and resolutions vary across the dataset.  

## 🛠️ Methodology
The notebook implements the following steps:  

1. **Data Preprocessing**  
   - Image resizing and normalization  
   - Data augmentation (rotation, flipping, brightness adjustment)  

2. **Model Architecture**  
   - Convolutional Neural Networks (CNNs)  
   - Transfer learning with pre-trained models  
   - Dense layers with softmax activation for multi-class classification  

3. **Training & Validation**  
   - Train-validation-test split on Yoga Pose dataset  
   - Optimizer: Adam  
   - Loss function: Categorical Cross-Entropy  
   - Evaluation metrics: Accuracy, Precision, Recall, F1-score  

4. **Results**  
   - Achieved significant classification accuracy across many yoga poses  
   - Demonstrated robustness against pose variations  

5. **Deployment**  
   - Trained model can be exported (`.h5`)  
   - Ready for real-time yoga pose recognition applications  

## 🚀 How to Run  

### 1. Clone Repository  
```bash
git clone https://github.com/yourusername/Automated-YogaPose-Classification.git
cd Automated-YogaPose-Classification
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Download Dataset  
- Download from Kaggle: [`shrutisaxena/yoga-pose-image-classification-dataset`](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset)  
- Place it in the `dataset/` directory.  

### 4. Run Notebook  
```bash
jupyter notebook "Automated YogaPose Classification.ipynb"
```

### 5. Inference Example  
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("yoga_pose_model.h5")
img = cv2.imread("test_pose.jpg")
img = cv2.resize(img, (224,224)) / 255.0
img = np.expand_dims(img, axis=0)
prediction = np.argmax(model.predict(img))
print("Predicted pose class:", prediction)
```

## 📊 Results  

### Overall Performance  
- The model successfully classified most yoga poses with **average accuracy around 60–70%**.  
- Certain poses with clear distinctive features achieved **>80% accuracy** (*Dandasana, Padmasana, Halasana*).  
- Some complex poses with subtle variations showed **lower performance (<30%)** (*Kurmasana, Mayurasana*).  

### Example Per-Pose Accuracy (Extracted from Test Set)  
| Pose | Accuracy |
|------|----------|
| Dandasana | **100%** |
| Padmasana | 86% |
| Halasana | 82% |
| Astavakrasana | 83% |
| Mayurasana | 19% |
| Kurmasana | 11% |
| Eka Pada Koundinyanasana I | 0% |

### Training Curves  
- **Loss** decreased steadily across epochs.  
- **Accuracy** improved consistently, demonstrating effective convergence.  
- Validation curves indicate reasonable generalization, though with some variance on difficult poses.  

### Classification Challenges  
- Poses with **similar arm/leg configurations** (e.g., *Eka Pada Rajakapotasana variations*) were more difficult to distinguish.  
- Class imbalance (fewer samples for certain poses) likely affected performance.  

## 📖 References  
[^1]: Shruti Saxena, Yoga Pose Image Classification Dataset, Kaggle.  

---

⚡️ *This project demonstrates how deep learning can enable robust and real-time yoga pose recognition, with applications in health, fitness, and therapy.*  
