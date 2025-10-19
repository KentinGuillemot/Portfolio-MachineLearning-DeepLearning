# 🛡️ Fraud Detection System

Welcome to the **Fraud Detection System** project! This project aims to develop a robust machine learning and deep learning pipeline to detect fraudulent transactions using a comprehensive dataset of financial transactions. The goal is to identify suspicious activities and reduce fraud in digital payments through advanced data analysis, ensemble learning, and hyperparameter optimization techniques.

## 📋 Project Overview

In today's digital world, fraud detection is a critical challenge for financial institutions. This project leverages a multi-phase approach to improve fraud detection accuracy and interpretability. The approach includes baseline models, ensemble learning, and deep learning techniques, addressing challenges such as **imbalanced data** and **overfitting** through careful preprocessing and model optimization.

## 🛠️ Technologies Used

- **Programming Languages**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Optuna, Matplotlib, Seaborn, Sweetviz
- **Machine Learning Models**: Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, XGBoost  
- **Data Handling**: SMOTE for imbalanced data, Z-score for outlier removal, Standardization for feature scaling  
- **Optimization**: Optuna for hyperparameter tuning  

## 📊 Dataset

The dataset contains **1 million transactions** with the following features:

- **distance_from_home**: Distance between the user's home and the transaction location  
- **distance_from_last_transaction**: Distance from the previous transaction  
- **ratio_to_median_purchase_price**: Ratio of the transaction amount to the median purchase price  
- **repeat_retailer**: Indicates if the transaction was made with a repeat retailer (1 = yes, 0 = no)  
- **used_chip**: Indicates if the transaction used a chip card (1 = yes, 0 = no)  
- **used_pin_number**: Indicates if a PIN was used (1 = yes, 0 = no)  
- **online_order**: Indicates if the transaction was made online (1 = yes, 0 = no)  
- **fraud**: Target variable indicating if the transaction is fraudulent (1 = yes, 0 = no)  

⚠️ The dataset is **imbalanced**, with only **9%** of transactions being fraudulent.

## 🧪 Methodology

### 🔍 Data Exploration and Preprocessing

1. **Neural Network Model**:  
   - **Standardization**: Standardized features to have a mean of 0 and a standard deviation of 1.  
   - **No SMOTE**: SMOTE was not used to avoid introducing noise and redundancy.  
   - **No Outlier Removal**: Outliers were retained to preserve potentially valuable information.  

2. **Base Models for Stacking**:  
   - **Outlier Removal**: Applied Z-score to remove extreme values.  
   - **SMOTE**: Used to balance the dataset and improve model performance.  
   - **Standardization**: Scaled features for consistent model training.  

### 🧠 Model Implementation

1. **Baseline Models**:  
   - **Logistic Regression**  
   - **K-Nearest Neighbors (KNN)**  

2. **Ensemble Learning (Stage 2)**:  
   - **Random Forest** and **XGBoost** optimized with **Optuna**.  
   - **Stacking**: Combined Random Forest and XGBoost with a logistic regression meta-model.  

3. **Deep Learning (Stage 3)**:  
   - **Deep Neural Network** optimized with **Optuna**.  
   - **Stacking** with a deep learning meta-learner combining Logistic Regression, KNN, Random Forest, and XGBoost.  

### 🛠️ Overfitting Mitigation

- **Dropout Layers**: Applied dropout for regularization.  
- **Batch Normalization**: Stabilized learning by normalizing activations.  
- **Early Stopping**: Halted training when the validation loss stopped improving.

## 📈 Evaluation Metrics

- **Recall**: Focused on minimizing false negatives.  
- **F2-Score**: Balanced metric prioritizing recall.  
- **Confusion Matrix**: Visualized model performance.

## 🏆 Key Results

| **Model**                  | **F2-Score** | **Recall** | **False Positives** | **False Negatives** |
|-----------------------------|--------------|------------|---------------------|---------------------|
| **Random Forest**          | 0.99972      | 0.99977    | 122                 | 4                   |
| **XGBoost**                | 0.99918      | 0.99760    | 358                 | 42                  |
| **Stage 2 Stacking**       | 0.99980      | 0.99971    | 117                 | 5                   |
| **Stage 3 Stacking (DL)**  | 0.99975      | 0.99970    | 187                 | 11                  |
