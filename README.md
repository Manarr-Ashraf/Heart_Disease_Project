# Heart_Disease_Project

## 📌 Project Overview
This project focuses on analyzing, predicting, and visualizing heart disease risks using the **Heart Disease UCI dataset**.  
The pipeline covers **data preprocessing, feature selection, dimensionality reduction (PCA), supervised and unsupervised learning, model optimization, and deployment-ready model export**.  

## 🎯 Objectives
- Perform **data preprocessing & cleaning** (missing values, encoding, scaling).
- Apply **dimensionality reduction (PCA)** to retain essential features.
- Conduct **feature selection** using statistical methods and ML-based techniques.
- Train supervised learning models:
  - Logistic Regression  
  - Decision Trees  
  - Random Forest  
  - Support Vector Machine (SVM)  
- Apply unsupervised learning:
  - K-Means Clustering  
  - Hierarchical Clustering  
- Optimize models with **GridSearchCV** and **RandomizedSearchCV**.
- Export final trained models in **.pkl format** for reproducibility.  

## 🛠 Tools & Libraries
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Dimensionality Reduction:** PCA  
- **Feature Selection:** RFE, Chi-Square Test, Feature Importance  
- **Supervised Models:** Logistic Regression, Decision Tree, Random Forest, SVM  
- **Unsupervised Models:** K-Means, Hierarchical Clustering  
- **Model Optimization:** GridSearchCV, RandomizedSearchCV  

## 📂 Workflow
### 1. Data Preprocessing & Cleaning
- Load dataset into Pandas DataFrame  
- Handle missing values  
- Apply one-hot encoding to categorical variables  
- Standardize numerical features  
- Perform EDA (histograms, correlation heatmaps, boxplots)  

### 2. Dimensionality Reduction (PCA)
- Apply PCA to reduce dimensions  
- Select optimal number of components using explained variance  
- Visualize cumulative variance & PCA scatter plots  

### 3. Feature Selection
- Rank features using Random Forest importance  
- Apply Recursive Feature Elimination (RFE)  
- Perform Chi-Square test  
- Select best predictors for modeling  

### 4. Supervised Learning - Classification
- Train/Test split (80/20)  
- Train models: Logistic Regression, Decision Tree, Random Forest, SVM  
- Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC  

### 5. Unsupervised Learning - Clustering
- Apply **K-Means** (choose K via elbow method)  
- Perform **Hierarchical Clustering** with dendrogram analysis  
- Compare clusters with actual disease labels  

### 6. Hyperparameter Tuning
- Apply **GridSearchCV** and **RandomizedSearchCV**  
- Select best performing model with tuned hyperparameters  

### 7. Model Export
- Save trained models with **joblib/pickle** (`.pkl` format)  
- Ensure reproducibility by saving preprocessing + model pipeline  

## 🚀 Deliverables
- Cleaned dataset ready for modeling  
- PCA-transformed dataset with visualizations  
- Feature importance ranking & reduced feature set  
- Trained supervised & unsupervised models with metrics  
- Optimized model with best hyperparameters  
- Exported `.pkl` model pipeline  

## 📑 Project Structure
