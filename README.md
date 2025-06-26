# Farm Fusion: Agricultural AI Platform

## Overview
Farm Fusion is an end-to-end AI platform designed to provide intelligent insights for agriculture. This project integrates multiple machine learning models for crop recommendation, fertilizer recommendation, and plant disease detection into a unified API using FastAPI. While a basic web interface exists, the primary focus of this project is on the robustness and functionality of the machine learning models and the API that serves them.

**Please Note:** The API backend for the machine learning models is not actively hosted at the moment. Therefore, while the frontend website might load, the machine learning functionalities (predictions, recommendations) will not be operational.

## Features
This platform includes the following key functionalities:

### Plant Disease Detection (Multiple Crops):

**Potato Disease Detection:** Identifies Early Blight, Late Blight, and Healthy potato leaves using a Convolutional Neural Network (CNN).

**Bell Pepper Disease Detection:** Detects Bacterial Spot disease and Healthy bell pepper leaves using a CNN.

**Wheat Disease Detection:** Utilizes a VGG19-based model to identify various wheat diseases like Brown Rust, Loose Smut, Septoria, Yellow Rust, and Healthy plants.

**Technical Details:** Images are preprocessed (resized, rescaled, augmented) and passed through robust CNN architectures.

### Crop Recommendation:

Recommends the optimal crop to cultivate based on environmental factors.

**Input Features:** Nitrogen (N), Phosphorus (P), Potassium (K) content in soil, Temperature, Humidity, pH, and Rainfall.

**Model:** Sequential Neural Network for multi-class classification, achieving 99.31% accuracy.

### Fertilizer Recommendation:

Suggests the most suitable fertilizer type for a given crop and soil conditions.

**Input Features:** Soil type, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, and Crop type.

**Model:** XGBoost classifier, demonstrating 99.34% overall accuracy.

## API Endpoints (FastAPI)
The backend API is built with FastAPI, providing clear and concise endpoints for each model:

### `/v1/predict` (POST): Potato Disease Detection
- **Input:** Image file
- **Output:** Predicted class (e.g., 'Early blight', 'Late blight', 'Healthy'), Confidence percentage.

### `/v2/predict` (POST): Bell Pepper Disease Detection
- **Input:** Image file
- **Output:** Predicted class (e.g., 'Bacterial Spot', 'Healthy'), Confidence percentage.

### `/v3/predict` (POST): Fertilizer Recommendation
- **Input:** Form data (soil, nitrogen, phosphorus, potassium, pH, rainfall, temperature, crop)
- **Output:** Recommended fertilizer type.

### `/v4/predict` (POST): Crop Recommendation
- **Input:** Form data (nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall)
- **Output:** Predicted crop class, Confidence percentage.

### `/v5/predict` (POST): Wheat Disease Detection
- **Input:** Image file
- **Output:** Predicted wheat disease class (e.g., 'Brown Rust', 'Loose Smut'), Confidence percentage.

### Additional API Notes:
- **Image Preprocessing:** Images are processed using Pillow (PIL) and converted to NumPy arrays before model inference.
- **Feature Scaling:** Numerical features are standardized using pre-trained scalers for optimal model performance.
- **CORS:** The API is configured with CORS middleware to allow cross-origin requests.

## Model Architectures & Performance

### Plant Disease Detection (Potato, Bell Pepper):
Custom Convolutional Neural Networks (CNNs).
- **Potato:** ~97.52% accuracy
- **Bell Pepper:** ~100% accuracy

### Plant Disease Detection (Wheat):
VGG19-based model (for overcoming overfitting issues in initial CNNs).
- **Wheat:** ~98.57% accuracy

### Crop Recommendation:
Sequential Neural Network.
- **Accuracy:** 99.31%

### Fertilizer Recommendation:
XGBoost.
- **Overall Accuracy:** 99.34%

## Technologies Used
- **Backend & API:** Python, FastAPI
- **Machine Learning:** TensorFlow, Scikit-learn, XGBoost
- **Data Handling:** Pandas, NumPy
- **Image Processing:** OpenCV, Pillow (PIL)
- **Other:** Git
