# 🌾 Crop Recommendation System

A machine learning web application that recommends the most suitable crop 
to grow based on soil nutrients and climate conditions.

**Built for Enduro'26 · Strategic ERP · July 2026**

---

## 🔗 Live Demo


---[Click here to open the app](https://crop-recommendation-system-hlhucikphddvct9q2ytwxt.streamlit.app/)

## 📌 Project Overview

Farmers often make crop decisions based on intuition, leading to poor yields 
and wasted resources. This system uses a trained Random Forest model to 
recommend the best crop based on 7 soil and climate inputs — instantly, 
from any browser.

---

## 📊 Dataset

- **Source:** Crop Recommendation Dataset (Kaggle)
- **Samples:** 2,200 rows · 22 crop classes · 100 samples per class
- **Features:** N, P, K, temperature, humidity, pH, rainfall
- **Target:** Crop label (22 classes)

---

## 🤖 Models Trained

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Decision Tree | 98.64% | 0.986 |
| **Random Forest** ✓ | **99.32%** | **0.993** |

Random Forest was selected for deployment due to lower overfitting risk 
and better recall on harder-to-distinguish crops (jute, lentil).

---

## 📈 Key EDA Findings

- **Humidity** showed the clearest bimodal distribution — the strongest predictor
- **Rainfall** is right-skewed, driven by high-rainfall crops like rice and coconut
- **P–K correlation = 0.74** — fruit crops need both nutrients in high quantities
- **pH** has the narrowest range — weakest standalone predictor (confirmed by feature importance)

---

## 🏗️ Project Structure

- **app.py                  # Streamlit web application**
- **ml.ipynb              # EDA, preprocessing, model training notebook**
- crop_model.pkl          # Trained Random Forest model
- scaler.pkl              # Fitted StandardScaler
- label_encoder.pkl       # Fitted LabelEncoder
- requirements.txt        # Python dependencies 
---

## ⚙️ How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/<YOUR USERNAME>/crop-recommendation-system
cd crop-recommendation-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open your browser at** `http://localhost:8501`

---

## 📦 Requirements
- streamlit
 - pandas
- scikit-learn
- joblib

---

## 🧠 How It Works

1. User enters 7 values via sliders (N, P, K, temperature, humidity, pH, rainfall)
2. Inputs are passed to the loaded Random Forest model (100 estimators)
3. Model returns a prediction and probability scores for all 22 crops
4. App displays the top recommended crop + top 3 with confidence percentages

---

## 📸 App Screenshot

![App Screenshot](images/app_screenshot.png)

---

## 👤 Author

**Rohith** · Enduro'26 · Strategic ERP 