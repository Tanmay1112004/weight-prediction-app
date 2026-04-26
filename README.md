# ⚖️ Weight Prediction AI — Health Analytics Micro-App

<p align="center">
  <b>From simple regression → real-world health insight tool</b><br>
  Built with Machine Learning + Streamlit for fast, interactive predictions
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-ML-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Streamlit-WebApp-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model-Linear%20Regression-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Focus-Health%20Analytics-red?style=flat-square"/>
</p>

---

## 💡 What This Project Does

This application predicts **estimated body weight based on height** using a trained machine learning model.

👉 But more importantly — it demonstrates how **simple models can power real-world health insights tools**

---

## 🚨 Problem Statement

In many fitness and health scenarios:

* People lack quick tools to estimate expected body metrics
* Raw data (height, weight) is often not analyzed properly
* Simple relationships are underutilized in decision-making

👉 Result: Missed opportunities for **basic health awareness**

---

## 🎯 Solution

A lightweight **ML-powered web app** that:

✅ Takes user height as input
✅ Predicts expected weight instantly
✅ Demonstrates correlation between physical attributes
✅ Provides a foundation for health analytics systems

---

## 🧠 Model Insight

This project uses **Linear Regression**, one of the most fundamental ML algorithms.

Here’s the core relationship:

genui{"math_block_widget_always_prefetch_v2": {"content": "y = mx + b"}}

Where:

* **y** = Predicted Weight
* **x** = Height
* **m** = Slope (relationship strength)
* **b** = Bias (baseline value)

👉 This simple equation powers the entire prediction engine.

---

## ⚡ Key Features

### 🎨 Clean & Modern UI

* Gradient-based styling
* Minimal and user-friendly layout
* Fast interaction with Streamlit

---

### ⚡ Real-Time Prediction

* Instant inference using pre-trained model
* No lag, no heavy computation

---

### 🧠 Lightweight ML Integration

* Pre-trained model stored using Pickle
* Efficient loading and execution

---

### 📦 Minimal Dependencies

* Easy setup
* Quick deployment
* Ideal for demos and interviews

---

## 🛠 Tech Stack

| Layer         | Technology                       |
| ------------- | -------------------------------- |
| Language      | Python                           |
| ML Model      | Scikit-learn (Linear Regression) |
| Frontend      | Streamlit                        |
| Data Handling | NumPy, Pandas                    |
| Model Storage | Pickle                           |

---

## 📸 Application Preview

### 📥 Input Interface

![Input](https://github.com/Tanmay1112004/weight-prediction-app/blob/main/HEIGHT%20%26%20WEIGHT/screenshots/Screenshot%202025-08-11%20125941.png?raw=true)

### 📊 Prediction Output

![Output](https://github.com/Tanmay1112004/weight-prediction-app/blob/main/HEIGHT%20%26%20WEIGHT/screenshots/Screenshot%202025-08-11%20130119.png?raw=true)

---

## 🏗 Project Structure

```
weight-prediction-app/
│
├── app.py                # Streamlit application
├── final_model.pkl       # Trained ML model
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/Tanmay1112004/weight-prediction-app.git
cd weight-prediction-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 What This Project Demonstrates

This may look simple — but it proves:

✅ Understanding of regression models
✅ Model serialization & deployment
✅ UI + ML integration
✅ Real-time prediction systems
✅ Clean product-focused design

---

## 💼 Recruiter Takeaway

This project shows:

👉 You can take a **mathematical concept**
👉 Convert it into a **working product**
👉 And deliver it through a **user-friendly interface**

That’s exactly what companies want.

---

## 🔮 Future Enhancements

* [ ] BMI calculation integration
* [ ] Multi-feature prediction (age, gender, activity level)
* [ ] Health recommendations engine
* [ ] REST API (FastAPI backend)
* [ ] Mobile-responsive UI

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and should not be used for medical decisions.

---

## ⭐ Support

If you found this useful:

* ⭐ Star the repo
* 🍴 Fork it
* 🚀 Improve it

---

## 👨‍💻 Author

**Tanmay Kshirsagar**

---

## 🔥 Final Thought

Simple models aren’t weak.

👉 **They’re powerful — when applied correctly.**
