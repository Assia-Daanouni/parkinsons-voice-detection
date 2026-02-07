#  Parkinson's Voice Detection

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-green)](#)  

**Early detection of Parkinson’s disease using voice analysis and machine learning.**  
This project includes a **Flask API**, an **MLP model**, and numeric feature analysis for research and public use.

---

##  Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Technologies](#technologies)  
- [Installation](#installation)  
- [How to Use](#how-to-use)  
- [Models Used](#models-used)  
- [Results](#results)  
- [Team](#team)  
- [License](#license)  
- [Video Demo](#video-demo)  

---

##  Overview
Parkinson’s disease is often detected **too late**, after major brain damage has already occurred.  
Small changes in voice appear early, even before motor symptoms, making **voice-based detection a powerful, low-cost, and non-invasive screening tool**.  

This project demonstrates how **voice recordings** can be used with **machine learning models** to detect Parkinson’s disease early.

---

##  Features
- **Voice Recording Detection for Public Users:**  
  Users can record their voice, and the system extracts key features and predicts Parkinson’s likelihood using an **MLP (Neural Network) model**.  

- **Manual Numeric Input for Researchers:**  
  Enter **22 vocal features** manually to study model predictions and experiment with different scenarios.  

- **Flask API:**  
  Connects trained machine learning models to a usable web application for **real-time predictions**.

---

##  Technologies
- Python  
- Flask  
- Scikit-learn  
- Pandas, NumPy, Matplotlib  
- SMOTE (for class balancing)  

---

##  Installation

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/parkinsons-voice-detection.git
cd parkinsons-voice-detection

2. **Create a virtual environment**
python -m venv venv
# Activate:
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

3. **Install dependencies**
pip install -r requirements.txt

4. **python app.py**
python app.py
