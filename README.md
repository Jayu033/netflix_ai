# 🎬 CineAI - Pre-Release OTT Content Rating & Show Clustering Engine

> **BCA Final Year Major Project (Academic Year 2026–2027)**  
> **Developed by:** Vastani Jay & Meet Radadiya

---

## 📌 Project Overview
**CineAI** is an advanced End-to-End Machine Learning intelligence platform designed for film studios, producers, and OTT streaming services (Netflix, Amazon Prime, Hotstar, Apple TV+). It evaluates unreleased movie/show scripts to predict:
1. **Hit vs. Average Rating Status** (Random Forest Classifier - **91.50% Accuracy**)
2. **Financial Box Office Revenue** in ₹ Crores (Linear Regression Model)
3. **Target Audience Segments** (K-Means Clustering - 4 Content Clusters)
4. **Actionable AI Script Advisory** for filmmakers.

---

## 🛠️ Technology Stack
- **Language:** Python 3.10
- **Machine Learning Libraries:** `scikit-learn`, `pandas`, `numpy`, `scipy`
- **NLP Vectorization:** TF-IDF Vectorizer (`max_features=1000`)
- **Web Backend:** Flask Web Framework
- **Frontend UI:** Glassmorphism Dark Mode (Bootstrap 5, Outfit & Inter Google Fonts, CSS3 Animations)
- **Data Visualization:** `matplotlib`, `seaborn`

---

## 📊 Dataset Audit (1.2 Million+ Records)
The project integrates 13 multi-platform dataset sources totaling **1,202,133 Rows (288 MB)**:
- Netflix Titles Dataset (`netflix_titles.csv`)
- Disney+ Hotstar Dataset (`hotstar.csv`)
- Amazon Prime Titles (`amazon_prime_titles.csv`)
- Bollywood Film Database (`Final Bollywood.csv`)
- Hollywood Film Database (`Final Hollywood.csv`)
- IMDb Top 1000 Database (`imdb_top_1000.csv`)
- Apple TV+ & TMDB Box Office Datasets

---

## 📈 Model Performance Metrics
| Model Name | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **91.50%** | **88.40%** | **90.10%** | **88.20%** |
| Logistic Regression | 79.91% | 83.99% | 91.30% | 87.49% |
| K-Nearest Neighbors (KNN) | 76.16% | 81.33% | 89.60% | 85.26% |

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Jayu033/netflix_ai.git
cd netflix_ai
```

### 2. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### 3. Launch Flask Web Application
```bash
cd flask_app
python app.py
```
Open your browser at `http://127.0.0.1:5000` to interact with the live AI Dashboard!

---

## 📜 License
Developed as a Major Academic Project for BCA Degree (2026–2027).
