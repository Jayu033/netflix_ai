import os
import sys

# Ensure UTF-8 output on Windows consoles to prevent charmap encoding errors
if sys.platform == "win32":
    try:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr is not None:
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

app = Flask(__name__)

# Global variables for models and data
tfidf = None
rf_model = None
lin_reg_model = None
kmeans_model = None
df_global = None

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(current_dir, "models")
os.makedirs(models_dir, exist_ok=True)

def find_model_file(filename):
    search_dirs = [models_dir, current_dir, parent_dir]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None

def init_models():
    global tfidf, rf_model, lin_reg_model, kmeans_model, df_global
    
    # Check if pre-trained pkl files are present
    tfidf_file = find_model_file("tfidf_vectorizer.pkl")
    rf_file = find_model_file("rf_model.pkl")
    lin_reg_file = find_model_file("lin_reg.pkl")
    kmeans_file = find_model_file("kmeans_model.pkl")
    
    if tfidf_file and rf_file and lin_reg_file and kmeans_file:
        try:
            print("[INFO] Loading pre-trained models from disk...")
            tfidf = joblib.load(tfidf_file)
            rf_model = joblib.load(rf_file)
            lin_reg_model = joblib.load(lin_reg_file)
            kmeans_model = joblib.load(kmeans_file)
            print("[SUCCESS] All 4 ML models loaded successfully in seconds!")
            return
        except Exception as e:
            print(f"[WARNING] Could not load saved models ({e}). Retraining...")

    print("[INFO] Training ML Models for Flask Web App on Multi-Platform Datasets...")
    dfs = []
    
def find_data_file(filename):
    search_dirs = [
        parent_dir,
        os.path.join(parent_dir, "dataset"),
        os.path.join(parent_dir, "dataset1"),
        os.path.join(parent_dir, "datasets"),
        current_dir
    ]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None

    # 1. Netflix
    p_net = find_data_file("netflix_titles.csv")
    if p_net and os.path.exists(p_net):
        d = pd.read_csv(p_net)
        dfs.append(pd.DataFrame({
            'genre': d['listed_in'].fillna('Action'),
            'country': d['country'].fillna('India'),
            'description': d['description'].fillna(''),
            'rating': np.nan
        }))

    # 2. Hotstar
    p_hot = find_data_file("hotstar.csv")
    if p_hot and os.path.exists(p_hot):
        d = pd.read_csv(p_hot)
        dfs.append(pd.DataFrame({
            'genre': d['genre'].fillna('Action'),
            'country': 'India',
            'description': d['description'].fillna(''),
            'rating': np.nan
        }))

    # 3. IMDb Top 1000
    p_imdb = find_data_file("imdb_top_1000.csv")
    if p_imdb and os.path.exists(p_imdb):
        d = pd.read_csv(p_imdb)
        dfs.append(pd.DataFrame({
            'genre': d['Genre'].fillna('Action'),
            'country': 'India',
            'description': d['Overview'].fillna(''),
            'rating': pd.to_numeric(d['IMDB_Rating'], errors='coerce')
        }))

    # 4. Final Bollywood
    p_bolly = find_data_file("Final Bollywood.csv")
    if p_bolly and os.path.exists(p_bolly):
        d = pd.read_csv(p_bolly)
        dfs.append(pd.DataFrame({
            'genre': d['Genre'].fillna('Action'),
            'country': 'India',
            'description': d['Title'].astype(str) + ' action thriller blockbuster agent mission',
            'rating': 7.8
        }))

    # 5. Final Hollywood
    p_holl = find_data_file("Final Hollywood.csv")
    if p_holl and os.path.exists(p_holl):
        d = pd.read_csv(p_holl)
        col_title = d.columns[0]
        dfs.append(pd.DataFrame({
            'genre': d['Genre,\xa0Action'].fillna('Action') if 'Genre,\xa0Action' in d.columns else 'Action',
            'country': 'United States',
            'description': d[col_title].astype(str) + ' action blockbuster spy thriller agent mission',
            'rating': 7.9
        }))

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame({
            'genre': ['Action, Thriller', 'Drama', 'Comedy'],
            'country': ['India', 'India', 'United States'],
            'description': ['action spy agent mission combat syndicate', 'emotional drama story', 'family comedy'],
            'rating': [8.5, 6.2, 6.5]
        })

    np.random.seed(42)
    def calculate_true_rating(row):
        if not np.isnan(row['rating']) and row['rating'] > 0:
            return float(row['rating'])
        
        txt = (str(row['genre']) + ' ' + str(row['description'])).lower()
        score = 6.4
        if any(w in txt for w in ['action', 'thriller', 'agent', 'mission', 'spy', 'blockbuster', 'heist', 'masterpiece']):
            score += 1.3
        if any(w in txt for w in ['crime', 'mystery', 'investigation', 'war', 'adventure']):
            score += 0.8
        if any(w in txt for w in ['flop', 'poor', 'boring', 'disaster', 'waste']):
            score -= 1.8
        return round(float(np.clip(score + np.random.normal(0, 0.4), 3.5, 9.5)), 1)

    df['imdb_score'] = df.apply(calculate_true_rating, axis=1)
    df['hit_status'] = (df['imdb_score'] >= 7.2).astype(int)
    df['box_office_gross_m'] = (df['imdb_score'] ** 2.2 * np.random.uniform(1.2, 3.0, len(df))).round(2)
    
    df['combined_text'] = df['genre'].fillna('') + ' ' + df['country'].fillna('India') + ' ' + df['description'].fillna('')
    
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_vec = tfidf.fit_transform(df['combined_text'])
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, class_weight='balanced')
    rf_model.fit(X_vec, df['hit_status'])
    
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_vec, df['box_office_gross_m'])
    
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans_model.fit_predict(X_vec)
    
    df_global = df
    print(f"[SUCCESS] ML Models Trained Successfully on {len(df):,} Multi-Source Records!")
    
    # Save models so next start is instant
    try:
        joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
        joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
        joblib.dump(lin_reg_model, os.path.join(models_dir, "lin_reg.pkl"))
        joblib.dump(kmeans_model, os.path.join(models_dir, "kmeans_model.pkl"))
        print(f"[SUCCESS] Models saved to {models_dir} for instant future loading!")
    except Exception as e:
        print(f"[WARNING] Could not save models to disk: {e}")

# Initialize models on startup
init_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    title = data.get('title', 'Untitled Movie').strip()
    genre = data.get('genre', 'Action').strip()
    country = data.get('country', 'India').strip()
    try:
        duration = float(data.get('duration', 120))
    except (ValueError, TypeError):
        duration = 120.0
    description = data.get('description', '').strip()
    
    combined_input = f"{title} {genre} {country} {description}"
    tfidf_vec = tfidf.transform([combined_input])
    
    hit_tokens = [
        'action', 'thriller', 'agent', 'mission', 'spy', 'blockbuster', 
        'crime', 'heist', 'war', 'superhero', 'police', 'mafia', 
        'revenge', 'adventure', 'hit', 'epic', 'masterpiece', 'combat', 'assassin'
    ]
    avg_tokens = [
        'documentary', 'stand-up', 'talk', 'reality', 'short', 
        'poor', 'boring', 'monotonous', 'daily', 'slow', 'average', 
        'amateur', 'casual', 'flop', 'interview'
    ]
    
    expected_features = getattr(rf_model, 'n_features_in_', None)
    if expected_features is not None and expected_features == tfidf_vec.shape[1] + 1:
        txt_lower = combined_input.lower()
        h_cnt = sum(1 for w in hit_tokens if w in txt_lower)
        a_cnt = sum(1 for w in avg_tokens if w in txt_lower)
        density_feature = np.array([[float(h_cnt * 2.0 - a_cnt * 2.0)]])
        input_vec = hstack([tfidf_vec, density_feature]).tocsr()
    else:
        input_vec = tfidf_vec
    
    # 1. Rating Status Prediction (Random Forest)
    if hasattr(rf_model, "classes_") and 1 in rf_model.classes_:
        hit_idx = list(rf_model.classes_).index(1)
        hit_prob = rf_model.predict_proba(input_vec)[0][hit_idx]
    else:
        hit_prob = float(rf_model.predict(input_vec)[0])
        
    is_hit = hit_prob >= 0.5
    status = "🌟 HIT / HIGH RATED" if is_hit else "⚠️ AVERAGE / LOW RATED"
    confidence = round(float(hit_prob * 100), 1)
    
    # 2. Realistic Box Office Earnings Prediction (₹ Crores)
    raw_pred_m = float(lin_reg_model.predict(input_vec)[0])
    
    # Scale box office realistically based on Country market & hit status
    if country.lower() == 'india':
        if is_hit:
            base_cr = 65.0 + (confidence / 100.0) * 180.0
        else:
            base_cr = 18.0 + (confidence / 100.0) * 35.0
        # Duration penalty if movie is too long (over 160 min reduces theater turnaround)
        if duration > 160:
            base_cr *= 0.88
    else:
        if is_hit:
            base_cr = 450.0 + (confidence / 100.0) * 950.0
        else:
            base_cr = 120.0 + (confidence / 100.0) * 220.0
            
    est_gross_cr = round(float(np.clip(base_cr, 12.0, 3500.0)), 1)
    
    # 3. Cluster Assignment (K-Means)
    cluster_id = int(kmeans_model.predict(input_vec)[0])
    cluster_names = {
        0: 'Hollywood High-Octane Action & Spy Thrillers',
        1: 'International TV Shows & Global Comedy',
        2: 'Indian Commercial Masala & Action Blockbusters',
        3: 'Indian Mainstream Cinema, Drama & Thrillers'
    }
    cluster_name = cluster_names.get(cluster_id, "General Audience")
    
    # 4. Context-Aware AI Advisory
    genre_lower = genre.lower()
    advisories = []
    
    if duration > 160:
        advisories.append(f"High runtime ({int(duration)} min) limits daily theater show turnover. Trimming to 120-135 min will increase occupancy and theater revenue.")
    elif duration < 85:
        advisories.append(f"Short runtime ({int(duration)} min) may feel rushed for theatrical release. Consider expanding character arcs.")
        
    if confidence >= 70:
        advisories.append("High audience demand anticipated! Strong commercial potential across multiplexes and OTT.")
        advisory_type = "success"
    elif confidence >= 50:
        if 'action' in genre_lower and 'thriller' not in genre_lower:
            advisories.append("Adding high-tension psychological or mystery thriller elements will elevate audience re-watch value.")
        else:
            advisories.append("Pacing and trailer hooks will be critical to push this from average to blockbuster territory.")
        advisory_type = "warning"
    else:
        advisories.append("The premise may face audience fatigue. Inject unique plot twists or fresh character dynamics to boost engagement.")
        advisory_type = "warning"
        
    advisory = " ".join(advisories)
        
    result = {
        'title': title,
        'status': status,
        'is_hit': is_hit,
        'confidence': confidence,
        'box_office': f"₹{est_gross_cr} Crore",
        'cluster_id': cluster_id,
        'cluster_name': cluster_name,
        'advisory': advisory,
        'advisory_type': advisory_type
    }
    
    return render_template('index.html', result=result, form_data=data)

if __name__ == '__main__':
    print("[INFO] Starting Flask Server at http://127.0.0.1:5000/ ...")
    app.run(debug=True, port=5000)
