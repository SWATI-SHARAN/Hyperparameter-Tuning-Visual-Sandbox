import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.utils.multiclass import type_of_target
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Page Setup ----------
st.set_page_config(layout="wide", page_title="🧠 Hyperparameter Tuner", page_icon="💡")

# ---------- Custom Neon CSS + Font + Background ----------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        background-color: #0d0d0d;
        color: #00ffff;
    }
    h1, h2, h3, h4 {
        color: #00ffff;
        text-shadow: 0 0 15px #00ffff;
    }
    .stButton>button {
        background: #111;
        color: #0ff;
        border: 1px solid #0ff;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.4s ease-in-out;
        text-shadow: 0 0 6px #0ff;
    }
    .stButton>button:hover {
        background: #0ff;
        color: #000;
        box-shadow: 0 0 20px #0ff;
    }
    .stSelectbox > div, .stSlider > div {
        color: #00ffff !important;
    }
    .block-container {
        padding: 1rem 3rem;
    }
    .card {
        background: #111;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #00ffff;
        border-radius: 15px;
        box-shadow: 0 0 15px #00ffff;
        transition: 0.3s ease-in-out;
    }
    .card:hover {
        box-shadow: 0 0 25px #ff00ff, 0 0 40px #00ffff;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Animated Particle Background ----------
st.components.v1.html("""
    <div id="particles-js"></div>
    <style>
        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: #0d0d0d;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <script>
    particlesJS('particles-js', {
      "particles": {
        "number": {"value": 80},
        "color": {"value": "#0ff"},
        "shape": {"type": "circle"},
        "opacity": {"value": 0.3},
        "size": {"value": 3},
        "line_linked": {"enable": true, "color": "#0ff", "opacity": 0.2},
        "move": {"enable": true, "speed": 2}
      },
      "interactivity": {
        "events": {
          "onhover": {"enable": true, "mode": "repulse"},
          "onclick": {"enable": true, "mode": "push"}
        }
      },
      "retina_detect": true
    });
    </script>
""", height=0)

# ---------- Title ----------
st.markdown("<h1 style='text-align:center;'>💡 Hyperparameter Tuning Visual Sandbox</h1>", unsafe_allow_html=True)

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("<div class='card'><h2>🧾 Data Preview</h2></div>", unsafe_allow_html=True)
    st.dataframe(df.style.set_properties(**{'background-color': '#111', 'color': '#0ff'}), height=300)

    target_col = st.selectbox("🎯 Select target column (label)", df.columns)

    model_name = st.selectbox("🤖 Choose ML Model", ["KNN", "Random Forest", "Logistic Regression"])

    param_grid = {}
    model = None

    with st.expander("⚙️ Model Hyperparameters", expanded=True):
        if model_name == "KNN":
            n_neighbors = st.slider("👥 n_neighbors range", 1, 20, (1, 10))
            weights_options = st.multiselect("⚖️ weights", ["uniform", "distance"], default=["uniform", "distance"])
            param_grid = {
                'n_neighbors': list(range(n_neighbors[0], n_neighbors[1] + 1)),
                'weights': weights_options
            }
            model = KNeighborsClassifier()

        elif model_name == "Random Forest":
            n_estimators = st.slider("🌲 n_estimators range", 10, 100, (10, 30), step=10)
            max_depth = st.slider("🧬 max_depth range", 1, 10, (1, 3))
            param_grid = {
                'n_estimators': list(range(n_estimators[0], n_estimators[1] + 1, 10)),
                'max_depth': list(range(max_depth[0], max_depth[1] + 1))
            }
            model = RandomForestClassifier(random_state=42)

        else:
            c_vals = st.slider("🧮 C range", 0.01, 10.0, (0.1, 1.0), step=0.1)
            solvers = st.multiselect("⚙️ Solver", ['lbfgs', 'liblinear', 'saga'], default=['lbfgs', 'liblinear'])
            param_grid = {
                'C': np.round(np.linspace(c_vals[0], c_vals[1], num=5), 2),
                'solver': solvers
            }
            model = LogisticRegression(max_iter=5000)

    run_grid = st.button("🚀 Run Grid Search")

    if run_grid:
        if not param_grid or any(len(v) == 0 for v in param_grid.values()):
            st.error("Please select valid hyperparameter options.")
        else:
            try:
                X = pd.get_dummies(df.drop(columns=[target_col]))
                y = df[target_col]

                if y.dtype == 'object' or str(y.dtype).startswith('category'):
                    y = y.astype('category').cat.codes

                label_type = type_of_target(y)
                if label_type not in ['binary', 'multiclass']:
                    st.error(f"❌ Target column must be categorical. Detected: '{label_type}'")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                    min_class_count = min(Counter(y_train).values())
                    safe_cv = min(3, min_class_count)

                    if safe_cv < 2:
                        st.error("Not enough samples per class to run Grid Search.")
                    else:
                        scorer = make_scorer(accuracy_score)
                        grid = GridSearchCV(model, param_grid, scoring=scorer, cv=safe_cv, n_jobs=-1)

                        with st.spinner("⏳ Running Grid Search..."):
                            grid.fit(X_train, y_train)

                        st.success("✅ Grid Search Completed!")

                        col1, col2, col3 = st.columns(3)
                        col1.markdown(f"<div class='card'><b>Best Params</b><br>{grid.best_params_}</div>", unsafe_allow_html=True)
                        col2.markdown(f"<div class='card'><b>CV Accuracy</b><br>{grid.best_score_:.4f}</div>", unsafe_allow_html=True)
                        col3.markdown(f"<div class='card'><b>Test Accuracy</b><br>{accuracy_score(y_test, grid.predict(X_test)):.4f}</div>", unsafe_allow_html=True)

                        results = grid.cv_results_
                        param_keys = list(param_grid.keys())

                        if len(param_keys) == 2:
                            p1, p2 = param_keys[0], param_keys[1]

                            df_viz = pd.DataFrame({
                                p1: [str(x) for x in results[f'param_{p1}']],
                                p2: [str(x) for x in results[f'param_{p2}']],
                                'score': results['mean_test_score']
                            })

                            st.markdown("<h3>🔥 Heatmap</h3>", unsafe_allow_html=True)
                            heatmap_data = df_viz.pivot(index=p2, columns=p1, values='score')
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", ax=ax)
                            st.pyplot(fig)

                            st.markdown("<h3>📈 Line Plot</h3>", unsafe_allow_html=True)
                            fig = px.line(df_viz, x=p1, y='score', color=p2, markers=True)
                            st.plotly_chart(fig)

                            st.markdown("<h3>🧊 3D Plot</h3>", unsafe_allow_html=True)
                            if not np.issubdtype(df_viz[p2].dtype, np.number):
                                df_viz[p2 + '_enc'] = df_viz[p2].astype('category').cat.codes
                                p2_enc = p2 + '_enc'
                            else:
                                p2_enc = p2

                            fig = px.scatter_3d(df_viz, x=p1, y=p2_enc, z='score',
                                                color='score', size='score')
                            st.plotly_chart(fig)
                        else:
                            st.warning("Visuals are available for exactly 2 hyperparameters only.")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("📥 Please upload a CSV file to get started.")
