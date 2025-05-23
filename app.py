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

st.set_page_config(layout="wide")
st.title("🔍 Hyperparameter Tuning Visual Sandbox")

uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🧾 Data Preview")
    st.dataframe(df)

    target_col = st.selectbox("🎯 Select target column (label)", df.columns)

    model_name = st.selectbox("🤖 Choose ML Model", ["KNN", "Random Forest", "Logistic Regression"])

    param_grid = {}
    model = None

    if model_name == "KNN":
        n_neighbors = st.slider("👥 n_neighbors range", 1, 20, (1, 10))
        weights_options = st.multiselect("⚖️ weights", ["uniform", "distance"], default=["uniform", "distance"])
        if not weights_options:
            st.warning("Select at least one weight option!")
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

    else:  # Logistic Regression
        c_vals = st.slider("🧮 C (inverse regularization strength) range", 0.01, 10.0, (0.1, 1.0), step=0.1)
        solvers = st.multiselect("⚙️ Solver", ['lbfgs', 'liblinear', 'saga'], default=['lbfgs', 'liblinear'])
        if not solvers:
            st.warning("Select at least one solver!")
        param_grid = {
            'C': np.round(np.linspace(c_vals[0], c_vals[1], num=5), 2),
            'solver': solvers
        }
        model = LogisticRegression(max_iter=5000)

    run_grid = st.button("🚀 Run Grid Search")

    if run_grid:
        if not param_grid or any(len(v) == 0 for v in param_grid.values()):
            st.error("Please make sure you have selected valid hyperparameter options.")
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
                        st.error("Not enough samples in each class to run Grid Search. Add more data.")
                    else:
                        scorer = make_scorer(accuracy_score)
                        grid = GridSearchCV(model, param_grid, scoring=scorer, cv=safe_cv, n_jobs=-1)

                        with st.spinner("Running Grid Search... This may take a moment."):
                            grid.fit(X_train, y_train)

                        st.success("✅ Grid Search Completed!")
                        st.write(f"**Best parameters:** `{grid.best_params_}`")
                        st.write(f"**Best cross-validation accuracy:** `{grid.best_score_:.4f}`")

                        test_score = accuracy_score(y_test, grid.predict(X_test))
                        st.write(f"**Test set accuracy:** `{test_score:.4f}`")

                        results = grid.cv_results_
                        param_keys = list(param_grid.keys())
                        if len(param_keys) == 2:
                            p1, p2 = param_keys[0], param_keys[1]

                            df_viz = pd.DataFrame({
                                p1: [str(x) for x in results[f'param_{p1}']],
                                p2: [str(x) for x in results[f'param_{p2}']],
                                'score': results['mean_test_score']
                            })

                            # Heatmap
                            st.write("### 🔥 Heatmap")
                            heatmap_data = df_viz.pivot(index=p2, columns=p1, values='score')
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="coolwarm", ax=ax)
                            st.pyplot(fig)

                            # Line Plot
                            st.write("### 📈 Line Plot")
                            fig_line = px.line(df_viz, x=p1, y='score', color=p2, markers=True,
                                              title=f"{p1} vs Accuracy (Grouped by {p2})")
                            st.plotly_chart(fig_line)

                            # 3D Plot
                            st.write("### 🕹️ 3D Plot")
                            if not np.issubdtype(df_viz[p2].dtype, np.number):
                                df_viz[p2 + '_enc'] = df_viz[p2].astype('category').cat.codes
                                p2_enc = p2 + '_enc'
                            else:
                                p2_enc = p2

                            fig_3d = px.scatter_3d(df_viz, x=p1, y=p2_enc, z='score',
                                                   color='score', size='score',
                                                   title=f"3D Accuracy Plot ({p1}, {p2}, Accuracy)",
                                                   labels={p1: p1, p2_enc: p2, 'score': 'Accuracy'})
                            st.plotly_chart(fig_3d)

                        else:
                            st.warning("⚠️ Please tune exactly two hyperparameters to view visualizations.")

            except Exception as e:
                st.error(f"Error during processing: {e}")
else:
    st.info("Please upload a CSV file to get started.")
