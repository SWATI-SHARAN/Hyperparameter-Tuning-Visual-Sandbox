import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.utils.multiclass import type_of_target
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🔍 Hyperparameter Tuning Visual Sandbox")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🧾 Data Preview")
    st.dataframe(df)

    # Step 2: Select target column
    target_col = st.selectbox("🎯 Select target column (label)", df.columns)

    # Step 3: Select model
    model_name = st.selectbox("🤖 Choose ML Model", ["KNN", "Random Forest", "Logistic Regression"])

    # Step 4: Define hyperparameters grids
    param_grid = {}
    if model_name == "KNN":
        st.write("Tune number of neighbors (n_neighbors) and weight function (weights)")
        n_neighbors = st.slider("👥 n_neighbors", 1, 20, (1, 10))
        weights_options = st.multiselect("⚖️weights", ["uniform", "distance"], default=["uniform", "distance"])
        param_grid = {
            'n_neighbors': list(range(n_neighbors[0], n_neighbors[1] + 1)),
            'weights': weights_options
        }
        model = KNeighborsClassifier()

    elif model_name == "Random Forest":
        st.write("Tune number of trees (n_estimators) and max depth (max_depth)")
        n_estimators = st.slider("🌲n_estimators", 10, 200, (10, 50))
        max_depth = st.slider("🧬max_depth", 1, 20, (1, 10))
        param_grid = {
            'n_estimators': list(range(n_estimators[0], n_estimators[1] + 1, 10)),
            'max_depth': list(range(max_depth[0], max_depth[1] + 1))
        }
        model = RandomForestClassifier(random_state=42)

    else:  # Logistic Regression
        st.write("Tune regularization strength (C) and solver")
        c_vals = st.slider("🧮C (inverse regularization strength)", 0.01, 10.0, (0.01, 1.0), step=0.01)
        solvers = st.multiselect("⚙️Solver", ['lbfgs', 'liblinear', 'saga'], default=['lbfgs', 'liblinear'])
        param_grid = {
            'C': np.round(np.linspace(c_vals[0], c_vals[1], num=10), 2),
            'solver': solvers
        }
        model = LogisticRegression(max_iter=5000)

    # Step 5: Run Grid Search
    if st.button("🚀 Run Grid Search"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X = pd.get_dummies(X)
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        label_type = type_of_target(y)
        if label_type not in ['binary', 'multiclass']:
            st.error(f"❌ Target column must be categorical for classification models. Detected label type: '{label_type}'.")
        else:
            st.info("🔄 Running GridSearchCV...⏳")

            scorer = make_scorer(accuracy_score)
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scorer, n_jobs=-1)
            grid_search.fit(X, y)

            st.success("✅ Grid Search Completed!")
            st.write(f"**Best parameters:** `{grid_search.best_params_}`")
            st.write(f"**Best accuracy:** `{grid_search.best_score_:.4f}`")

            # Results processing
            results = grid_search.cv_results_
            params = list(param_grid.keys())

            if len(params) == 2:
                param1, param2 = params[0], params[1]
                scores = results['mean_test_score']
                param1_vals = results['param_' + param1]
                param2_vals = results['param_' + param2]

                df_viz = pd.DataFrame({
                    param1: param1_vals,
                    param2: param2_vals,
                    'score': scores
                })

                viz_type = st.selectbox("📊 Select Visualization Type", ["Heatmap", "Line Plot", "3D Plot"])

                # Heatmap
                if viz_type == "Heatmap":
                    heatmap_data = df_viz.pivot_table(index=param2, columns=param1, values='score', aggfunc='mean')
                    st.write(f"### 🔥 Heatmap: Accuracy vs {param1} & {param2}")
                    fig, ax = plt.subplots()
                    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".3f", ax=ax)
                    st.pyplot(fig)

                # Line Plot
                elif viz_type == "Line Plot":
                    if not np.issubdtype(df_viz[param2].dtype, np.number):
                        df_viz[param2] = df_viz[param2].astype(str)
                    fig = px.line(df_viz, x=param1, y='score', color=param2, markers=True,
                                  title=f"📈 Line Plot: {param1} vs Accuracy (Grouped by {param2})")
                    st.plotly_chart(fig)

                # 3D Plot
                elif viz_type == "3D Plot":
                    if not np.issubdtype(df_viz[param2].dtype, np.number):
                        df_viz[param2 + '_enc'] = df_viz[param2].astype('category').cat.codes
                        param2_3d = param2 + '_enc'
                    else:
                        param2_3d = param2

                    fig = px.scatter_3d(df_viz,
                                        x=param1,
                                        y=param2_3d,
                                        z='score',
                                        color='score',
                                        size='score',
                                        title=f"🧊 3D Plot: {param1}, {param2}, and Accuracy",
                                        labels={param1: param1, param2_3d: param2, 'score': 'Accuracy'})
                    st.plotly_chart(fig)
            else:
                st.warning("⚠️ Visualizations require tuning exactly 2 hyperparameters.")
