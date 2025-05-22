import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Hyperparameter Tuning Visual Sandbox")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df)

    # Step 2: Select target column
    target_col = st.selectbox("Select target column (label)", df.columns)

    # Step 3: Select model
    model_name = st.selectbox("Choose ML Model", ["KNN", "Random Forest", "Logistic Regression"])

    # Step 4: Define hyperparameters grids for each model (two params to visualize)
    param_grid = {}
    if model_name == "KNN":
        st.write("Tune number of neighbors (n_neighbors) and weight function (weights)")
        n_neighbors = st.slider("n_neighbors", 1, 20, (1, 10))
        weights_options = st.multiselect("weights", ["uniform", "distance"], default=["uniform", "distance"])
        param_grid = {
            'n_neighbors': list(range(n_neighbors[0], n_neighbors[1] + 1)),
            'weights': weights_options
        }
        model = KNeighborsClassifier()
    elif model_name == "Random Forest":
        st.write("Tune number of trees (n_estimators) and max depth (max_depth)")
        n_estimators = st.slider("n_estimators", 10, 200, (10, 50))
        max_depth = st.slider("max_depth", 1, 20, (1, 10))
        param_grid = {
            'n_estimators': list(range(n_estimators[0], n_estimators[1] + 1, 10)),
            'max_depth': list(range(max_depth[0], max_depth[1] + 1))
        }
        model = RandomForestClassifier(random_state=42)
    else:  # Logistic Regression
        st.write("Tune regularization strength (C) and solver")
        c_vals = st.slider("C (inverse regularization strength)", 0.01, 10.0, (0.01, 1.0), step=0.01)
        solvers = st.multiselect("Solver", ['lbfgs', 'liblinear', 'saga'], default=['lbfgs', 'liblinear'])
        param_grid = {
            'C': np.round(np.linspace(c_vals[0], c_vals[1], num=10), 2),
            'solver': solvers
        }
        model = LogisticRegression(max_iter=5000)

    # Step 5: Button to start Grid Search
    if st.button("Run Grid Search"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Simple preprocessing: encode categorical features & labels
        X = pd.get_dummies(X)
        if y.dtype == 'object':
            y = y.astype('category').cat.codes

        st.write("Running GridSearchCV... (this may take some seconds)")

        scorer = make_scorer(accuracy_score)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scorer, n_jobs=-1)
        grid_search.fit(X, y)

        st.write(f"Best parameters: {grid_search.best_params_}")
        st.write(f"Best accuracy: {grid_search.best_score_:.4f}")

        # Step 6: Visualize results as heatmap (only works if exactly 2 hyperparams)
        results = grid_search.cv_results_

        # Extract param names (keys)
        params = list(param_grid.keys())
        if len(params) == 2:
            param1 = params[0]
            param2 = params[1]

            scores = results['mean_test_score']
            param1_vals = results['param_' + param1].data
            param2_vals = results['param_' + param2].data

            # Create pivot table for heatmap
            df_heatmap = pd.DataFrame({
                param1: param1_vals,
                param2: param2_vals,
                'score': scores
            })
            heatmap_data = df_heatmap.pivot(index=param2, columns=param1, values='score')

            st.write(f"### Heatmap of accuracy scores for {param1} vs {param2}")
            fig, ax = plt.subplots()
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", ax=ax)
            plt.xlabel(param1)
            plt.ylabel(param2)
            st.pyplot(fig)
        else:
            st.write("Heatmap visualization only supports tuning exactly 2 hyperparameters at a time.")
