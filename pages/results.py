import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import joblib
import os
import plotly.graph_objects as go
import statsmodels.api as sm

# =======================
# STYLE CSS
# =======================
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(to bottom, #E6E6FA, #FFFFFF); }
    .main-title { color: #9370DB; text-align: center; font-size: 3em; font-weight: bold; margin-bottom: 20px; }
    .cluster-title { color:#9370DB; font-size:1.5em; font-weight:bold; text-align:center; margin:20px 0; }
    .interpretation { color:#6A5ACD; font-size:1.2em; text-align:center; margin:20px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">Vos Résultats</p>', unsafe_allow_html=True)

# =======================
# MAPPING DES QUESTIONS
# =======================
question_mapping = {
    'Q1': {'variable': 'Work_Stress', 'inverse': False},
    'Q2': {'variable': 'Sleep_Hours', 'inverse': True},
    'Q3': {'variable': 'Social_Support_Score', 'inverse': True},
    'Q4': {'variable': 'Financial_Stress', 'inverse': False},
    'Q5': {'variable': 'Self_Esteem_Score', 'inverse': True},
    'Q6': {'variable': 'Family_History_Mental_Illness', 'inverse': True},
    'Q7': {'variable': 'Loneliness_Score', 'inverse': False}
}

continuous_cols = ['Age', 'Sleep_Hours', 'Social_Support_Score', 'Financial_Stress',
                   'Work_Stress', 'Self_Esteem_Score', 'Loneliness_Score']
binary_cols = ['Family_History_Mental_Illness']
dependent_vars = ['Anxiety_Score', 'Depression_Score', 'Stress_Level']
independent_vars = continuous_cols + binary_cols

# =======================
# CHARGEMENT DES DONNÉES
# =======================
df_ref = pd.read_csv("df_clusters.csv")

scaler_path, model_path = 'scaler.save', 'kmeans_model.save'

if os.path.exists(scaler_path) and os.path.exists(model_path):
    scaler_ref = joblib.load(scaler_path)
    kmeans = joblib.load(model_path)
else:
    scaler_ref = StandardScaler()
    X_ref_continuous = df_ref[continuous_cols]
    X_ref_continuous_scaled = scaler_ref.fit_transform(X_ref_continuous)
    X_ref_binary = df_ref[binary_cols].values
    X_ref_scaled = np.hstack((X_ref_continuous_scaled, X_ref_binary))
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X_ref_scaled)
    joblib.dump(scaler_ref, scaler_path)
    joblib.dump(kmeans, model_path)

# =======================
# DONNÉES UTILISATEUR
# =======================
if 'profile_info' in st.session_state:
    date_naissance = st.session_state.profile_info['date_naissance']
    age = datetime.now().year - date_naissance.year - (
        (datetime.now().month, datetime.now().day) < (date_naissance.month, date_naissance.day))
else:
    age = 25
age_normalise = int(((age - 18) / (99 - 18)) * 9) + 1

user_data = {col: [0] for col in continuous_cols + binary_cols}
user_data['Age'] = [age_normalise]

for q, response in st.session_state.reponses_df.iloc[0].items():
    if q in question_mapping:
        val = 10 - response if question_mapping[q]['inverse'] else response
        user_data[question_mapping[q]['variable']] = [val]

if 'Q6' in st.session_state.reponses_df.columns:
    user_data['Family_History_Mental_Illness'] = [1 if st.session_state.reponses_df.iloc[0]['Q6'] <= 5 else 0]

user_df = pd.DataFrame(user_data)[continuous_cols + binary_cols]

# Clustering
user_continuous_scaled = scaler_ref.transform(user_df[continuous_cols])
user_binary = user_df[binary_cols].values
user_data_scaled = np.hstack((user_continuous_scaled, user_binary))
user_cluster = kmeans.predict(user_data_scaled.reshape(1, -1))[0]

# =======================
# CLUSTERING - RADAR 1
# =======================
features = continuous_cols + binary_cols
user_values = user_df.iloc[0].values.tolist()
user_values.append(user_values[0])

feature_labels = {
    'Age': 'Âge', 'Sleep_Hours': 'Sommeil',
    'Social_Support_Score': 'Soutien social', 'Financial_Stress': 'Stress financier',
    'Work_Stress': 'Stress travail', 'Self_Esteem_Score': 'Estime de soi',
    'Family_History_Mental_Illness': 'Antécédents familiaux', 'Loneliness_Score': 'Solitude'
}
features_display = [feature_labels[f] for f in features]
features_display.append(features_display[0])

fig_cluster = go.Figure()
fig_cluster.add_trace(go.Scatterpolar(
    r=user_values,
    theta=features_display,
    fill='toself',
    name='Vos valeurs',
    line_color='#9370DB',
    fillcolor='rgba(147,112,219,0.1)'
))
fig_cluster.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 10], tickfont=dict(color='#9370DB'), color='#9370DB'),
        angularaxis=dict(tickfont=dict(color='#9370DB'), direction="clockwise", color='#9370DB')
    ),
    showlegend=False,
    margin=dict(l=20, r=20, b=20, t=40),
)

# =======================
# RÉGRESSION - RADAR 2
# =======================
df_encoded = sm.add_constant(df_ref[independent_vars + dependent_vars].dropna())

models = {}
for dep_var in dependent_vars:
    if dep_var in df_encoded.columns:
        X = df_encoded[['const'] + independent_vars].astype(float)
        y = df_encoded[dep_var].astype(float)
        models[dep_var] = sm.OLS(y, X).fit()

user_data_reg = {**{col: user_df[col].iloc[0] for col in independent_vars}, 'Age': age_normalise}
user_df_regression = pd.DataFrame([user_data_reg])
user_df_regression = sm.add_constant(user_df_regression, has_constant='add')
user_df_regression = user_df_regression.reindex(columns=['const'] + independent_vars, fill_value=0)

predicted_scores = {}
for dep_var, model in models.items():
    cols_needed = list(model.params.index)
    X_user = user_df_regression.reindex(columns=cols_needed, fill_value=0).astype(float)
    predicted_scores[dep_var] = float(model.predict(X_user)[0])

min_score, max_score = min(predicted_scores.values()), max(predicted_scores.values())
normalized_scores = {k: (v - min_score) / (max_score - min_score) if max_score > min_score else 0.5
                     for k, v in predicted_scores.items()}

fig_scores = go.Figure()
fig_scores.add_trace(go.Scatterpolar(
    r=list(normalized_scores.values()) + [list(normalized_scores.values())[0]],
    theta=[var.replace('_', ' ') for var in normalized_scores.keys()] +
          [list(normalized_scores.keys())[0].replace('_', ' ')],
    fill='toself',
    name='Scores prédits',
    line_color='#9370DB',
    fillcolor='rgba(147,112,219,0.1)'
))
fig_scores.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='#9370DB'), color='#9370DB'),
        angularaxis=dict(tickfont=dict(color='#9370DB'), color='#9370DB')
    ),
    showlegend=False,
    margin=dict(l=20, r=20, b=20, t=40),
)

# =======================
# AFFICHAGE CÔTE À CÔTE
# =======================
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<p class="cluster-title">Cluster {user_cluster + 1}</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_cluster, use_container_width=True, height=400)
with col2:
    st.markdown(f'<p class="cluster-title">Scores de bien-être</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_scores, use_container_width=True, height=400)
