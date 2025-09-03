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
    .cluster-title { color:#9370DB; font-size:1.4em; font-weight:bold; text-align:center; margin:10px 0; }
    .interpretation { color:#6A5ACD; font-size:1.1em; text-align:center; margin:10px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">Vos Résultats</p>', unsafe_allow_html=True)

# =======================
# CONFIG / MAPPINGS
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
# CHECK session_state
# =======================
if 'reponses_df' not in st.session_state:
    st.info("Aucune réponse trouvée. Répondez d'abord au questionnaire.")
    st.stop()

# =======================
# CHARGER RÉFÉRENCES
# =======================
df_ref = pd.read_csv("df_clusters.csv")

scaler_path, model_path = 'scaler.save', 'kmeans_model.save'

# Charger ou (re)former scaler et kmeans
if os.path.exists(scaler_path) and os.path.exists(model_path):
    scaler_ref = joblib.load(scaler_path)
    kmeans = joblib.load(model_path)
else:
    scaler_ref = StandardScaler()
    X_ref_cont = df_ref[continuous_cols].astype(float)
    X_ref_cont_scaled = scaler_ref.fit_transform(X_ref_cont)
    X_ref_bin = df_ref[binary_cols].values
    X_ref_scaled = np.hstack((X_ref_cont_scaled, X_ref_bin))
    kmeans = KMeans(n_clusters=5, random_state=42).fit(X_ref_scaled)
    joblib.dump(scaler_ref, scaler_path)
    joblib.dump(kmeans, model_path)

# Si scaler existe mais n'a pas d'attribut mean_ (pas fitted) -> refit sur df_ref
if not hasattr(scaler_ref, "mean_"):
    scaler_ref = StandardScaler().fit(df_ref[continuous_cols].astype(float))
    joblib.dump(scaler_ref, scaler_path)

# Obtenir la liste de features que le scaler attend (fallback = continuous_cols)
scaler_features = list(getattr(scaler_ref, "feature_names_in_", continuous_cols))

# =======================
# PREPARER L'UTILISATEUR
# =======================
# calcul âge normalisé
if 'profile_info' in st.session_state and 'date_naissance' in st.session_state.profile_info:
    date_naissance = st.session_state.profile_info['date_naissance']
    age = datetime.now().year - date_naissance.year - (
        (datetime.now().month, datetime.now().day) < (date_naissance.month, date_naissance.day)
    )
else:
    age = 25
age_normalise = int(((age - 18) / (99 - 18)) * 9) + 1

# construire dict utilisateur (valeurs scalaires)
user_data = {col: 0 for col in continuous_cols + binary_cols}
user_data['Age'] = age_normalise

# remplir depuis les réponses (sécurité si une question manque)
row = st.session_state.reponses_df.iloc[0]
for q in row.index:
    if q in question_mapping:
        val = row[q]
        if pd.isna(val):
            continue
        if question_mapping[q]['inverse']:
            val = 10 - val
        user_data[question_mapping[q]['variable']] = val

# Q6 => binaire
if 'Q6' in st.session_state.reponses_df.columns:
    q6val = st.session_state.reponses_df.iloc[0].get('Q6', np.nan)
    if not pd.isna(q6val):
        user_data['Family_History_Mental_Illness'] = 1 if q6val <= 5 else 0

# DataFrame utilisateur (scalars -> 1 ligne)
user_df = pd.DataFrame([user_data])

# =======================
# SCALER ALIGN + TRANSFORM (sécurisé)
# =======================
# Réindexer / ordonner les colonnes continues selon scaler_features (remplit manquantes par 0)
user_continuous_df = user_df.reindex(columns=scaler_features, fill_value=0).astype(float).apply(pd.to_numeric, errors='coerce').fillna(0)

# Essayer de transformer ; si erreur, refit un scaler sur df_ref[sclaer_features] puis transformer
try:
    user_continuous_scaled = scaler_ref.transform(user_continuous_df)
except Exception:
    # fallback : refit sur df_ref
    try:
        ref_cont = df_ref.reindex(columns=scaler_features)[scaler_features].astype(float)
        scaler_ref = StandardScaler().fit(ref_cont)
        joblib.dump(scaler_ref, scaler_path)
        user_continuous_scaled = scaler_ref.transform(user_continuous_df)
    except Exception as e:
        st.error("Erreur lors du scaling des variables continues : " + str(e))
        st.stop()

# Binaire (réindexer + numeric)
user_binary_df = user_df.reindex(columns=binary_cols, fill_value=0).astype(float).apply(pd.to_numeric, errors='coerce').fillna(0)
user_binary = user_binary_df.values

# Construire vecteur complet pour kmeans (même ordre que X_ref_scaled utilisé à l'entraînement)
user_data_scaled = np.hstack((user_continuous_scaled, user_binary))

# =======================
# CLUSTERING
# =======================
try:
    user_cluster = int(kmeans.predict(user_data_scaled.reshape(1, -1))[0])
except Exception as e:
    st.error("Erreur lors de la prédiction du cluster : " + str(e))
    st.stop()

# =======================
# RADAR 1 (CLUSTER) - valeurs non-scalées pour lisibilité
# =======================
plot_features = continuous_cols + binary_cols
user_plot_values = user_df.reindex(columns=plot_features, fill_value=0).iloc[0].tolist()
user_plot_values.append(user_plot_values[0])

feature_labels = {
    'Age': 'Âge', 'Sleep_Hours': 'Sommeil',
    'Social_Support_Score': 'Soutien social', 'Financial_Stress': 'Stress financier',
    'Work_Stress': 'Stress travail', 'Self_Esteem_Score': 'Estime de soi',
    'Family_History_Mental_Illness': 'Antécédents familiaux', 'Loneliness_Score': 'Solitude'
}
features_display = [feature_labels[f] for f in plot_features]
features_display.append(features_display[0])

fig_cluster = go.Figure()
fig_cluster.add_trace(go.Scatterpolar(
    r=user_plot_values,
    theta=features_display,
    fill='toself',
    name='Vos valeurs',
    line_color='#9370DB',
    fillcolor='rgba(147,112,219,0.12)'
))
fig_cluster.update_layout(
    polar=dict(
        radialaxis=dict(range=[0, 10], tickfont=dict(color='#9370DB'), color='#9370DB', gridcolor='#E6E6FA'),
        angularaxis=dict(tickfont=dict(color='#9370DB'), color='#9370DB', direction='clockwise')
    ),
    showlegend=False,
    margin=dict(l=20, r=20, b=20, t=40),
    height=420
)

# =======================
# RÉGRESSION : préparation X/Y propre (dropna maîtrisé)
# =======================
# Construire X_ref et y_ref en retirant les lignes incomplètes (sur toutes les colonnes concernées)
tmp = pd.concat([df_ref[independent_vars], df_ref[dependent_vars]], axis=1).dropna()
if tmp.shape[0] < 5:
    st.warning("Jeu de référence trop petit après suppression des NaN pour construire les modèles de régression.")
# X et y
X_ref = tmp[independent_vars].astype(float)
X_ref_const = sm.add_constant(X_ref, has_constant='add')
models = {}
for dep in dependent_vars:
    if dep in tmp.columns:
        y_ref = tmp[dep].astype(float)
        try:
            models[dep] = sm.OLS(y_ref, X_ref_const).fit()
        except Exception:
            # skip si erreur
            pass

if not models:
    st.warning("Aucun modèle de régression valide n'a pu être estimé à partir des données de référence.")
# =======================
# PREPARER user_df pour la régression (aligné & ordonné)
# =======================
user_df_reg = user_df.reindex(columns=independent_vars, fill_value=0).astype(float).apply(pd.to_numeric, errors='coerce').fillna(0)
user_df_reg_const = sm.add_constant(user_df_reg, has_constant='add')
user_df_reg_const = user_df_reg_const.reindex(columns=['const'] + independent_vars, fill_value=0)

# Prédictions (alignement sur model.params.index)
predicted_scores = {}
for dep_var, model in models.items():
    cols_needed = list(model.params.index)
    X_user = user_df_reg_const.reindex(columns=cols_needed, fill_value=0).astype(float)
    try:
        predicted_scores[dep_var] = float(model.predict(X_user)[0])
    except Exception:
        predicted_scores[dep_var] = 0.0

if predicted_scores:
    min_score = min(predicted_scores.values())
    max_score = max(predicted_scores.values())
    if max_score == min_score:
        normalized_scores = {k: 0.5 for k in predicted_scores}
    else:
        normalized_scores = {k: (v - min_score) / (max_score - min_score) for k, v in predicted_scores.items()}
else:
    normalized_scores = {v: 0.0 for v in dependent_vars}

# =======================
# RADAR 2 (SCORES PRÉDITS)
# =======================
theta_scores = [k.replace('_', ' ') for k in normalized_scores.keys()]
r_scores = list(normalized_scores.values())
if r_scores:
    r_scores.append(r_scores[0])
    theta_scores.append(theta_scores[0])

fig_scores = go.Figure()
fig_scores.add_trace(go.Scatterpolar(
    r=r_scores,
    theta=theta_scores,
    fill='toself',
    name='Scores prédits',
    line_color='#9370DB',
    fillcolor='rgba(147,112,219,0.12)'
))
fig_scores.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='#9370DB'), color='#9370DB'),
        angularaxis=dict(tickfont=dict(color='#9370DB'), color='#9370DB')
    ),
    showlegend=False,
    margin=dict(l=20, r=20, b=20, t=40),
    height=420
)

# =======================
# AFFICHAGE CÔTE À CÔTE
# =======================
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<p class="cluster-title">Cluster {user_cluster + 1}</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_cluster, use_container_width=True)
with col2:
    st.markdown(f'<p class="cluster-title">Scores de bien-être</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_scores, use_container_width=True)

# Interprétation simple
interpretations = {
    0: "Votre profil indique un bien-être général élevé.",
    1: "Votre profil indique un bien-être moyen avec quelques points à améliorer.",
    2: "Votre profil indique un niveau de stress modéré.",
    3: "Votre profil indique des signes de fatigue ou d'anxiété.",
    4: "Votre profil indique un besoin d'attention particulière pour votre bien-être mental."
}
st.markdown(f'<div class="interpretation">{interpretations.get(user_cluster, "Interprétation non disponible")}</div>', unsafe_allow_html=True)
