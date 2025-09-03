import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import joblib
import os
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Style CSS
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(to bottom, #E6E6FA, #FFFFFF); }
    .main-title { color: #9370DB; text-align: center; font-size: 2.5em; margin-bottom: 0.5em; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .result-container { background-color: rgba(248,248,255,0.9); border-radius: 15px; padding: 30px; margin: 30px auto; max-width: 800px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border: 1px solid #E6E6FA; text-align:center; }
    .cluster-title { color:#9370DB; font-size:1.8em; font-weight:bold; margin-bottom:20px; }
    .cluster-text { color:#6A5ACD; font-size:1.2em; line-height:1.6; margin-bottom:20px; }
    .interpretation { color:#6A5ACD; font-size:1.3em; font-weight:500; margin-top:20px; padding:20px; background-color: rgba(147,112,219,0.1); border-radius:10px; border-left:4px solid #9370DB; }
    .cluster-image { text-align:center; margin-top:40px; margin-bottom:40px; }
    .cluster-image img { width:700px; height:auto; display:block; margin-left:auto; margin-right:auto; }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre principal
st.markdown('<p class="main-title">Vos Résultats</p>', unsafe_allow_html=True)

# Mapping questions -> variables
question_mapping = {
    'Q1': {'variable': 'Work_Stress', 'inverse': False},
    'Q2': {'variable': 'Sleep_Hours', 'inverse': True},
    'Q3': {'variable': 'Social_Support_Score', 'inverse': True},
    'Q4': {'variable': 'Financial_Stress', 'inverse': False},
    'Q5': {'variable': 'Self_Esteem_Score', 'inverse': True},
    'Q6': {'variable': 'Family_History_Mental_Illness', 'inverse': True},
    'Q7': {'variable': 'Loneliness_Score', 'inverse': False}
}

# Affichage résultats si réponses existantes
if 'reponses_df' in st.session_state:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)

    # Charger données référence et modèles
    df_ref = pd.read_csv("df_clusters.csv")
    continuous_cols = ['Age','Sleep_Hours','Social_Support_Score','Financial_Stress','Work_Stress','Self_Esteem_Score','Loneliness_Score']
    binary_cols = ['Family_History_Mental_Illness']
    scaler_path = 'scaler.save'
    model_path = 'kmeans_model.save'

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        scaler_ref = joblib.load(scaler_path)
        kmeans = joblib.load(model_path)
    else:
        scaler_ref = StandardScaler()
        X_ref_continuous = df_ref[continuous_cols]
        X_ref_continuous_scaled = scaler_ref.fit_transform(X_ref_continuous)
        X_ref_binary = df_ref[binary_cols].values
        X_ref_scaled = np.hstack((X_ref_continuous_scaled, X_ref_binary))
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(X_ref_scaled)
        joblib.dump(scaler_ref, scaler_path)
        joblib.dump(kmeans, model_path)

    # Calcul âge utilisateur
    if 'profile_info' in st.session_state:
        date_naissance = st.session_state.profile_info['date_naissance']
        age = datetime.now().year - date_naissance.year - ((datetime.now().month, datetime.now().day) < (date_naissance.month, date_naissance.day))
    else:
        age = 25
    age_normalise = int(((age-18)/(99-18))*9)+1

    # Préparer données utilisateur
    user_data = {col: [0] for col in continuous_cols + binary_cols}
    user_data['Age'] = [age_normalise]

    for q, response in st.session_state.reponses_df.iloc[0].items():
        if q in question_mapping:
            val = response
            if question_mapping[q]['inverse']:
                val = 10 - val
            user_data[question_mapping[q]['variable']] = [val]

    # Corriger la logique pour Family_History_Mental_Illness
    if 'Q6' in st.session_state.reponses_df.columns:
        user_data['Family_History_Mental_Illness'] = [1 if st.session_state.reponses_df.iloc[0]['Q6'] <= 5 else 0]

    user_df = pd.DataFrame(user_data)[continuous_cols + binary_cols]

    # Debug: Afficher les données avant scaling
    st.subheader("🔍 Debugging")
    st.write("User data avant scaling :", user_df)

    # Vérifier l'ordre des colonnes
    st.write("Colonnes du scaler :", scaler_ref.feature_names_in_)
    st.write("Colonnes de user_df avant réorganisation :", user_df[continuous_cols].columns.tolist())

    # Réorganiser les colonnes de user_df pour correspondre à celles du scaler
    user_df[continuous_cols] = user_df[continuous_cols][scaler_ref.feature_names_in_]

    # Scaling des colonnes continues uniquement
    user_continuous_scaled = scaler_ref.transform(user_df[continuous_cols])
    user_binary = user_df[binary_cols].values
    user_data_scaled = np.hstack((user_continuous_scaled, user_binary))

    # Debug: Vérifier les shapes
    st.write("Shape user_continuous_scaled:", user_continuous_scaled.shape)
    st.write("Shape user_binary:", user_binary.shape)
    st.write("Shape user_data_scaled:", user_data_scaled.shape)
    st.write("Nb colonnes attendues:", len(continuous_cols + binary_cols))

    # Création du DataFrame
    df_scaled = pd.DataFrame(user_data_scaled, columns=continuous_cols + binary_cols)
    st.write("User data après scaling :", df_scaled)

    # Cluster utilisateur
    user_cluster = kmeans.predict(user_data_scaled.reshape(1, -1))[0]
    st.write("Cluster prédit :", user_cluster)

    # Clustering sur df_ref
    X_ref_continuous_scaled = scaler_ref.transform(df_ref[continuous_cols])
    X_ref_scaled = np.hstack((X_ref_continuous_scaled, df_ref[binary_cols].values))
    df_ref['cluster'] = kmeans.predict(X_ref_scaled)
    st.write("Répartition clusters df_ref :", df_ref['cluster'].value_counts())

    st.markdown(f'<p class="cluster-title">Vous appartenez au groupe : {user_cluster+1}</p>', unsafe_allow_html=True)

    # Interprétation clusters
    interpretations = {
        0: "Votre profil indique un bien-être général élevé.",
        1: "Votre profil indique un bien-être moyen avec quelques points à améliorer.",
        2: "Votre profil indique un niveau de stress modéré.",
        3: "Votre profil indique des signes de fatigue ou d'anxiété.",
        4: "Votre profil indique un besoin d'attention particulière pour votre bien-être mental."
    }
    st.markdown(f'<div class="interpretation">{interpretations.get(user_cluster,"Interprétation non disponible")}</div>', unsafe_allow_html=True)

    # Radar Chart
    features = continuous_cols + binary_cols
    user_values = user_df.iloc[0].values.tolist()
    user_values.append(user_values[0])
    feature_labels = {
        'Age':'Âge','Sleep_Hours':'Heures de sommeil','Social_Support_Score':'Soutien social',
        'Financial_Stress':'Stress financier','Work_Stress':'Stress au travail','Self_Esteem_Score':'Estime de soi',
        'Family_History_Mental_Illness':'Antécédents familiaux','Loneliness_Score':'Sentiment de solitude'
    }
    features_display = [feature_labels[f] for f in features]
    features_display.append(features_display[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=features_display,
        fill='toself',
        name='Vos valeurs',
        line_color='#9370DB',
        fillcolor='rgba(147,112,219,0.1)',
        hovertemplate='%{theta}: %{r}<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,10], tickfont=dict(color='#6A5ACD'), gridcolor='#E6E6FA'),
                   angularaxis=dict(direction='clockwise', tickfont=dict(color='#6A5ACD')),
                   bgcolor='rgba(0,0,0,0)'),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50,r=50,b=50,t=50),
        title=dict(text="Radar Chart de vos indicateurs", font=dict(size=14,color='#6A5ACD'),x=0.38)
    )
    st.plotly_chart(fig, use_container_width=True)

    # PCA clusters
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_ref_scaled)
    df_ref['pca1'] = pca_result[:,0]
    df_ref['pca2'] = pca_result[:,1]
    user_pca = pca.transform(user_data_scaled.reshape(1, -1))

    fig_pca = px.scatter(df_ref, x="pca1", y="pca2",
                         color=df_ref['cluster'].astype(str),
                         title="Visualisation des clusters (PCA 2D)",
                         opacity=0.6, width=700, height=500,
                         color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pca.add_scatter(x=[user_pca[0,0]], y=[user_pca[0,1]], mode="markers",
                        marker=dict(size=15, color="red", symbol="x"), name="Utilisateur")
    st.plotly_chart(fig_pca, use_container_width=True)

    # Image cluster
    cluster_images = {0:"Cluster_1.png", 1:"Cluster_2.png", 2:"Cluster_3.png", 3:"Cluster_4.png", 4:"Cluster_5.png"}
    script_dir = os.path.dirname(__file__)
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    image_filename = os.path.join(images_dir, cluster_images.get(user_cluster, 'Cluster_1.png'))

    if os.path.exists(image_filename):
        st.markdown('<div class="cluster-image">', unsafe_allow_html=True)
        st.image(image_filename)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"L'image {image_filename} est introuvable.")
        st.write("Chemin images:", images_dir)
        st.write("Fichiers disponibles :", os.listdir(images_dir) if os.path.exists(images_dir) else "Dossier introuvable")
        st.write("Dossier courant :", os.listdir(os.path.dirname(script_dir)))

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color:#6A5ACD;font-size:1.2em;text-align:center;">Aucune réponse enregistrée.</p>', unsafe_allow_html=True)
