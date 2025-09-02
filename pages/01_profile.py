import streamlit as st
from datetime import datetime
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Mon Espace Personnel",
    page_icon="👤",
    layout="wide"
)

# Style personnalisé pour la charte graphiques
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #E6E6FA, #FFFFFF);
    }
    .main-title {
        color: #9370DB;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
        font-weight: bold;
    }
    .subtitle {
        color: #6A5ACD;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 0;
    }
    .form-container {
        background-color: rgba(248, 248, 255, 0.9);
        border-radius: 15px;
        padding: 25px;
        margin: 20px auto;
        max-width: 600px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > label,
    .stDateInput > div > label {
        color: #6A5ACD !important;
        font-weight: bold;
    }
    .stTextInput > div > div > input,
    .stDateInput > div > div > input {
        border: 1px solid #6A5ACD !important;
    }
    .stButton > button {
        background-color: transparent;
        color: #6A5ACD;
        border: 2px solid #6A5ACD;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #6A5ACD;
        color: white;
    }
    .success-message {
        color: #6A5ACD;
        font-size: 1.2em;
        text-align: center;
        font-weight: bold;
    }
    .warning-message {
        color: #FF6347;
        font-size: 1.1em;
        text-align: center;
    }
    .next-button-container {
        display: flex;
        justify-content: flex-end;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre principal
st.markdown('<p class="main-title">Mon Espace Personnel</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Complétez votre profil pour accéder à des ressources personnalisées</p>', unsafe_allow_html=True)

# Conteneur pour le formulaire
st.markdown('<div class="form-container">', unsafe_allow_html=True)

# Champ pour le nom
nom = st.text_input("Nom", placeholder="Entrez votre nom")

# Champ pour le prénom
prenom = st.text_input("Prénom", placeholder="Entrez votre prénom")

# Sélecteur de date de naissance
date_naissance = st.date_input(
    "Date de Naissance",
    min_value=datetime(1940, 1, 1),
    max_value=datetime(2007, 12, 31),
    value=None,
    format="DD/MM/YYYY"
)

# Bouton pour valider le formulaire
if st.button("Enregistrer mes informations"):
    if nom and prenom and date_naissance:
        date_formatee = date_naissance.strftime("%d/%m/%Y")
        st.markdown(f'<p class="success-message">Bonjour {prenom} {nom}, né(e) le {date_formatee} !<br>Votre profil a été enregistré avec succès.</p>', unsafe_allow_html=True)
        # Stocker les informations de profil dans la session
        st.session_state.profile_info = {
            'nom': nom,
            'prenom': prenom,
            'date_naissance': date_naissance
        }
    else:
        st.markdown('<p class="warning-message">Veuillez remplir tous les champs pour continuer.</p>', unsafe_allow_html=True)


# Initialisation du DataFrame pour les réponses
if 'reponses_df' not in st.session_state:
    st.session_state.reponses_df = pd.DataFrame(columns=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"], index=[0])
    st.session_state.reponses_df.loc[0] = [None, None, None, None, None, None, None]  # Une seule ligne avec des valeurs None



# Bouton "Accéder au questionnaire" en bas à droite
st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
if st.button("Accéder au questionnaire"):
    st.switch_page("pages/question_1.py")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown('<p style="text-align: center; color: #6A5ACD; font-size: 1em;">© 2025 - Application de Bien-être Mental</p>', unsafe_allow_html=True)
