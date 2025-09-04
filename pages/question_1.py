import streamlit as st
import pandas as pd

# Style CSS personnalisé
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
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .subtitle {
        color: #D8BFD8;
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 1.5em;
    }
    .question-container {
        background-color: rgba(248, 248, 255, 0.8);
        border-radius: 15px;
        padding: 30px;
        margin: 20px auto;
        max-width: 800px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #E6E6FA;
    }
    .question-text {
        color: #6A5ACD;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .question-description {
        color: #6A5ACD;
        font-size: 1.1em;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
    }

    /* Style pour le slider */
    .stSlider > div > div > div > div {
        background-color: #E6E6FA !important;
        height: 8px !important;
        border-radius: 4px !important;
    }

    /* Style pour le curseur */
    .stSlider > div > div > div > div > div {
        background: #000000 !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        box-shadow: 0 0 0 3px #FFFFFF !important;
        top: -7px !important;
    }

    /* Style pour les valeurs min/max */
    .stSlider > div > div > div {
        padding: 15px 0 !important;
    }


    .stSlider .st-ai {
        color: #6A5ACD !important;
    }

    .stButton > button {
        background-color: #9370DB;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 1em;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #6A5ACD;
        transform: translateY(-2px);
    }

    .next-button-container {
        display: flex;
        justify-content: flex-end;
        margin-top: 30px;
    }

    /* Style pour la piste du slider */
    .stSlider > div > div > div > div::before {
        top: 25px !important;
    }

    .stSlider > div > div > div > div::after {
        top: 25px !important;
    }
    </style>

    <!-- CSS supplémentaire pour les labels -->
    <style>
    .slider-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        padding: 0 15px;
    }
    .slider-label {
        color: #6A5ACD;
        font-weight: bold;
        font-size: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre principal de la page
st.markdown('<p class="main-title">Évaluation de votre bien-être</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Question 1/7</p>', unsafe_allow_html=True)

# Conteneur de la question
st.markdown('''<div class="question-container">''', unsafe_allow_html=True)

# Titre de la question
st.markdown('<p class="question-text">Sur une échelle de 1 à 10, quel est votre niveau de stress lié à votre environnement professionnel au cours du dernier mois ?</p>', unsafe_allow_html=True)

# Description optionnelle
st.markdown('<p class="question-description">Utilisez le curseur ci-dessous pour indiquer votre niveau de stress</p>', unsafe_allow_html=True)

# Affichage des valeurs min/max
st.markdown('''<div class="slider-labels">
    <span class="slider-label">0<br><small>Pas du tout</small></span>
    <span class="slider-label">10<br><small>Extrêmement</small></span>
</div>''', unsafe_allow_html=True)

# Slider pour la réponse
reponse = st.slider("", 0, 10, 5, key="q1", label_visibility="collapsed")

# Enregistrement de la réponse
st.session_state.reponses_df.at[0, "Q1"] = reponse
st.session_state.reponses_df.at[0, "Q1"] = reponse
# Fermeture du conteneur
st.markdown('</div>', unsafe_allow_html=True)

# Bouton pour passer à la question suivante
st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
if st.button("Question suivante"):
    st.switch_page("pages/question_2.py")
st.markdown('</div>', unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown('<p style="text-align: center; color: #9370DB; font-size: 0.9em; margin-top: 40px;">© 2025 - Application de Santé Mentale</p>', unsafe_allow_html=True)
