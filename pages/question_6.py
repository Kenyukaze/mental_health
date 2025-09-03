import streamlit as st

# Style CSS (inchangé)
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
        padding: 40px;
        margin: 20px auto;
        max-width: 800px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #E6E6FA;
        text-align: center;
    }
    .question-text {
        color: #6A5ACD;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .stButton > button {
        border: 2px solid #9370DB;
        border-radius: 10px;
        padding: 20px 30px;
        font-weight: bold;
        font-size: 1.2em;
        width: 200px;
        height: 60px;
        margin: 10px auto;
        transition: all 0.3s ease;
        display: block;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        color: #9370DB;
    }
    .next-btn {
        background-color: #9370DB;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 20px 30px;
        font-weight: bold;
        font-size: 1em;
        transition: all 0.3s ease;
        display: block;
        margin: 30px auto 0;
    }
    .next-btn:hover {
        background-color: #6A5ACD;
        transform: translateY(-2px);
        color: #9370DB;
    }
    .next-btn:disabled {
        background-color: #D8BFD8;
        cursor: not-allowed;
        transform: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre principal
st.markdown('<p class="main-title">Antécédents familiaux</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Question 6/7</p>', unsafe_allow_html=True)

# Conteneur de la question
st.markdown('<div class="question-container">', unsafe_allow_html=True)
st.markdown('<p class="question-text">Avez-vous des antécédents familiaux de troubles de la santé mentale ?</p>', unsafe_allow_html=True)

# Initialiser la variable de réponse si elle n'existe pas
if "reponse_q6" not in st.session_state:
    st.session_state.reponse_q6 = None

# Appliquer le style du bouton sélectionné **avant** de dessiner les boutons
if st.session_state.reponse_q6 == 1:
    st.markdown("<style>#q6_vrai button {background-color:#9370DB !important; color:white !important;}</style>", unsafe_allow_html=True)
elif st.session_state.reponse_q6 == 0:
    st.markdown("<style>#q6_faux button {background-color:#9370DB !important; color:white !important;}</style>", unsafe_allow_html=True)

# Boutons exclusifs Vrai / Faux
c_sp, c_vrai, c_faux, c_sp2 = st.columns([1, 1, 1, 1])
with c_vrai:
    if st.button("Oui", key="q6_vrai", use_container_width=True):
        st.session_state.reponse_q6 = 1
        st.rerun()  # Forcer la réexécution pour appliquer le style
with c_faux:
    if st.button("Non", key="q6_faux", use_container_width=True):
        st.session_state.reponse_q6 = 0
        st.rerun()  # Forcer la réexécution pour appliquer le style

# Stocker la réponse dans le DataFrame **uniquement si elle existe**
if st.session_state.reponse_q6 is not None:
    st.session_state.reponses_df.at[0, "Q6"] = st.session_state.reponse_q6

# Fermeture du conteneur
st.markdown('</div>', unsafe_allow_html=True)

# Bouton pour passer à la question suivante
if st.button("Question suivante", disabled=st.session_state.reponse_q6 is None, key="next_btn"):
    try:
        st.switch_page("pages/question_7.py")
    except AttributeError:
        st.info("⚠️ Votre version de Streamlit ne supporte pas `st.switch_page`. Mettez à jour avec : `pip install --upgrade streamlit`")

# Pied de page
st.markdown("---")
st.markdown('<p style="text-align: center; color: #9370DB; font-size: 0.9em; margin-top: 40px;">© 2025 - Application de Santé Mentale</p>', unsafe_allow_html=True)
