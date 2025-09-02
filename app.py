import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Application de Santé Mentale",
    page_icon="🧠",
    layout="wide"
)

# Style personnalisé pour le dégradé de fond et les couleurs
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #E6E6FA, #FFFFFF);
    }
    .title {
        color: #9370DB;
        text-align: center;
        font-size: 4em !important;
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
    .tab {
        background-color: rgba(248, 248, 255, 0.8);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .tab h2, .tab p, .tab li, .tab ul {
        color: #6A5ACD !important;  /* Mauve foncé pour le texte */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre principal
st.markdown('<p class="title">Application de Santé Mentale</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Un espace dédié à votre bien-être mental</p>', unsafe_allow_html=True)

# Barre latérale pour la navigation
menu = st.sidebar.radio(
    "Navigation",
    ["Description", "Objectif"]
)

# Contenu des onglets
if menu == "Description":
    st.markdown(
        """
        <div class="tab">
            <h2>Description</h2>
            <p>
                La santé mentale est aujourd’hui une préoccupation majeure et légitime, touchant un nombre croissant d’individus à travers le monde.
                Pourtant, malgré cette prise de conscience collective, <strong>beaucoup de personnes hésitent encore à franchir le pas</strong> pour demander de l’aide ou entamer un parcours de soin.
                Les raisons sont multiples : méconnaissance des ressources disponibles, appréhension des démarches à suivre, peur du jugement, ou simplement le sentiment de ne pas savoir par où commencer.
            </p>
            <p>
                Notre projet s’inscrit dans cette dynamique en proposant une <strong>solution accessible, bienveillante et guidée</strong>.
                Nous souhaitons accompagner chaque utilisateur, à son rythme, en lui offrant :
            </p>
            <ul>
                <li><strong>Des suggestions personnalisées</strong> pour identifier ses besoins et ses émotions.</li>
                <li><strong>Des ressources claires et fiables</strong> pour mieux comprendre les enjeux de la santé mentale.</li>
                <li><strong>Un espace sécurisé</strong> pour explorer des pistes d’amélioration, sans pression ni jugement.</li>
            </ul>
            <p>
                Que vous soyez en quête d’outils pour gérer votre stress, améliorer votre sommeil, ou simplement mieux vous connaître,
                cette application est conçue pour <strong>vous orienter et vous soutenir</strong>, étape par étape, vers un mieux-être durable.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "Objectif":
    st.markdown(
        """
        <div class="tab">
            <h2>Objectif</h2>
            <p>
                Notre objectif est de vous fournir un espace sûr et bienveillant pour explorer votre santé mentale.
                Nous souhaitons vous aider à identifier vos besoins, à développer des stratégies d'adaptation,
                et à trouver un équilibre émotionnel durable.
            </p>
            <p>
                Grâce à des fonctionnalités simples et accessibles, nous espérons vous accompagner vers une meilleure connaissance
                de vous-même et une amélioration de votre qualité de vie.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Pied de page
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9370DB; font-size: 1.2em;'>© 2025 - Application de Santé Mentale</p>", unsafe_allow_html=True)
