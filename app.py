
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# Style CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }

    .result-box {
        padding: 22px;
        border-radius: 12px;
        background-color: var(--secondary-background-color);
        border-left: 5px solid #14b8a6;
        margin-top: 12px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Chargement du modèle
model_path = Path(__file__).parent / "random_forest_model.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(
        "Modèle introuvable. Vérifiez la présence de "
        "random_forest_model.pkl dans le dossier du projet."
    )
    st.stop()
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()


# Segments métier
segments = {
    0: "Clients VIP / très fidèles",
    1: "Clients inactifs / à réactiver",
    2: "Clients réguliers"
}


# En-tête
st.markdown(
    '<div class="main-title">👥 Customer Segmentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyse RFM et prédiction du segment client avec Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# Présentation du projet
st.markdown("### 🎯 Prédire le segment d'un client")
st.write(
    "Saisissez les indicateurs RFM d'un client pour identifier "
    "son segment à l'aide du modèle Random Forest."
)


# Formulaire de saisie
with st.form("rfm_form"):

    st.markdown("#### Informations du client")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "📅 Recency (jours)",
            min_value=0,
            value=10,
            step=1,
            help="Nombre de jours depuis le dernier achat."
        )

    with col2:
        frequency = st.number_input(
            "🛒 Frequency (commandes)",
            min_value=1,
            value=5,
            step=1,
            help="Nombre de commandes du client."
        )

    with col3:
        monetary = st.number_input(
            "💰 Monetary (montant)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Montant total dépensé par le client."
        )

    submitted = st.form_submit_button(
        "🔍 Prédire le segment",
        use_container_width=True
    )


# Prédiction
if submitted:

    new_client = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    try:
        cluster = int(model.predict(new_client)[0])

        if cluster not in segments:
            st.error(f"Cluster {cluster} : aucun nom métier défini.")
            st.stop()

        segment = segments[cluster]

        st.divider()
        st.markdown("### 📊 Résultat de la prédiction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Cluster prédit", cluster)

        with col2:
            st.metric("Montant dépensé", f"{monetary:,.2f}")

        st.markdown(
            f"""
            <div class="result-box">
                <h4>Segment identifié</h4>
                <h2>{segment}</h2>
                <p>Le modèle Random Forest a attribué ce client
                au cluster {cluster}.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### 🧾 Récapitulatif des informations RFM")

        result_df = pd.DataFrame({
            "Indicateur": ["Recency", "Frequency", "Monetary"],
            "Valeur": [recency, frequency, monetary]
        })

        st.dataframe(
            result_df,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "La prédiction reproduit la segmentation apprise à partir "
            "des clusters K-Means."
        )

    except Exception as e:
        st.error(
            f"Erreur lors de la prédiction : {e}. "
            "Vérifiez que les variables correspondent à celles "
            "utilisées pendant l'entraînement."
        )


# Pied de page
st.divider()
st.caption(
    "Projet Machine Learning | RFM • K-Means • Random Forest • Streamlit"
)
