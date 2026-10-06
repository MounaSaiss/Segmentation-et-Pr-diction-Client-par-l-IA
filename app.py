import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("random_forest_model.pkl")

# Configuration de la page
st.title("👤 Segmentation des clients")

st.write("Entrez les informations RFM du client :")

# Données RFM
recency = st.number_input(
    "Recency",
    min_value=0,
    value=10
)

frequency = st.number_input(
    "Frequency",
    min_value=1,
    value=5
)

monetary = st.number_input(
    "Monetary",
    min_value=0.0,
    value=1000.0
)

# Segments
segments = {
    0: "Clients VIP / très fidèles",
    1: "Clients inactifs / à réactiver",
    2: "Clients réguliers"
}

# Prédiction
if st.button("Prédire le segment"):

    new_client = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    cluster = model.predict(new_client)[0]
    segment = segments[cluster]

    st.success(f"Cluster prédit : {cluster}")

    st.subheader("Segment du client")
    st.write(segment)

    st.subheader("Informations RFM")
    st.dataframe(new_client)
    
    
    
    
    
    
# streamlit run app.py