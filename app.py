import joblib
import pandas as pd
import streamlit as st

# verification google
<meta name="google-site-verification" content="sJU8JxhkUDwAyO9eYWW036oyCEh_qTgFjKE7lX2L6FQ" />
unsafe_allow_html=true

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Estimation Immobilière", page_icon="🏡", layout="centered"
)

# Titre et description
st.title("🏡 Estimation de Prix Immobilier")
st.write(
    "Ajustez les caractéristiques du bien ci-dessous pour obtenir une estimation immédiate issue du modèle IA (XGBoost)."
)


# Chargement du modèle avec mise en cache
@st.cache_resource
def load_model():
    return joblib.load("modele_immobilier_xgb.joblib")


model = load_model()

# Formulaire d'entrée des caractéristiques
st.subheader("Caractéristiques du bien")

col1, col2 = st.columns(2)

with col1:
    surface = st.slider(
        "Surface habitable (m²)", min_value=20, max_value=250, value=70, step=5
    )

with col2:
    chambres = st.slider(
        "Nombre de chambres", min_value=1, max_value=6, value=2, step=1
    )

# Bouton d'estimation
if st.button("Estimer le prix", type="primary", use_container_width=True):
    # Préparation des données pour le modèle
    input_data = pd.DataFrame([{"surface_m2": surface, "chambres": chambres}])

    # Prédiction
    prediction_k = model.predict(input_data)[0]
    prediction_eur = prediction_k * 1000

    # Affichage du résultat
    st.markdown("---")
    st.success("### Estimation calculée avec succès !")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Prix estimé (k€)", value=f"{prediction_k:.2f} k€")
    with col_res2:
        st.metric(
            label="Prix estimé (Euros)", value=f"{prediction_eur:,.0f} €"
        )
