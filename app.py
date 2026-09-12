import gradio as gr
import joblib
import pandas as pd

# Chargement du modèle
model = joblib.load("modele_immobilier_xgb.joblib")

def estimer_prix(surface, chambres):
    donnees = pd.DataFrame([{"surface_m2": surface, "chambres": chambres}])
    prix_k = model.predict(donnees)[0]
    return f"{prix_k:.2f} k€ ({prix_k * 1000:,.0f} €)".replace(",", " ")

demo = gr.Interface(
    fn=estimer_prix,
    inputs=[
        gr.Slider(20, 200, step=5, value=70, label="Surface (m²)"),
        gr.Slider(1, 6, step=1, value=2, label="Nombre de chambres")
    ],
    outputs=gr.Textbox(label="Prix estimé"),
    title="🏡 Estimation Immobilière par IA",
    description="Application de prédiction de prix immobiliers basée sur un modèle XGBoost."
)

demo.launch()
