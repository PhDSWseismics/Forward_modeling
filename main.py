import streamlit as st

st.set_page_config(
    page_title="Toolbox Géophysique",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Toolbox Géophysique : Forward Modelling")
st.markdown("---")

st.markdown("""
Bienvenue dans votre environnement de modélisation géophysique. 
Cette toolbox est conçue pour simuler les réponses théoriques du sous-sol face à différentes méthodes d'investigation.

### 🛠️ Méthodes disponibles
Utilisez le menu latéral à gauche pour naviguer entre les modules :

* **🧲 Gravimétrie & Aliasing :** Simulez l'anomalie de Bouguer générée par diverses géométries (sphère, cylindre, plan) et étudiez l'impact du pas d'échantillonnage et des incertitudes instrumentales/GPS.
* **📡 Géoradar (GPR) :** Modélisez la cinématique des ondes électromagnétiques (radargramme théorique) en fonction de la permittivité du milieu et de la fréquence de l'antenne.

*Développé pour l'optimisation et le cadrage des propositions techniques.*
""")