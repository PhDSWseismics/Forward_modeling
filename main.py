import streamlit as st

st.set_page_config(
    page_title="Toolbox Géophysique",
    layout="centered"
)

st.title("Toolbox Géophysique : Forward Modelling")
st.markdown("---")

st.markdown("""
Cette application permet de modéliser les réponses théoriques du sous-sol pour différentes méthodes de prospection géophysique. L'objectif est de quantifier l'impact des paramètres d'acquisition et des propriétés physiques du milieu pour optimiser le dimensionnement et le cadrage des campagnes de terrain.

### Méthodes implémentées

**1. Gravimétrie**
La méthode gravimétrique repose sur la mesure des variations du champ de pesanteur terrestre causées par les contrastes de densité du sous-sol. Ce module modélise l'anomalie de Bouguer générée par différentes géométries (sphérique, cylindrique, tabulaire) et permet d'évaluer l'impact du pas d'échantillonnage spatial (aliasing) ainsi que la propagation des incertitudes instrumentales et altimétriques (GPS).

**2. Géoradar (GPR - Ground Penetrating Radar)**
Le géoradar exploite la propagation et la réflexion d'ondes électromagnétiques à haute fréquence. Ce module propose d'évaluer la profondeur d'investigation théorique en fonction de l'atténuation du signal (liée à la conductivité et la permittivité diélectrique du milieu), et d'anticiper l'empreinte cinématique (radargramme) et la résolution verticale selon la fréquence de l'antenne sélectionnée.
""")