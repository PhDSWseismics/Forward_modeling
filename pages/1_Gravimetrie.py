import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon


# ==========================================
# MOTEUR PHYSIQUE : GRAVIMÉTRIE
# ==========================================
def calculer_anomalie(x_array, forme, depth, size, dRho):
    """
    Calcule l'anomalie de gravité selon la géométrie.
    Retourne la valeur en µGal.
    """
    G = 6.67430e-11
    factor = 1e8  # Conversion m/s^2 vers µGal (10^5 pour mGal, 10^8 pour µGal)

    radius = size / 2.0

    if forme == 'sphère':
        mass = (4 / 3) * np.pi * (radius ** 3) * dRho
        gz = factor * (G * mass * depth) / ((x_array ** 2 + depth ** 2) ** 1.5)
    elif forme == 'cylindre':
        massLin = np.pi * (radius ** 2) * dRho
        gz = factor * (2 * G * massLin * depth) / (x_array ** 2 + depth ** 2)
    elif forme == 'plan':
        # Couche horizontale de 10m de large (x de -5 à 5), size = épaisseur
        gz = factor * 2 * G * dRho * size * (np.arctan((x_array + 5) / depth) - np.arctan((x_array - 5) / depth))
    else:
        gz = np.zeros_like(x_array)

    return gz


# ==========================================
# INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Module Gravimétrie", layout="wide")

st.title("Module Gravimétrie & Aliasing")

st.markdown(
    "Ajustez les paramètres pour observer l'impact du pas de mesure (aliasing) et la réponse de différentes géométries sur le signal gravimétrique.")

# --- BARRE LATÉRALE : PARAMÈTRES ---
st.sidebar.header("Paramètres de l'anomalie")
forme = st.sidebar.selectbox(
    "Forme de la cible",
    ['sphère', 'cylindre', 'plan'],
    format_func=lambda x: "Sphère (Grotte isolée)" if x == 'sphère' else (
        "Cylindre (Galerie infinie)" if x == 'cylindre' else "Plan (Couche horizontale 10m large)")
)

z_depth = st.sidebar.slider("Profondeur du centre (m)", 1.0, 15.0, 4.0, 0.5)
size = st.sidebar.slider("Épaisseur / Diamètre (m)", 0.5, 10.0, 1.0, 0.1)
density_contrast = st.sidebar.slider("Contraste de densité (kg/m³)", -3000.0, 3000.0, -2700.0, 50.0)

if size >= (z_depth * 2) and forme in ['sphère', 'cylindre']:
    st.sidebar.warning("⚠️ Attention : l'anomalie affleure ou dépasse la surface.")

st.sidebar.header("Paramètres d'acquisition")
x_min = st.sidebar.number_input("Profil min (m)", value=-15.0)
x_max = st.sidebar.number_input("Profil max (m)", value=15.0)
esp = st.sidebar.slider("Espacement des stations (m)", 1.0, 10.0, 5.0, 0.5)
dec = st.sidebar.slider("Décalage de la grille (m)", 0.0, float(esp), 0.0, 0.1)

st.sidebar.header("Incertitudes & Précision")
incertitude_gps_cm = st.sidebar.number_input("Incertitude altimétrique GPS (cm)", min_value=0.0, value=2.0, step=0.5)
incertitude_gravi_ugal = st.sidebar.number_input("Précision du gravimètre (µGal)", min_value=0.0, value=5.0, step=1.0)

# Calcul de l'incertitude combinée
fag_ugal_per_cm = 3.086
erreur_altimetrique_ugal = incertitude_gps_cm * fag_ugal_per_cm
erreur_totale_ugal = np.sqrt(incertitude_gravi_ugal ** 2 + erreur_altimetrique_ugal ** 2)

st.sidebar.info(f"**Incertitude totale : ±{erreur_totale_ugal:.1f} µGal**")

# --- CALCUL DES DONNÉES ---
x_continu = np.linspace(x_min, x_max, 1000)
y_continu = calculer_anomalie(x_continu, forme, z_depth, size, density_contrast)

start_x = x_min + (dec % esp)
x_mesure = np.arange(start_x, x_max + 0.001, esp)
y_mesure = calculer_anomalie(x_mesure, forme, z_depth, size, density_contrast)

# Calcul de la "Pire Courbe" (Scénario le plus plat possible dans les barres d'erreur)
if len(y_mesure) > 0:
    amplitude_max = np.max(np.abs(y_mesure))
    # Poids allant de 0 (sur les bords) à 1 (au centre de l'anomalie)
    poids = np.abs(y_mesure) / amplitude_max if amplitude_max > 0 else np.zeros_like(y_mesure)

    if density_contrast < 0:
        # Anomalie négative : on remonte le pic (vers 0) et on descend les bords
        y_pire = y_mesure + erreur_totale_ugal * poids - erreur_totale_ugal * (1 - poids)
    else:
        # Anomalie positive : on descend le pic (vers 0) et on remonte les bords
        y_pire = y_mesure - erreur_totale_ugal * poids + erreur_totale_ugal * (1 - poids)
else:
    y_pire = np.array([])

# --- ZONE D'AFFICHAGE ET POP-UP THÉORIQUE ---
col_title, col_help = st.columns([0.85, 0.15])

with col_help:
    with st.popover("📖 Résumé Théorique"):
        st.markdown("### Fondamentaux du Module")

        st.subheader("1. L'impact de l'altitude")
        st.write(
            "En gravimétrie, l'altitude est souvent la plus grande source d'erreur. Le gradient à l'air libre (Free Air Gradient) est d'environ 0.3086 mGal/m, ce qui signifie qu'une erreur d'altitude d'un seul centimètre engendre une erreur d'environ 3.086 µGal.")

        st.subheader("2. Calcul de l'incertitude totale")
        st.write(
            "Puisque l'erreur de l'instrument et l'erreur du GPS sont indépendantes, la méthode classique consiste à les combiner via la racine carrée de la somme des carrés (Root-Sum-Square) :")
        st.latex(r"E_{total} = \sqrt{E_{gravi}^2 + (E_{gps\_cm} \times 3.086)^2}")

        st.subheader("3. Modélisation de la cible (Sphère)")
        st.write(
            "Pour la modélisation gravimétrique analytique d'une sphère, la formule de l'anomalie de la composante verticale est régie par :")
        st.latex(
            r"\Delta g = G \cdot \frac{\frac{4}{3} \pi R^3 \cdot \Delta \rho \cdot z}{(x^2 + z^2)^{3/2}} \cdot 10^8")
        st.info("Note : Le facteur 10⁸ est utilisé ici pour obtenir un résultat directement en µGal.")

# --- AFFICHAGE DES RÉSULTATS (KPIs) ---
max_theo = np.min(y_continu) if density_contrast < 0 else np.max(y_continu)
if density_contrast < 0:
    max_mes = np.min(y_mesure) if len(y_mesure) > 0 else 0
else:
    max_mes = np.max(y_mesure) if len(y_mesure) > 0 else 0

col_stat1, col_stat2, col_stat3 = st.columns(3)
col_stat1.metric("Amplitude Théorique Max", f"{abs(max_theo):.2f} µGal")
col_stat2.metric("Amplitude Mesurée Max", f"{abs(max_mes):.2f} µGal",
                 delta=f"{abs(max_mes) - abs(max_theo):.2f} µGal", delta_color="inverse")
col_stat3.metric("Bruit de mesure (±)", f"{erreur_totale_ugal:.2f} µGal")

# --- GRAPHIQUES ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Signal Gravimétrique")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Courbes
    ax.plot(x_continu, y_continu, label='Signal Réel (Continu)', color='#3498db', linewidth=2)
    ax.errorbar(x_mesure, y_mesure, yerr=erreur_totale_ugal, label='Mesures Terrain ± Erreur',
                color='#e67e22', fmt='o', linestyle='--', linewidth=1.5, markersize=6, capsize=4)

    # Ajout de la Pire Courbe
    if len(y_pire) > 0:
        ax.plot(x_mesure, y_pire, label='Pire scénario (Signal aplati)', color='red', linestyle='-.', linewidth=1.5)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.fill_between(x_continu, -erreur_totale_ugal, erreur_totale_ugal, color='gray', alpha=0.1, label="Seuil de bruit")

    ax.set_xlabel("Distance X (m)")
    ax.set_ylabel("Anomalie (µGal)")
    ax.set_xlim(x_min, x_max)

    if density_contrast < 0:
        ax.invert_yaxis()

    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc="upper right", fontsize="small")
    st.pyplot(fig)

with col2:
    st.subheader("Coupe du Sous-sol")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    profondeur_max_affichage = max(15, z_depth + size + 2)
    ax2.axhspan(-2, 0, facecolor='#e0f7fa', alpha=1)
    ax2.axhspan(0, profondeur_max_affichage, facecolor='#d7ccc8', alpha=1)
    ax2.axhline(0, color='#5d4037', linewidth=3)
    color_target = '#ffffff' if density_contrast < 0 else '#2c3e50'

    if forme in ['sphère', 'cylindre']:
        circle = Circle((0, z_depth), size / 2, facecolor=color_target, edgecolor='black', linewidth=2, zorder=3)
        ax2.add_patch(circle)
    elif forme == 'plan':
        rect = Rectangle((-5, z_depth - size / 2), 10, size, facecolor=color_target, edgecolor='black', linewidth=2,
                         zorder=3)
        ax2.add_patch(rect)

    taille_marqueur_x = (x_max - x_min) * 0.02
    taille_marqueur_y = profondeur_max_affichage * 0.05
    for x_m in x_mesure:
        triangle = Polygon(
            [[x_m, 0], [x_m - taille_marqueur_x, -taille_marqueur_y], [x_m + taille_marqueur_x, -taille_marqueur_y]],
            closed=True, facecolor='#e67e22', zorder=4)
        ax2.add_patch(triangle)
        ax2.plot([x_m, x_m], [0, profondeur_max_affichage], color='#e67e22', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(profondeur_max_affichage, -2)
    ax2.set_aspect('equal')
    ax2.set_xlabel("Distance X (m)")
    ax2.set_ylabel("Profondeur (m)")
    st.pyplot(fig2)