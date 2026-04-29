import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ==========================================
# MOTEUR PHYSIQUE : GÉORADAR (COMPLET)
# ==========================================
def physique_electromagnetique(eps_r, sigma_mS_m, f_MHz):
    """
    Calcule la vitesse, l'atténuation et la profondeur de peau
    basé sur les équations de Maxwell pour un milieu à pertes.
    """
    sigma = sigma_mS_m / 1000.0  # Conversion en S/m
    f = f_MHz * 1e6  # Conversion en Hz
    w = 2 * np.pi * f
    mu = 4 * np.pi * 1e-7  # Perméabilité du vide
    eps0 = 8.854e-12  # Permittivité du vide
    eps = eps_r * eps0

    # Constante d'atténuation (alpha) exacte en Neper/m
    term1 = (mu * eps) / 2.0
    term2 = np.sqrt(1 + (sigma / (w * eps)) ** 2) - 1
    alpha_neper = w * np.sqrt(term1 * term2)

    # Conversion en dB/m
    alpha_dB = 8.686 * alpha_neper

    # Profondeur de peau (Skin depth)
    skin_depth = 1.0 / alpha_neper if alpha_neper > 0 else float('inf')

    # Vitesse de phase exacte (m/s puis m/ns)
    term3 = np.sqrt(1 + (sigma / (w * eps)) ** 2) + 1
    beta = w * np.sqrt(term1 * term3)
    v_m_s = w / beta
    v_m_ns = v_m_s / 1e9

    return v_m_ns, alpha_dB, skin_depth


def temps_trajet_hyperbole(x_profil, x_cible, z_cible, vitesse):
    distance = np.sqrt((x_profil - x_cible) ** 2 + z_cible ** 2)
    return 2 * distance / vitesse


# ==========================================
# INTERFACE UTILISATEUR
# ==========================================
st.set_page_config(page_title="Module Géoradar", layout="wide")
st.title("Module Géoradar (GPR) : Atténuation & Cinématique")
st.markdown(
    "Évaluez la profondeur d'investigation face à la conductivité du sol, et modélisez l'empreinte cinématique des cibles.")

# --- BARRE LATÉRALE ---
st.sidebar.header("Paramètres du Milieu")
milieu = st.sidebar.selectbox(
    "Type de sol de référence",
    [
        ("Air (Vides)", 1.0, 0.0),
        ("Sable sec", 4.0, 0.01),
        ("Calcaire sec", 7.0, 1.0),
        ("Sable humide", 15.0, 5.0),
        ("Terre végétale", 20.0, 20.0),
        ("Argile humide", 25.0, 150.0),
        ("Eau douce", 80.0, 5.0)
    ],
    format_func=lambda x: f"{x[0]}"
)

# Overrides manuels pour peaufiner
eps_r = st.sidebar.number_input("Permittivité relative (εr)", min_value=1.0, max_value=81.0, value=float(milieu[1]),
                                step=1.0)
sigma_mS = st.sidebar.number_input("Conductivité (mS/m)", min_value=0.0, max_value=2000.0, value=float(milieu[2]),
                                   step=1.0)

st.sidebar.header("Paramètres GPR")
frequence_mhz = st.sidebar.select_slider("Fréquence centrale (MHz)", options=[100, 250, 400, 500, 800, 1000, 2000],
                                         value=500)
dynamic_range_db = st.sidebar.slider("Performances du radar (Plage dynamique en dB)", 50, 150, 100, 10)

st.sidebar.header("Cibles & Géométrie")
x_cible = st.sidebar.slider("Position X de la cible ponctuelle (m)", -10.0, 10.0, 0.0, 0.5)
z_cible = st.sidebar.slider("Profondeur de la cible ponctuelle (m)", 0.5, 10.0, 2.0, 0.1)

# --- CALCULS PHYSIQUES ---
vitesse, alpha_dB, skin_depth = physique_electromagnetique(eps_r, sigma_mS, frequence_mhz)
longueur_onde_m = vitesse / (frequence_mhz / 1000)
resolution_verticale_cm = (longueur_onde_m / 4) * 100

# Calcul de la profondeur maximum d'investigation
z_array = np.linspace(0.1, 20.0, 500)
# Perte totale = Atténuation matérielle (Aller-Retour) + Perte par divergence sphérique
perte_totale_db = 2 * alpha_dB * z_array + 20 * np.log10(2 * z_array)

try:
    index_max = np.where(perte_totale_db <= dynamic_range_db)[0][-1]
    z_max_investigation = z_array[index_max]
except IndexError:
    z_max_investigation = 0.0

# Cinématique (Cible ponctuelle)
x_profil = np.linspace(-15, 15, 500)
twt_hyperbole = temps_trajet_hyperbole(x_profil, x_cible, z_cible, vitesse)

# --- KPIs ---
st.subheader("Bilan Électromagnétique")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vitesse (v)", f"{vitesse:.3f} m/ns")
col2.metric("Atténuation (α)", f"{alpha_dB:.2f} dB/m")
col3.metric("Effet de peau (δ)", f"{skin_depth:.2f} m")
col4.metric("Limite d'Investigation", f"{z_max_investigation:.2f} m", delta="Max théorique", delta_color="off")

# --- GRAPHIQUES ---
tab1, tab2 = st.tabs(["Atténuation & Réflecteurs", "Cinématique (Radargramme)"])

with tab1:
    fig_att, (ax_att1, ax_att2) = plt.subplots(1, 2, figsize=(14, 5))

    # Graphe 1 : Chute du signal
    ax_att1.plot(perte_totale_db, z_array, color='red', linewidth=2, label="Atténuation totale")
    ax_att1.axvline(dynamic_range_db, color='black', linestyle='--', label="Seuil de détection radar")
    ax_att1.axhline(z_max_investigation, color='blue', linestyle=':', label="Profondeur Max")

    ax_att1.fill_betweenx(z_array, perte_totale_db, dynamic_range_db, where=(perte_totale_db > dynamic_range_db),
                          color='gray', alpha=0.3, label="Zone de bruit (Signal perdu)")

    ax_att1.set_ylim(max(10, z_max_investigation + 2), 0)
    ax_att1.set_xlim(0, dynamic_range_db + 20)
    ax_att1.set_xlabel("Perte de signal (dB)")
    ax_att1.set_ylabel("Profondeur (m)")
    ax_att1.set_title("Bilan de liaison et Profondeur d'investigation")
    ax_att1.grid(True, linestyle=':', alpha=0.7)
    ax_att1.legend()

    # Graphe 2 : Coupe Géologique avec réflecteurs plans
    ax_att2.axhspan(0, 20, facecolor='#d7ccc8', alpha=0.5)

    # Réflecteurs de test
    reflecteurs = [1.5, 3.0, z_max_investigation + 1.0]
    for r in reflecteurs:
        if r <= z_max_investigation:
            ax_att2.axhline(r, color='green', linewidth=3, label="Réflecteur DÉTECTABLE" if r == reflecteurs[0] else "")
        else:
            ax_att2.axhline(r, color='red', linewidth=3, alpha=0.3,
                            label="Réflecteur INVISIBLE" if r == reflecteurs[-1] else "")

    ax_att2.axhline(z_max_investigation, color='blue', linestyle=':', linewidth=2)
    ax_att2.fill_between([-15, 15], z_max_investigation, 20, color='black', alpha=0.4)
    ax_att2.text(0, z_max_investigation + 0.5, "SILENCE RADAR", color='white', ha='center', fontweight='bold')

    ax_att2.set_xlim(-15, 15)
    ax_att2.set_ylim(max(10, z_max_investigation + 2), -1)
    ax_att2.set_xlabel("Distance (m)")
    ax_att2.set_title("Visibilité des couches du sous-sol")
    ax_att2.legend(loc="lower right")

    st.pyplot(fig_att)

with tab2:
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.subheader("Radargramme Théorique (B-Scan)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor('#d3d3d3')
        ax.plot(x_profil, twt_hyperbole, color='black', linewidth=3, label="Écho de la cible")

        # Effacement de l'hyperbole si la cible est au-delà de la profondeur max
        if z_cible > z_max_investigation:
            ax.text(0, 100, "CIBLE INVISIBLE\n(Signal totalement atténué)", color='red', fontsize=14, ha='center',
                    weight='bold')
            ax.set_alpha(0.2)

        ax.set_xlabel("Distance sur le profil (m)")
        ax.set_ylabel("Temps de trajet aller-retour (ns)")
        ax.set_xlim(-15, 15)
        ax.set_ylim(max(200, temps_trajet_hyperbole(0, 0, z_max_investigation, vitesse)), 0)
        ax.grid(True, linestyle='--', alpha=0.5, color='white')
        ax.legend()
        st.pyplot(fig)

    with g_col2:
        st.subheader("Coupe du Sous-sol")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.axhspan(0, 15, facecolor='#d7ccc8', alpha=1)
        ax2.axhline(0, color='#5d4037', linewidth=3)

        # Ombre d'atténuation
        ax2.fill_between([-15, 15], z_max_investigation, 15, color='black', alpha=0.5)
        ax2.axhline(z_max_investigation, color='blue', linestyle=':', label="Limite de détection")

        circle = Circle((x_cible, z_cible), 0.3, facecolor='black', edgecolor='white', linewidth=1, zorder=3)
        ax2.add_patch(circle)

        ax2.set_xlim(-15, 15)
        ax2.set_ylim(max(10, z_max_investigation + 2), -2)
        ax2.set_aspect('equal')
        ax2.set_xlabel("Distance X (m)")
        ax2.set_ylabel("Profondeur réelle (m)")
        ax2.legend()
        st.pyplot(fig2)