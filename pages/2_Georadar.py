import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io


# ==========================================
# MOTEUR PHYSIQUE : GÉORADAR (COMPLET)
# ==========================================
def physique_electromagnetique(eps_r, sigma_mS_m, f_MHz):
    sigma = sigma_mS_m / 1000.0
    f = f_MHz * 1e6
    w = 2 * np.pi * f
    mu = 4 * np.pi * 1e-7
    eps0 = 8.854e-12
    eps = eps_r * eps0

    term1 = (mu * eps) / 2.0
    term2 = np.sqrt(1 + (sigma / (w * eps)) ** 2) - 1
    alpha_neper = w * np.sqrt(term1 * term2)

    alpha_dB = 8.686 * alpha_neper
    skin_depth = 1.0 / alpha_neper if alpha_neper > 0 else float('inf')

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

z_array = np.linspace(0.1, 20.0, 500)
perte_totale_db = 2 * alpha_dB * z_array + 20 * np.log10(2 * z_array)

try:
    index_max = np.where(perte_totale_db <= dynamic_range_db)[0][-1]
    z_max_investigation = z_array[index_max]
except IndexError:
    z_max_investigation = 0.0

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

    ax_att1.plot(perte_totale_db, z_array, color='red', linewidth=2, label="Atténuation totale")
    ax_att1.axvline(dynamic_range_db, color='black', linestyle='--', label="Seuil de détection radar")
    ax_att1.axhline(z_max_investigation, color='blue', linestyle=':', label="Profondeur Max")
    ax_att1.fill_betweenx(z_array, perte_totale_db, dynamic_range_db, where=(perte_totale_db > dynamic_range_db),
                          color='gray', alpha=0.3, label="Zone de bruit")

    ax_att1.set_ylim(max(10, z_max_investigation + 2), 0)
    ax_att1.set_xlim(0, dynamic_range_db + 20)
    ax_att1.set_xlabel("Perte de signal (dB)")
    ax_att1.set_ylabel("Profondeur (m)")
    ax_att1.set_title("Bilan de liaison")
    ax_att1.grid(True, linestyle=':', alpha=0.7)
    ax_att1.legend()

    ax_att2.axhspan(0, 20, facecolor='#d7ccc8', alpha=0.5)
    reflecteurs = [1.5, 3.0, z_max_investigation + 1.0]
    for r in reflecteurs:
        if r <= z_max_investigation:
            ax_att2.axhline(r, color='green', linewidth=3, label="DÉTECTABLE" if r == reflecteurs[0] else "")
        else:
            ax_att2.axhline(r, color='red', linewidth=3, alpha=0.3, label="INVISIBLE" if r == reflecteurs[-1] else "")

    ax_att2.axhline(z_max_investigation, color='blue', linestyle=':', linewidth=2)
    ax_att2.fill_between([-15, 15], z_max_investigation, 20, color='black', alpha=0.4)
    ax_att2.text(0, z_max_investigation + 0.5, "SILENCE RADAR", color='white', ha='center', fontweight='bold')

    ax_att2.set_xlim(-15, 15)
    ax_att2.set_ylim(max(10, z_max_investigation + 2), -1)
    ax_att2.set_xlabel("Distance (m)")
    ax_att2.set_title("Couches du sous-sol")
    ax_att2.legend(loc="lower right")

    st.pyplot(fig_att)

    # BOUTON SNAPSHOT GPR TAB 1
    buf_gpr1 = io.BytesIO()
    fig_att.savefig(buf_gpr1, format="png", dpi=300, bbox_inches='tight')
    st.download_button(label="📸 Snapshot Bilan d'Atténuation", data=buf_gpr1.getvalue(),
                       file_name="gpr_attenuation.png", mime="image/png")

with tab2:
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.subheader("Radargramme Théorique (B-Scan)")
        fig_rad, ax_rad = plt.subplots(figsize=(8, 5))
        ax_rad.set_facecolor('#d3d3d3')
        ax_rad.plot(x_profil, twt_hyperbole, color='black', linewidth=3, label="Écho de la cible")

        if z_cible > z_max_investigation:
            ax_rad.text(0, 100, "CIBLE INVISIBLE", color='red', fontsize=14, ha='center', weight='bold')
            ax_rad.set_alpha(0.2)

        ax_rad.set_xlabel("Distance sur le profil (m)")
        ax_rad.set_ylabel("Temps aller-retour (ns)")
        ax_rad.set_xlim(-15, 15)
        ax_rad.set_ylim(max(200, temps_trajet_hyperbole(0, 0, z_max_investigation, vitesse)), 0)
        ax_rad.grid(True, linestyle='--', alpha=0.5, color='white')
        ax_rad.legend()

        st.pyplot(fig_rad)

        # BOUTON SNAPSHOT GPR TAB 2 (Radargramme)
        buf_rad = io.BytesIO()
        fig_rad.savefig(buf_rad, format="png", dpi=300, bbox_inches='tight')
        st.download_button(label="📸 Snapshot Radargramme", data=buf_rad.getvalue(), file_name="gpr_radargramme.png",
                           mime="image/png", use_container_width=True)

    with g_col2:
        st.subheader("Coupe du Sous-sol")
        fig_sub, ax_sub = plt.subplots(figsize=(8, 5))
        ax_sub.axhspan(0, 15, facecolor='#d7ccc8', alpha=1)
        ax_sub.axhline(0, color='#5d4037', linewidth=3)

        ax_sub.fill_between([-15, 15], z_max_investigation, 15, color='black', alpha=0.5)
        ax_sub.axhline(z_max_investigation, color='blue', linestyle=':', label="Limite de détection")

        circle = Circle((x_cible, z_cible), 0.3, facecolor='black', edgecolor='white', linewidth=1, zorder=3)
        ax_sub.add_patch(circle)

        ax_sub.set_xlim(-15, 15)
        ax_sub.set_ylim(max(10, z_max_investigation + 2), -2)
        ax_sub.set_aspect('equal')
        ax_sub.set_xlabel("Distance X (m)")
        ax_sub.set_ylabel("Profondeur réelle (m)")
        ax_sub.legend()

        st.pyplot(fig_sub)

        # BOUTON SNAPSHOT GPR TAB 2 (Coupe)
        buf_sub = io.BytesIO()
        fig_sub.savefig(buf_sub, format="png", dpi=300, bbox_inches='tight')
        st.download_button(label="📸 Snapshot Coupe", data=buf_sub.getvalue(), file_name="gpr_coupe.png",
                           mime="image/png", use_container_width=True)