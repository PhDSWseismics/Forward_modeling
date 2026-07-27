import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon
import io


# ==========================================
# MOTEUR PHYSIQUE : GRAVIMÉTRIE
# ==========================================
G = 6.67430e-11
FACTOR_UGAL = 1e8  # conversion m/s^2 -> µGal


def anomalie_sphere(x, z, size, dRho):
    r = size / 2.0
    mass = (4 / 3) * np.pi * (r ** 3) * dRho
    return FACTOR_UGAL * (G * mass * z) / ((x ** 2 + z ** 2) ** 1.5)


def anomalie_cylindre_horizontal(x, z, size, dRho):
    r = size / 2.0
    mass_lin = np.pi * (r ** 2) * dRho
    return FACTOR_UGAL * (2 * G * mass_lin * z) / (x ** 2 + z ** 2)


def anomalie_plan(x, z, size, dRho, longueur_plan):
    L_demi = longueur_plan / 2.0
    return FACTOR_UGAL * 2 * G * dRho * size * (
        np.arctan((x + L_demi) / z) - np.arctan((x - L_demi) / z)
    )


def anomalie_cheminee_verticale(x, z_top, longueur, diametre, dRho):
    """
    Approximation "ligne de masse" : cheminée karstique/aven modélisé comme un
    empilement de masses ponctuelles le long d'un axe vertical (valable quand
    le diamètre est petit devant la profondeur).
    Intégrale fermée de G*pi*r^2*dRho*z/(x^2+z^2)^1.5 entre z_top et z_base.
    """
    r = diametre / 2.0
    z_base = z_top + longueur
    lin_mass = np.pi * (r ** 2) * dRho
    contrib = 1.0 / np.sqrt(x ** 2 + z_top ** 2) - 1.0 / np.sqrt(x ** 2 + z_base ** 2)
    return FACTOR_UGAL * G * lin_mass * contrib


def anomalie_chapelet(x, positions, profondeurs, tailles, dRho):
    """Réseau karstique : somme de cavités sphériques alignées le long du profil."""
    total = np.zeros_like(x)
    for x0, z0, s0 in zip(positions, profondeurs, tailles):
        total += anomalie_sphere(x - x0, z0, s0, dRho)
    return total


def calculer_anomalie(x_array, forme, params):
    dRho = params["dRho"]
    if forme == "sphère":
        return anomalie_sphere(x_array, params["z_depth"], params["size"], dRho)
    elif forme == "cylindre horizontal":
        return anomalie_cylindre_horizontal(x_array, params["z_depth"], params["size"], dRho)
    elif forme == "cheminée verticale (aven)":
        return anomalie_cheminee_verticale(x_array, params["z_top"], params["longueur"], params["size"], dRho)
    elif forme == "plan (couche)":
        return anomalie_plan(x_array, params["z_depth"], params["size"], dRho, params["longueur_plan"])
    elif forme == "chapelet karstique":
        return anomalie_chapelet(x_array, params["positions"], params["profondeurs"], params["tailles"], dRho)
    return np.zeros_like(x_array)


# ==========================================
# INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Module Gravimétrie Maxxée", layout="wide")

st.title("Module Gravimétrie Maxxée — Karst")

st.markdown(
    "Version étendue du module gravimétrie : géométries karstiques (aven, chapelet de cavités), "
    "préréglages de remplissage, garde-fou de dimensionnement du profil et test de détectabilité (SNR).")

# --- BARRE LATÉRALE : CIBLE ---
st.sidebar.header("Contexte karstique")
preset = st.sidebar.selectbox(
    "Préréglage de remplissage de la cavité",
    ["Personnalisé", "Cavité vide (air)", "Cavité noyée (eau)", "Cavité comblée (argile)", "Cavité comblée (sable)"],
)
preset_dRho = {
    "Cavité vide (air)": -2700.0,
    "Cavité noyée (eau)": -1700.0,
    "Cavité comblée (argile)": -900.0,
    "Cavité comblée (sable)": -800.0,
}

st.sidebar.header("Paramètres de l'anomalie")
forme = st.sidebar.selectbox(
    "Forme de la cible",
    ["sphère", "cylindre horizontal", "cheminée verticale (aven)", "plan (couche)", "chapelet karstique"],
    format_func=lambda x: {
        "sphère": "Sphère (grotte isolée)",
        "cylindre horizontal": "Cylindre horizontal (galerie infinie)",
        "cheminée verticale (aven)": "Cheminée verticale (aven / cheminée d'effondrement)",
        "plan (couche)": "Plan (couche horizontale)",
        "chapelet karstique": "Chapelet karstique (réseau de cavités)",
    }[x],
)

if preset != "Personnalisé":
    default_dRho = preset_dRho[preset]
else:
    default_dRho = -2700.0

density_contrast = st.sidebar.slider("Contraste de densité (kg/m³)", -3000.0, 3000.0, default_dRho, 50.0)

params = {"dRho": density_contrast}
longueur_plan = 10.0

if forme in ["sphère", "cylindre horizontal", "plan (couche)"]:
    z_depth = st.sidebar.slider("Profondeur du centre (m)", 1.0, 30.0, 4.0, 0.5)
    size = st.sidebar.slider("Épaisseur / Diamètre (m)", 0.5, 10.0, 1.0, 0.1)
    params["z_depth"] = z_depth
    params["size"] = size
    if forme == "plan (couche)":
        longueur_plan = st.sidebar.slider("Longueur du plan (m)", 1.0, 50.0, 10.0, 1.0)
        params["longueur_plan"] = longueur_plan
    if size >= (z_depth * 2) and forme in ["sphère", "cylindre horizontal"]:
        st.sidebar.warning("⚠️ Attention : l'anomalie affleure ou dépasse la surface.")
    profondeur_bas = z_depth + size / 2.0

elif forme == "cheminée verticale (aven)":
    z_top = st.sidebar.slider("Profondeur du sommet (m)", 0.5, 25.0, 2.0, 0.5)
    longueur_cheminee = st.sidebar.slider("Longueur de la cheminée (m)", 1.0, 20.0, 8.0, 0.5)
    diametre = st.sidebar.slider("Diamètre (m)", 0.3, 5.0, 1.0, 0.1)
    params["z_top"] = z_top
    params["longueur"] = longueur_cheminee
    params["size"] = diametre
    profondeur_bas = z_top + longueur_cheminee
    z_depth = z_top  # pour les avertissements génériques

elif forme == "chapelet karstique":
    st.sidebar.subheader("Réseau de cavités")
    n_cavites = st.sidebar.slider("Nombre de cavités", 2, 6, 4, 1)
    espacement_moyen = st.sidebar.slider("Espacement moyen (m)", 1.0, 10.0, 4.0, 0.5)
    profondeur_moyenne = st.sidebar.slider("Profondeur moyenne (m)", 1.0, 30.0, 6.0, 0.5)
    variabilite_prof = st.sidebar.slider("Variabilité de profondeur (± m)", 0.0, 10.0, 2.0, 0.5)
    taille_moyenne = st.sidebar.slider("Taille moyenne des cavités (m)", 0.5, 6.0, 1.5, 0.1)
    seed_chapelet = st.sidebar.number_input("Graine aléatoire (géométrie)", min_value=0, value=1, step=1)

    rng_geo = np.random.default_rng(int(seed_chapelet))
    positions = (np.arange(n_cavites) - (n_cavites - 1) / 2.0) * espacement_moyen
    positions = positions + rng_geo.uniform(-espacement_moyen * 0.15, espacement_moyen * 0.15, n_cavites)
    profondeurs = profondeur_moyenne + rng_geo.uniform(-variabilite_prof, variabilite_prof, n_cavites)
    profondeurs = np.clip(profondeurs, 0.5, None)
    tailles = taille_moyenne + rng_geo.uniform(-taille_moyenne * 0.3, taille_moyenne * 0.3, n_cavites)
    tailles = np.clip(tailles, 0.3, None)

    params["positions"] = positions
    params["profondeurs"] = profondeurs
    params["tailles"] = tailles

    z_depth = np.mean(profondeurs)
    profondeur_bas = np.max(profondeurs + tailles / 2.0)

st.sidebar.header("Paramètres d'acquisition")
x_min = st.sidebar.number_input("Profil min (m)", value=-15.0)
x_max = st.sidebar.number_input("Profil max (m)", value=15.0)
esp = st.sidebar.slider("Espacement des stations (m)", 1.0, 10.0, 5.0, 0.5)
dec = st.sidebar.slider("Décalage de la grille (m)", 0.0, float(esp), 0.0, 0.1)

if (x_max - x_min) < 6 * z_depth:
    st.sidebar.warning(
        "⚠️ Profil probablement trop court par rapport à la profondeur de la cible : "
        "l'anomalie risque d'être tronquée en bord de grille (règle empirique : longueur ≥ 6× la profondeur)."
    )

st.sidebar.header("Incertitudes & Précision")
incertitude_gps_cm = st.sidebar.number_input("Incertitude altimétrique GPS (cm)", min_value=0.0, value=2.0, step=0.5)
incertitude_gravi_ugal = st.sidebar.number_input("Précision du gravimètre (µGal)", min_value=0.0, value=5.0, step=1.0)

fag_ugal_per_cm = 3.086
erreur_altimetrique_ugal = incertitude_gps_cm * fag_ugal_per_cm
erreur_totale_ugal = np.sqrt(incertitude_gravi_ugal ** 2 + erreur_altimetrique_ugal ** 2)

st.sidebar.info(f"**Incertitude totale : ±{erreur_totale_ugal:.1f} µGal**")

# --- CALCUL DES DONNÉES ---
x_continu = np.linspace(x_min, x_max, 1000)
y_continu = calculer_anomalie(x_continu, forme, params)

start_x = x_min + (dec % esp)
x_mesure = np.arange(start_x, x_max + 0.001, esp)
y_mesure = calculer_anomalie(x_mesure, forme, params)

# Calcul de la "Pire Courbe"
if len(y_mesure) > 0:
    amplitude_max = np.max(np.abs(y_mesure))
    poids = np.abs(y_mesure) / amplitude_max if amplitude_max > 0 else np.zeros_like(y_mesure)

    if density_contrast < 0:
        y_pire = y_mesure + erreur_totale_ugal * poids - erreur_totale_ugal * (1 - poids)
    else:
        y_pire = y_mesure - erreur_totale_ugal * poids + erreur_totale_ugal * (1 - poids)
else:
    y_pire = np.array([])

# --- SNR / DÉTECTABILITÉ ---
amplitude_theo_max = np.max(np.abs(y_continu)) if len(y_continu) > 0 else 0.0
snr = amplitude_theo_max / erreur_totale_ugal if erreur_totale_ugal > 0 else float("inf")
if snr >= 5:
    verdict_snr, couleur_snr = "Détectable", "green"
elif snr >= 2:
    verdict_snr, couleur_snr = "Marginal", "orange"
else:
    verdict_snr, couleur_snr = "Non détectable / noyé dans le bruit", "red"

# --- ZONE D'AFFICHAGE ET POP-UP THÉORIQUE ---
col_title, col_help = st.columns([0.85, 0.15])

with col_help:
    with st.popover("📖 Résumé Théorique"):
        st.markdown("### Fondamentaux du Module")
        st.subheader("1. Cheminée verticale (aven)")
        st.write(
            "Approximée par une ligne de masses ponctuelles entre le sommet z₁ et la base z₂ "
            "(valable si le diamètre est petit devant la profondeur) :")
        st.latex(r"\Delta g(x) = G \pi r^2 \Delta\rho \left(\frac{1}{\sqrt{x^2+z_1^2}} - \frac{1}{\sqrt{x^2+z_2^2}}\right)")
        st.subheader("2. Chapelet karstique")
        st.write("Somme linéaire des contributions de N cavités sphériques indépendantes le long du profil.")
        st.subheader("3. Détectabilité (SNR)")
        st.latex(r"SNR = \frac{|\Delta g_{max}|}{\sigma_{total}}")
        st.write("SNR ≥ 5 : détectable · 2 ≤ SNR < 5 : marginal · SNR < 2 : noyé dans le bruit.")
        st.subheader("4. Incertitude totale")
        st.latex(r"E_{total} = \sqrt{E_{gravi}^2 + (E_{gps\_cm} \times 3.086)^2}")

# --- AFFICHAGE DES RÉSULTATS (KPIs) ---
max_theo = np.min(y_continu) if density_contrast < 0 else np.max(y_continu)
if density_contrast < 0:
    max_mes = np.min(y_mesure) if len(y_mesure) > 0 else 0
else:
    max_mes = np.max(y_mesure) if len(y_mesure) > 0 else 0

col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
col_stat1.metric("Amplitude Théorique Max", f"{abs(max_theo):.2f} µGal")
col_stat2.metric("Amplitude Mesurée Max", f"{abs(max_mes):.2f} µGal",
                 delta=f"{abs(max_mes) - abs(max_theo):.2f} µGal", delta_color="inverse")
col_stat3.metric("Bruit de mesure (±)", f"{erreur_totale_ugal:.2f} µGal")
col_stat4.markdown(f"**SNR = {snr:.1f}**" if snr != float("inf") else "**SNR = ∞**")
col_stat4.markdown(f":{couleur_snr}[{verdict_snr}]")

# --- GRAPHIQUES ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Signal Gravimétrique")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_continu, y_continu, label="Signal Réel (Continu)", color="#3498db", linewidth=2)
    ax.errorbar(x_mesure, y_mesure, yerr=erreur_totale_ugal, label="Mesures Terrain ± Erreur",
                color="#e67e22", fmt="o", linestyle="--", linewidth=1.5, markersize=6, capsize=4)

    if len(y_pire) > 0:
        ax.plot(x_mesure, y_pire, label="Pire scénario (Signal aplati)", color="red", linestyle="-.", linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.fill_between(x_continu, -erreur_totale_ugal, erreur_totale_ugal, color="gray", alpha=0.1,
                     label="Seuil de bruit")
    ax.set_xlabel("Distance X (m)")
    ax.set_ylabel("Anomalie (µGal)")
    ax.set_xlim(x_min, x_max)
    if density_contrast < 0:
        ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper right", fontsize="small")

    st.pyplot(fig)

    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Signal", data=buf1.getvalue(), file_name="gravi_signal.png",
                       mime="image/png", use_container_width=True)

with col2:
    st.subheader("Coupe du Sous-sol")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    profondeur_max_affichage = max(15, profondeur_bas + 2)
    ax2.axhspan(-2, 0, facecolor="#e0f7fa", alpha=1)
    ax2.axhspan(0, profondeur_max_affichage, facecolor="#d7ccc8", alpha=1)
    ax2.axhline(0, color="#5d4037", linewidth=3)
    color_target = "#ffffff" if density_contrast < 0 else "#2c3e50"

    if forme in ["sphère", "cylindre horizontal"]:
        circle = Circle((0, params["z_depth"]), params["size"] / 2, facecolor=color_target, edgecolor="black",
                         linewidth=2, zorder=3)
        ax2.add_patch(circle)
    elif forme == "plan (couche)":
        x_rect = -longueur_plan / 2.0
        rect = Rectangle((x_rect, params["z_depth"] - params["size"] / 2), longueur_plan, params["size"],
                          facecolor=color_target, edgecolor="black", linewidth=2, zorder=3)
        ax2.add_patch(rect)
    elif forme == "cheminée verticale (aven)":
        ellipse = Ellipse((0, (params["z_top"] + profondeur_bas) / 2), params["size"], params["longueur"],
                           facecolor=color_target, edgecolor="black", linewidth=2, zorder=3)
        ax2.add_patch(ellipse)
    elif forme == "chapelet karstique":
        for x0, z0, s0 in zip(params["positions"], params["profondeurs"], params["tailles"]):
            circle = Circle((x0, z0), s0 / 2, facecolor=color_target, edgecolor="black", linewidth=2, zorder=3)
            ax2.add_patch(circle)

    taille_marqueur_x = (x_max - x_min) * 0.02
    taille_marqueur_y = profondeur_max_affichage * 0.05
    for x_m in x_mesure:
        triangle = Polygon(
            [[x_m, 0], [x_m - taille_marqueur_x, -taille_marqueur_y], [x_m + taille_marqueur_x, -taille_marqueur_y]],
            closed=True, facecolor="#e67e22", zorder=4)
        ax2.add_patch(triangle)
        ax2.plot([x_m, x_m], [0, profondeur_max_affichage], color="#e67e22", linestyle="--", linewidth=1, alpha=0.5)

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(profondeur_max_affichage, -2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Distance X (m)")
    ax2.set_ylabel("Profondeur (m)")

    st.pyplot(fig2)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Coupe 2D", data=buf2.getvalue(), file_name="gravi_coupe.png",
                       mime="image/png", use_container_width=True)
