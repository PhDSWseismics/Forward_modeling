import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io


# ==========================================
# MOTEUR PHYSIQUE : MASW
# ==========================================
def calculer_parametres_masw(n_geophones, dx, vr, f_coupure, critere_prudence, facteur_profondeur):
    L = (n_geophones - 1) * dx
    lambda_min = 2 * dx
    lambda_max_array = 0.5 * L if critere_prudence == "Prudent (λmax = 0.5·L)" else L

    f_max_aliasing = vr / lambda_min
    f_min_array = vr / lambda_max_array
    f_min_effectif = max(f_min_array, f_coupure)
    lambda_max_effectif = vr / f_min_effectif

    z_min = facteur_profondeur * lambda_min
    z_max = facteur_profondeur * lambda_max_effectif

    limitant = "Géophone (fréquence de coupure)" if f_coupure > f_min_array else "Dispositif (longueur du profil)"

    return {
        "L": L,
        "lambda_min": lambda_min,
        "lambda_max_array": lambda_max_array,
        "lambda_max_effectif": lambda_max_effectif,
        "f_max_aliasing": f_max_aliasing,
        "f_min_array": f_min_array,
        "f_min_effectif": f_min_effectif,
        "z_min": z_min,
        "z_max": z_max,
        "limitant": limitant,
    }


def epaisseur_min_resolvable(z, vr, facteur_profondeur, diviseur_resolution):
    """Résolution verticale approchée : à la profondeur z, la longueur d'onde de Rayleigh
    dominante qui échantillonne cette profondeur vaut lambda(z) = z / facteur_profondeur.
    Une transition de vitesse, une couche mince ou un vide n'est perceptible sur la courbe
    de dispersion que s'il représente une fraction significative de cette longueur d'onde
    (critère classique lambda/n, n=2 à 4 selon la littérature MASW)."""
    lam = np.maximum(z, 1e-6) / facteur_profondeur
    return lam / diviseur_resolution


# ==========================================
# INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Module MASW", layout="wide")

st.title("Module MASW (Multichannel Analysis of Surface Waves)")

st.markdown(
    "Dimensionnez un dispositif MASW avant chantier : bande de fréquences réellement exploitable "
    "sur la courbe de dispersion, profondeur d'investigation atteignable, et épaisseur minimale "
    "de couche / transition de vitesse / vide résolvable en fonction du nombre de géophones, "
    "de l'espacement et de la vitesse de Rayleigh du site.")

# --- BARRE LATÉRALE : DISPOSITIF ---
st.sidebar.header("Dispositif d'acquisition")
n_geophones = st.sidebar.slider("Nombre de géophones", 12, 96, 48, 1)
dx = st.sidebar.slider("Espacement inter-géophones Δx (m)", 0.5, 10.0, 1.0, 0.25)

st.sidebar.header("Milieu (vitesse de l'onde de Rayleigh)")
milieu = st.sidebar.selectbox(
    "Type de sol de référence",
    [
        ("Remblai / sol meuble récent", 150.0),
        ("Sable ou argile lâche", 200.0),
        ("Sable ou argile compact(e)", 300.0),
        ("Grave / alluvions denses", 450.0),
        ("Roche altérée / marne", 600.0),
        ("Roche saine (calcaire, granite)", 1000.0),
    ],
    index=2,
    format_func=lambda x: x[0],
)
vr = st.sidebar.number_input("Vitesse de Rayleigh estimée Vr (m/s)", min_value=50.0, max_value=2500.0,
                             value=float(milieu[1]), step=10.0)

st.sidebar.header("Matériel (géophones)")
f_coupure = st.sidebar.select_slider(
    "Fréquence de coupure du géophone (Hz)",
    options=[1.0, 2.0, 4.5, 10.0, 14.0, 28.0, 40.0, 100.0],
    value=4.5,
)

st.sidebar.header("Critères d'interprétation")
critere_prudence = st.sidebar.radio(
    "Critère de longueur d'onde maximale",
    ["Prudent (λmax = 0.5·L)", "Optimiste (λmax = 1·L)"],
)
facteur_profondeur = st.sidebar.slider("Facteur profondeur / longueur d'onde (z ≈ k·λ)", 0.2, 0.5, 0.3, 0.05)
diviseur_resolution = st.sidebar.radio("Critère de résolution verticale (λ/n)", [2, 3, 4], index=1)

# --- CALCULS ---
res = calculer_parametres_masw(n_geophones, dx, vr, f_coupure, critere_prudence, facteur_profondeur)

# --- ZONE D'AFFICHAGE ET POP-UP THÉORIQUE ---
col_title, col_help = st.columns([0.85, 0.15])

with col_help:
    with st.popover("📖 Résumé Théorique"):
        st.markdown("### Fondamentaux du Module")
        st.subheader("1. Longueur du dispositif")
        st.latex(r"L = (N_{g\acute{e}o} - 1) \times \Delta x")
        st.subheader("2. Longueurs d'onde exploitables")
        st.write("Aliasing spatial (limite haute fréquence) et limite basse fréquence liée au dispositif :")
        st.latex(r"\lambda_{min} = 2\Delta x \qquad \lambda_{max} = 0.5\,L \text{ à } 1\,L")
        st.subheader("3. Fréquences associées")
        st.latex(r"f_{max} = \frac{V_r}{\lambda_{min}} \qquad f_{min,\,dispositif} = \frac{V_r}{\lambda_{max}}")
        st.write(
            "Au-delà de fmax, repliement en nombre d'onde (aliasing spatial) : les vitesses apparentes "
            "sur l'image de dispersion deviennent fausses. En dessous de fmin, la courbe de dispersion "
            "devient instable (longueur d'onde trop grande devant le dispositif). Par ailleurs, le géophone "
            "ne répond pas efficacement sous sa fréquence de coupure : la fréquence minimale réellement "
            "exploitable est le maximum entre celle imposée par le dispositif et celle imposée par le capteur.")
        st.latex(r"f_{min,\,effectif} = \max(f_{min,\,dispositif},\ f_{coupure})")
        st.subheader("4. Profondeur d'investigation")
        st.latex(r"z_{min} \approx k\,\lambda_{min} \qquad z_{max} \approx k\,\lambda_{max,\,effectif}")
        st.write("k ≈ 0.3 est une règle empirique courante (0.2 à 0.5 selon les auteurs).")
        st.subheader("5. Résolution verticale (transitions, couches minces, vides)")
        st.write(
            "La MASW ne détecte pas un objet isolé : elle restitue un profil 1D de vitesse Vs par inversion "
            "de la courbe de dispersion. Une transition de vitesse, une couche mince ou un vide n'est "
            "perceptible que si son épaisseur représente une fraction significative de la longueur d'onde "
            "qui échantillonne cette profondeur — plus la profondeur augmente, plus la longueur d'onde "
            "nécessaire est grande, et plus la résolution verticale se dégrade.")
        st.latex(r"\lambda(z) = \frac{z}{k} \qquad \acute{e}paisseur_{min}(z) = \frac{\lambda(z)}{n},\ \ n \in [2, 4]")

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Longueur du dispositif (L)", f"{res['L']:.1f} m")
col2.metric("λmin / λmax effectif", f"{res['lambda_min']:.2f} m / {res['lambda_max_effectif']:.2f} m")
col3.metric("fmax (aliasing) / fmin effectif", f"{res['f_max_aliasing']:.1f} Hz / {res['f_min_effectif']:.1f} Hz")
col4.metric("Facteur limitant", res["limitant"])

col5, col6 = st.columns(2)
col5.metric("Profondeur min investigable", f"{res['z_min']:.2f} m")
col6.metric("Profondeur max investigable", f"{res['z_max']:.2f} m")

st.info(
    f"Bande de fréquences à pointer sur l'image de dispersion : "
    f"**{res['f_min_effectif']:.1f} Hz → {res['f_max_aliasing']:.1f} Hz**. "
    f"En dehors de cette bande, la courbe de dispersion n'est pas exploitable."
)

# --- GRAPHIQUES ---
col_g1, col_g2 = st.columns(2)

z_affichage_max = max(res["z_max"] * 1.3, 10.0)
z_array = np.linspace(0.05, z_affichage_max, 500)
epaisseur_array = epaisseur_min_resolvable(z_array, vr, facteur_profondeur, diviseur_resolution)

with col_g1:
    st.subheader("Résolution verticale vs Profondeur")
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epaisseur_array, z_array, color="#3498db", linewidth=2,
             label="Épaisseur min. résolvable (couche / transition / vide)")

    ax1.axhspan(0, res["z_min"], color="gray", alpha=0.3, label="Zone peu fiable (proche surface)")
    ax1.axhspan(res["z_max"], z_affichage_max, color="black", alpha=0.4, label="Hors profondeur d'investigation")

    ax1.set_xlabel("Épaisseur minimale résolvable (m)")
    ax1.set_ylabel("Profondeur (m)")
    ax1.set_ylim(z_affichage_max, 0)
    ax1.set_xlim(left=0)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, -0.12), ncol=1, fontsize="small", frameon=False)
    fig1.subplots_adjust(bottom=0.3)

    st.pyplot(fig1)

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Résolution", data=buf1.getvalue(), file_name="masw_resolution.png",
                       mime="image/png", use_container_width=True)

with col_g2:
    st.subheader("Schéma du Dispositif")
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    positions_geophones = np.arange(n_geophones) * dx
    ax2.axhspan(-2, 0, facecolor="#e0f7fa", alpha=1)
    ax2.axhspan(0, z_affichage_max, facecolor="#d7ccc8", alpha=1)
    ax2.axhline(0, color="#5d4037", linewidth=3)

    ax2.axhspan(res["z_min"], res["z_max"], color="#2ecc71", alpha=0.15, label="Zone d'investigation fiable")
    ax2.axhspan(res["z_max"], z_affichage_max, color="black", alpha=0.4)
    ax2.axhline(res["z_min"], color="#2ecc71", linestyle=":", linewidth=1.5)
    ax2.axhline(res["z_max"], color="#2ecc71", linestyle=":", linewidth=1.5)

    ax2.scatter(positions_geophones, np.zeros(n_geophones), marker="v", color="#e67e22", s=60, zorder=4,
                label="Géophones")

    ax2.set_xlim(-dx, res["L"] + dx)
    ax2.set_ylim(z_affichage_max, -2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Distance le long du profil (m)")
    ax2.set_ylabel("Profondeur (m)")
    ax2.legend(loc="upper right", bbox_to_anchor=(1.0, -0.12), ncol=1, fontsize="small", frameon=False)
    fig2.subplots_adjust(bottom=0.3)

    st.pyplot(fig2)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Dispositif", data=buf2.getvalue(), file_name="masw_dispositif.png",
                       mime="image/png", use_container_width=True)
