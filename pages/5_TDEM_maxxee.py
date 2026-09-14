import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import io

try:
    import empymod
    EMPYMOD_OK = True
except ImportError:
    EMPYMOD_OK = False


# ==========================================
# PARAMÈTRES FIXES (GroundTEM Trek, non modifiables dans l'interface)
# ==========================================
MU0 = 4e-7 * np.pi
RES_AIR = 2e14          # résistivité de l'air (demi-espace supérieur)
FACTOR_NV = 1e9         # conversion V -> nV

COTE_TX = 0.65          # côté de la boucle émettrice (m)
TOURS_TX = 4            # nombre de spires de la boucle émettrice
COTE_RX = 0.65          # côté de la boucle réceptrice (m)
TOURS_RX = 53           # nombre de spires de la boucle réceptrice
COURANT = 10.0          # courant d'émission, moment "High" (A)

T_MIN = 5e-6            # premier temps de mesure (s)
T_MAX = 5e-3            # dernier temps de mesure (s)
N_GATES = 30            # nombre de temps de mesure

MOMENT_TX = COURANT * COTE_TX ** 2 * TOURS_TX
AIRE_EFF_RX = COTE_RX ** 2 * TOURS_RX
TIMES = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_GATES)

# Matériaux : résistivités typiques (Ω·m)
TERRAINS = {
    "Argile": 10.0,
    "Alluvions sablo-argileuses": 30.0,
    "Limon": 40.0,
    "Craie": 80.0,
    "Sable saturé": 100.0,
    "Marno-calcaire": 120.0,
    "Remblai": 150.0,
    "Sable sec": 500.0,
    "Calcaire compact": 800.0,
    "Granite": 3000.0,
}
REMPLISSAGES = {
    "Argile": 10.0,
    "Eau": 30.0,
    "Zone décomprimée (roche fracturée)": 40.0,
    "Sable saturé": 80.0,
    "Remblai / béton": 200.0,
    "Sable sec": 500.0,
    "Vide (air)": 5000.0,
}
# Bruit du site : (bruit à 1 ms en nV/m², ordre de grandeur usuel en TDEM ; erreur systématique en %)
BRUITS = {
    "Calme (campagne, loin des lignes électriques)": (1.0, 3.0),
    "Moyen (zone péri-urbaine, routes)": (5.0, 5.0),
    "Bruité (ville, clôtures, lignes haute tension)": (30.0, 8.0),
}
PENTE_BRUIT = 0.5


# ==========================================
# MOTEUR PHYSIQUE : TDEM (forward 1D)
# ==========================================
def construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                      karst_actif, z_toit, ep_karst, res_karst):
    """
    Construit le modèle 1D en couches attendu par empymod.
    Retourne (res, depth) avec :
      res   = [air, couche1, couche2, ...]
      depth = profondeurs des interfaces (la première est 0 = surface)
    """
    if not karst_actif:
        return [RES_AIR, res_recouvrement, res_substratum], [0.0, ep_recouvrement]

    z_base = z_toit + ep_karst

    if z_toit > ep_recouvrement:
        # cible entièrement dans le substratum
        return ([RES_AIR, res_recouvrement, res_substratum, res_karst, res_substratum],
                [0.0, ep_recouvrement, z_toit, z_base])
    if z_base >= ep_recouvrement:
        # cible à cheval sur l'interface : elle remplace la base du recouvrement
        return ([RES_AIR, res_recouvrement, res_karst, res_substratum],
                [0.0, z_toit, z_base])
    # cible entièrement dans le recouvrement
    return ([RES_AIR, res_recouvrement, res_karst, res_recouvrement, res_substratum],
            [0.0, z_toit, z_base, ep_recouvrement])


def reponse_tdem(res, depth, offset):
    """
    Réponse transitoire dB/dt d'un dispositif à boucles horizontales séparées de `offset`,
    exprimée en tension induite aux bornes du récepteur (V).
    Vérifié contre la solution analytique de Ward & Hohmann (demi-espace, eq. 4.70).
    """
    sans_permittivite = [0.0] * len(res)      # régime diffusif : sans courants de déplacement
    out = empymod.bipole(
        src=[0.0, 0.0, 0.0, 0.0, 90.0],      # boucle horizontale (dip 90°)
        rec=[offset, 0.0, 0.0, 0.0, 90.0],
        depth=depth,
        res=res,
        freqtime=TIMES,
        signal=-1,                            # coupure du courant (switch-off)
        msrc="b", mrec="b",                   # boucles -> dB/dt (et non B)
        epermH=sans_permittivite, epermV=sans_permittivite,
        verb=0,
    )
    return -np.asarray(out, dtype=float) * MOMENT_TX * AIRE_EFF_RX


def niveau_bruit(bruit_1ms_nv_m2):
    """
    Bruit en loi de puissance du temps : bruit(t) = bruit(1 ms) * (t / 1 ms) ** (-pente).
    Donné en nV/m², converti en tension aux bornes du récepteur (V).
    """
    return (bruit_1ms_nv_m2 * AIRE_EFF_RX / FACTOR_NV) * (TIMES / 1e-3) ** (-PENTE_BRUIT)


def anomalie_relative(sig_avec, sig_sans):
    """Écart relatif (%) entre la réponse avec cible et la réponse de référence."""
    base = np.where(np.abs(sig_sans) <= 0, np.nan, np.abs(sig_sans))
    return np.nan_to_num(np.abs(sig_avec - sig_sans) / base * 100.0)


def espacement_profils(largeur, z_centre, detection_nette):
    """
    Ordre de grandeur de l'espacement des profils et des stations (règle empirique).
    - Largeur de l'anomalie en surface : la cavité est « vue » dans un cône partant de son centre.
      Cône à 45° si la détection est nette, plus étroit (≈ 27°) si elle est incertaine,
      car seul le cœur de l'anomalie dépasse le bruit.
    - Profils : au moins 3 profils recoupent l'anomalie (centre + deux bords) -> largeur / 3.
    - Stations : au moins 4 à 5 mesures sur l'anomalie le long d'un profil -> largeur / 4.
    """
    facteur = 2.0 if detection_nette else 1.0
    largeur_anomalie = largeur + facteur * z_centre
    arrondi = lambda x: max(1.0, np.floor(x * 2.0) / 2.0)
    return largeur_anomalie, arrondi(largeur_anomalie / 3.0), arrondi(largeur_anomalie / 4.0)


def gates_exploitables(sig_avec, sig_sans, bruit_v, seuil_pct):
    """Temps où l'écart dépasse à la fois le bruit et l'erreur systématique."""
    ecart = np.abs(sig_avec - sig_sans)
    return (ecart >= bruit_v) & (anomalie_relative(sig_avec, sig_sans) >= seuil_pct)


def evaluer(sig_avec, sig_sans, bruit_v, seuil_pct):
    """
    Verdict de détectabilité :
    - détectable : écart ≥ 5 × bruit et au moins 3 temps exploitables
    - incertain  : écart ≥ 2 × bruit et au moins 1 temps exploitable
    """
    snr = float(np.max(np.abs(sig_avec - sig_sans) / bruit_v))
    n_valides = int(np.sum(gates_exploitables(sig_avec, sig_sans, bruit_v, seuil_pct)))
    if snr >= 5 and n_valides >= 3:
        return "detectable"
    if snr >= 2 and n_valides >= 1:
        return "incertain"
    return "non_detectable"


PROFONDEURS_TEST = np.arange(2.0, 60.5, 2.0)


@st.cache_data(show_spinner=False)
def profondeur_limite(res_recouvrement, ep_recouvrement, res_substratum, ep_karst, res_karst,
                      offset, bruit_1ms_nv_m2, seuil_pct):
    """
    Déplace la cavité de 2 à 60 m et renvoie (anomalies max, profondeur limite).
    Profondeur limite = plus grande profondeur testée où la cavité est encore détectable
    (None si elle ne l'est à aucune profondeur).
    """
    bruit_v = niveau_bruit(bruit_1ms_nv_m2)
    r_s, d_s = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                 False, 0.0, ep_karst, res_karst)
    s_s = reponse_tdem(r_s, d_s, offset)
    mesurable = np.abs(s_s) >= bruit_v
    anomalies, z_limite = [], None
    for z in PROFONDEURS_TEST:
        r_a, d_a = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                     True, z, ep_karst, res_karst)
        s_a = reponse_tdem(r_a, d_a, offset)
        anomalies.append(float(np.max(anomalie_relative(s_a, s_s)[mesurable], initial=0.0)))
        if evaluer(s_a, s_s, bruit_v, seuil_pct) != "non_detectable":
            z_limite = float(z)
    return np.array(anomalies), z_limite


# ==========================================
# INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Module TDEM Maxxé", layout="wide")

st.title("Module TDEM Maxxé — Karst")

st.markdown(
    "Est-ce qu'une cavité karstique, connue par sondage, peut être vue en TDEM ? "
    "Décrivez le terrain et la cavité dans le panneau de gauche : le module simule la mesure "
    "avec et sans cavité et indique si la différence est plus forte que le bruit."
)

if not EMPYMOD_OK:
    st.error(
        "Le module **empymod** est requis pour le calcul. "
        "Installation : `pip install empymod`"
    )
    st.stop()


def libelle(materiaux):
    return lambda nom: f"{nom} (≈ {materiaux[nom]:.0f} Ω·m)"


# --- BARRE LATÉRALE : 1. TERRAIN ---
st.sidebar.header("1. Terrain")
recouvrement = st.sidebar.selectbox(
    "Couche de surface (recouvrement)", list(TERRAINS.keys()),
    index=list(TERRAINS.keys()).index("Alluvions sablo-argileuses"), format_func=libelle(TERRAINS))
ep_recouvrement = st.sidebar.slider("Épaisseur de la couche de surface (m)", 1.0, 60.0, 18.0, 0.5)
substratum = st.sidebar.selectbox(
    "Roche en profondeur (substratum)", list(TERRAINS.keys()),
    index=list(TERRAINS.keys()).index("Calcaire compact"), format_func=libelle(TERRAINS))

# --- BARRE LATÉRALE : 2. CAVITÉ ---
st.sidebar.header("2. Cavité karstique")
remplissage = st.sidebar.selectbox(
    "Remplissage de la cavité", list(REMPLISSAGES.keys()),
    index=0, format_func=libelle(REMPLISSAGES))
z_toit = st.sidebar.slider("Profondeur du toit de la cavité (m)", 1.0, 60.0, 20.5, 0.5)
ep_karst = st.sidebar.slider("Hauteur de la cavité (m)", 0.5, 15.0, 2.5, 0.1)
largeur_karst = st.sidebar.slider(
    "Largeur de la cavité (m)", 1.0, 30.0, 5.0, 0.5,
    help="Extension horizontale de la cavité. Sert uniquement à estimer l'espacement des profils.")

# --- BARRE LATÉRALE : 3. MESURE ---
st.sidebar.header("3. Mesure")
offset = st.sidebar.slider(
    "Offset Tx–Rx (m)", 1.0, 50.0, 15.0, 0.5,
    help="Distance entre la boucle qui émet (Tx) et la boucle qui reçoit (Rx). "
         "Plus elle est grande, plus la mesure regarde en profondeur, mais plus le signal est faible.")
site = st.sidebar.radio("Bruit électromagnétique du site", list(BRUITS.keys()), index=0)

res_recouvrement = TERRAINS[recouvrement]
res_substratum = TERRAINS[substratum]
res_karst = REMPLISSAGES[remplissage]
bruit_1ms_nv, seuil_pct = BRUITS[site]
bruit_v = niveau_bruit(bruit_1ms_nv)

if res_karst > res_substratum:
    st.sidebar.warning(
        "⚠️ Le remplissage est **plus résistant** que la roche autour (ex. : vide, sable sec). "
        "Le TDEM voit très mal ce type de cible : attendez-vous à une anomalie faible."
    )

# --- CALCUL ---
res_ref, dep_ref = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                     False, z_toit, ep_karst, res_karst)
res_cib, dep_cib = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                     True, z_toit, ep_karst, res_karst)

with st.spinner("Calcul des réponses transitoires..."):
    sig_sans = reponse_tdem(res_ref, dep_ref, offset)
    sig_avec = reponse_tdem(res_cib, dep_cib, offset)

ecart_rel = anomalie_relative(sig_avec, sig_sans)
valides = gates_exploitables(sig_avec, sig_sans, bruit_v, seuil_pct)
n_valides = int(np.sum(valides))
snr = float(np.max(np.abs(sig_avec - sig_sans) / bruit_v))
mesurable = np.abs(sig_sans) >= bruit_v                 # temps où le signal sort du bruit
ecart_mesurable = np.where(mesurable, ecart_rel, 0.0)
idx_pic = int(np.argmax(ecart_mesurable))
anomalie_max_pct = float(ecart_mesurable[idx_pic])
t_pic = TIMES[idx_pic]
verdict = evaluer(sig_avec, sig_sans, bruit_v, seuil_pct)

with st.spinner("Recherche de la profondeur limite de détection..."):
    anomalies_prof, z_limite = profondeur_limite(res_recouvrement, ep_recouvrement, res_substratum,
                                                 ep_karst, res_karst, offset, bruit_1ms_nv, seuil_pct)

# --- VERDICT ---
if verdict == "detectable":
    st.success(f"### ✅ Cavité détectable\nLa différence due à la cavité dépasse nettement le bruit "
               f"(anomalie max {anomalie_max_pct:.1f} %, sur {n_valides} temps de mesure).")
elif verdict == "incertain":
    st.warning(f"### ⚠️ Détection incertaine\nLa cavité produit une différence tout juste "
               f"au-dessus du bruit (anomalie max {anomalie_max_pct:.1f} %). "
               f"Essayez un autre offset ou un site moins bruité.")
elif snr < 2:
    st.error("### ❌ Cavité non détectable\nL'effet de la cavité est **plus faible que le bruit** "
             "du site : il ne ressortira pas de la mesure.")
else:
    st.error(f"### ❌ Cavité non détectable\nL'effet de la cavité est trop faible "
             f"(anomalie max {anomalie_max_pct:.1f} %) : il faut au moins {seuil_pct:.0f} % "
             f"d'écart pour l'interpréter sur ce site.")

if z_limite is None:
    texte_limite = "à aucune profondeur"
elif z_limite >= PROFONDEURS_TEST[-1]:
    texte_limite = "> 60 m"
else:
    texte_limite = f"≈ {z_limite:.0f} m"

col_stat1, col_stat2, col_stat3 = st.columns(3)
col_stat1.metric("Anomalie maximale", f"{anomalie_max_pct:.1f} %",
                 help="Écart maximal entre le signal avec et sans cavité, sur les temps où le "
                      "signal sort du bruit.")
col_stat2.metric("Signal / bruit", f"{snr:.1f}" if snr < 100 else "> 100",
                 help="Combien de fois l'écart dû à la cavité est plus fort que le bruit. "
                      "≥ 5 : bien visible · 2 à 5 : limite · < 2 : invisible.")
col_stat3.metric("Détectable jusqu'à", texte_limite,
                 help="Profondeur maximale du toit à laquelle cette même cavité (même remplissage, "
                      "même hauteur) resterait détectable, pour ce terrain et ce bruit.")

# --- GRAPHIQUES ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Courbe de décroissance")
    fig, ax = plt.subplots(figsize=(8, 5))

    t_us = TIMES * 1e6
    ax.loglog(t_us, np.abs(sig_sans) * FACTOR_NV, label="Terrain sans cavité",
              color="#3498db", linewidth=2.5)
    ax.loglog(t_us, np.abs(sig_avec) * FACTOR_NV, label="Terrain avec cavité",
              color="#e67e22", linewidth=2, linestyle="--", marker="o", markersize=4)
    ax.fill_between(t_us, 1e-6, bruit_v * FACTOR_NV, color="gray", alpha=0.25,
                    label="Bruit (non mesurable)")
    if n_valides > 0:
        ax.scatter(t_us[valides], np.abs(sig_avec[valides]) * FACTOR_NV, s=90,
                   facecolors="none", edgecolors="#27ae60", linewidths=2, zorder=5,
                   label="Cavité visible")

    ax.set_xlabel("Temps après la coupure du courant (µs)")
    ax.set_ylabel("Signal mesuré (nV)")
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.set_ylim(bottom=max(np.min(bruit_v * FACTOR_NV) * 0.1, 1e-6))
    ax.text(0.02, 0.03, "← proche surface", transform=ax.transAxes, fontsize=9,
            color="#2c3e50", fontweight="bold")
    ax.text(0.98, 0.03, "plus profond →", transform=ax.transAxes, fontsize=9,
            color="#2c3e50", fontweight="bold", ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize="small", frameon=False)
    fig.subplots_adjust(bottom=0.3)

    st.pyplot(fig)

    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Décroissance", data=buf1.getvalue(),
                       file_name="tdem_decroissance.png", mime="image/png")

with col2:
    st.subheader("Coupe du terrain")
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    prof_max_affichage = max(40.0, (z_toit + ep_karst) * 1.5, ep_recouvrement * 1.5)
    interfaces = list(dep_cib) + [prof_max_affichage]
    couleurs = plt.cm.viridis_r(np.log10(np.clip(res_cib[1:], 1, 1e4)) / 4.0)

    for i in range(len(res_cib) - 1):
        haut, bas = interfaces[i], interfaces[i + 1]
        ax2.add_patch(Rectangle((0, haut), 1, bas - haut, facecolor=couleurs[i],
                                edgecolor="black", linewidth=1.2, zorder=2))
        ax2.text(1.05, (haut + bas) / 2, f"{res_cib[i + 1]:.0f} Ω·m",
                 va="center", fontsize=10, fontweight="bold")

    ax2.add_patch(Rectangle((0, z_toit), 1, ep_karst, facecolor="none",
                            edgecolor="red", linewidth=2.5, linestyle="--", zorder=5))
    ax2.text(0.5, z_toit + ep_karst / 2, remplissage.upper(), ha="center", va="center",
             color="red", fontweight="bold", fontsize=10, zorder=6)

    ax2.axhline(0, color="#5d4037", linewidth=3, zorder=4)
    if z_limite is not None and z_limite < min(PROFONDEURS_TEST[-1], prof_max_affichage):
        ax2.axhline(z_limite, color="#c0392b", linewidth=1.5, linestyle="-.", zorder=4)
        ax2.text(0.02, z_limite - 1, f"Limite de détection ≈ {z_limite:.0f} m",
                 color="#c0392b", fontsize=9, fontweight="bold")

    # dispositif en surface
    for x, couleur, nom in [(0.3, "#e67e22", "Tx"), (0.7, "#2c3e50", "Rx")]:
        ax2.add_patch(Polygon([[x, 0], [x - 0.05, -prof_max_affichage * 0.03],
                               [x + 0.05, -prof_max_affichage * 0.03]],
                              closed=True, facecolor=couleur, zorder=6))
        ax2.text(x, -prof_max_affichage * 0.05, nom, ha="center", fontsize=9, fontweight="bold")
    ax2.text(0.5, -prof_max_affichage * 0.035, f"{offset:.1f} m", ha="center", fontsize=8)

    ax2.set_xlim(0, 1.6)
    ax2.set_ylim(prof_max_affichage, -prof_max_affichage * 0.08)
    ax2.set_xticks([])
    ax2.set_ylabel("Profondeur (m)")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    st.pyplot(fig2)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Coupe", data=buf2.getvalue(),
                       file_name="tdem_coupe.png", mime="image/png")

# --- EXPLICATION DE LA COURBE ---
st.subheader("📖 Comment lire la courbe de décroissance")
st.markdown(f"""
1. **La mesure** : la boucle émettrice (Tx) fait passer un courant, puis le **coupe d'un coup**.
   Cette coupure crée dans le sol des courants qui **s'éteignent peu à peu en s'enfonçant**.
   La boucle réceptrice (Rx), placée à {offset:.1f} m, enregistre ce signal qui s'éteint : c'est la courbe de décroissance.
2. **Axe horizontal = temps après la coupure.** À gauche (quelques µs), le signal vient de la
   **proche surface** ; plus on va vers la droite, plus il vient **de profondeur**.
3. **Axe vertical = force du signal.** Il chute très vite (échelle logarithmique) : c'est normal.
4. **La pente renseigne sur le terrain** : un terrain **conducteur** (argile, eau) garde le signal
   longtemps, la courbe descend lentement. Un terrain **résistant** (calcaire sec, air) le laisse
   s'éteindre vite, la courbe chute rapidement.
5. **Bleu = terrain sans cavité, orange = avec cavité.** Là où les deux courbes **s'écartent**,
   la cavité modifie le signal.
6. **Zone grise = bruit.** Tout ce qui est dedans n'est pas mesurable. La cavité n'est visible que
   si l'écart apparaît **au-dessus** de la zone grise : ces points sont entourés en **vert**.
7. **Un creux en « V »** sur la courbe n'est pas une erreur : avec des boucles séparées, le signal
   change de sens à un instant donné et passe brièvement par zéro.
""")

# --- ANOMALIE RELATIVE ---
st.subheader("Écart dû à la cavité, temps par temps")
st.caption("Violet : écart entre les deux courbes (en %). Zone rouge : écart trop faible pour être "
           "interprété sur ce site. Zone verte : temps où la cavité est réellement visible.")

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.semilogx(TIMES * 1e6, np.where(mesurable, ecart_rel, np.nan), color="#8e44ad", linewidth=2.5,
             marker="o", markersize=5, label="Écart avec / sans cavité (signal hors bruit)")
ax3.axhline(seuil_pct, color="red", linestyle="--", linewidth=1.8,
            label=f"Seuil d'interprétation ({seuil_pct:.0f} %)")
ax3.fill_between(TIMES * 1e6, 0, seuil_pct, color="red", alpha=0.08)
if n_valides > 0:
    ax3.fill_between(TIMES * 1e6, 0, ecart_rel, where=valides, color="#27ae60", alpha=0.2,
                     label="Cavité visible")
ax3.annotate(f"max : {anomalie_max_pct:.1f} %", xy=(t_pic * 1e6, anomalie_max_pct),
             xytext=(t_pic * 1e6 * 1.6, anomalie_max_pct * 0.85), fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#2c3e50"))
ax3.set_xlabel("Temps après la coupure du courant (µs)")
ax3.set_ylabel("Écart (%)")
ax3.set_xlim(TIMES[0] * 1e6, TIMES[-1] * 1e6)
ax3.grid(True, which="both", linestyle=":", alpha=0.7)
ax3.legend(loc="upper right", fontsize="small", frameon=False)

st.pyplot(fig3)

buf3 = io.BytesIO()
fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight")
st.download_button(label="📸 Snapshot Écart", data=buf3.getvalue(),
                   file_name="tdem_ecart.png", mime="image/png")

# --- ESPACEMENT DES PROFILS ---
st.subheader("📐 Espacement des profils pour localiser la cavité")

z_centre = z_toit + ep_karst / 2.0

if verdict == "non_detectable":
    st.error("La cavité n'est pas détectable avec ces réglages : resserrer les profils n'y changera rien. "
             "Modifiez l'offset ou vérifiez le bruit du site avant de dimensionner la maille.")
else:
    largeur_anomalie, esp_profils, esp_stations = espacement_profils(
        largeur_karst, z_centre, verdict == "detectable")

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("Largeur de l'anomalie en surface", f"≈ {largeur_anomalie:.0f} m",
                  help="Zone, en surface, au-dessus de laquelle la cavité modifie le signal.")
    col_e2.metric("Espacement des profils", f"≤ {esp_profils:.1f} m")
    col_e3.metric("Espacement des stations", f"≤ {esp_stations:.1f} m",
                  help="Distance entre deux mesures successives le long d'un profil.")

    col_p1, col_p2 = st.columns([0.55, 0.45])

    with col_p1:
        fig5, ax5 = plt.subplots(figsize=(7, 6))
        demi = max(largeur_anomalie * 1.3, 3 * esp_profils)

        ax5.add_patch(plt.Circle((0, 0), largeur_anomalie / 2, facecolor="#f5b041", alpha=0.25,
                                 edgecolor="#e67e22", linestyle="--", linewidth=2,
                                 label="Anomalie en surface"))
        ax5.add_patch(plt.Circle((0, 0), largeur_karst / 2, facecolor="#c0392b", alpha=0.6,
                                 edgecolor="#922b21", linewidth=1.5, label="Cavité (vue de dessus)"))

        # profils décalés d'un demi-pas : cas le moins favorable (aucun profil pile sur la cavité)
        y_profils = np.arange(-demi, demi + esp_profils, esp_profils) + esp_profils / 2
        x_stations = np.arange(-demi, demi + esp_stations, esp_stations)
        for k, y in enumerate(y_profils):
            ax5.axhline(y, color="#2c3e50", linewidth=1, alpha=0.6,
                        label="Profils" if k == 0 else None)
            ax5.plot(x_stations, np.full_like(x_stations, y), "o", color="#2c3e50", markersize=3)
            dans = x_stations ** 2 + y ** 2 <= (largeur_anomalie / 2) ** 2
            ax5.plot(x_stations[dans], np.full(int(dans.sum()), y), "o", color="#27ae60",
                     markersize=6, label="Stations sur l'anomalie" if k == 0 else None)

        ax5.set_xlim(-demi, demi)
        ax5.set_ylim(-demi, demi)
        ax5.set_aspect("equal")
        ax5.set_xlabel("Distance (m)")
        ax5.set_ylabel("Distance (m)")
        ax5.set_title("Vue en plan de la maille conseillée", fontsize=11)
        ax5.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize="small", frameon=False)
        fig5.subplots_adjust(bottom=0.25)

        st.pyplot(fig5)

        buf5 = io.BytesIO()
        fig5.savefig(buf5, format="png", dpi=300, bbox_inches="tight")
        st.download_button(label="📸 Snapshot Maille", data=buf5.getvalue(),
                           file_name="tdem_maille.png", mime="image/png")

    with col_p2:
        st.markdown(f"""
**D'où viennent ces valeurs ?**

1. **La cavité se voit sur une zone plus large qu'elle.** Le signal TDEM « regarde » dans un
   cône sous le dispositif : plus la cavité est profonde, plus la zone en surface où elle
   influence la mesure est étendue, mais plus l'effet y est dilué.
   Ici : {largeur_karst:.1f} m de cavité à {z_centre:.1f} m de profondeur
   → anomalie sur **≈ {largeur_anomalie:.0f} m**.
2. **Détecter ≠ localiser.** Un seul profil qui passe dessus suffit à *voir* quelque chose.
   Pour *localiser* la cavité (centre et bords), il faut qu'**au moins 3 profils** la recoupent
   → profils tous les **{esp_profils:.1f} m**.
3. **Le long d'un profil**, il faut **4 à 5 mesures** sur l'anomalie pour dessiner sa forme
   → une station tous les **{esp_stations:.1f} m**.
4. **Sur le plan**, les profils sont placés dans le cas le moins favorable (aucun ne passe pile
   sur la cavité) : les points **verts** sont les mesures qui la « voient ».

**Conseils terrain**
- Si la cavité est une **galerie allongée**, orientez les profils **perpendiculairement** à sa direction supposée.
- Gardez **toujours la même orientation Tx → Rx** d'une station à l'autre : le point de mesure est au milieu des deux boucles.
- Faites passer **un profil sur le sondage** qui a recoupé la cavité, pour avoir une mesure de référence.
""")
        if verdict == "incertain":
            st.warning("Détection incertaine : seul le cœur de l'anomalie dépasse le bruit, la zone "
                       "utile est donc plus étroite et la maille plus serrée.")
        if largeur_karst < z_centre:
            st.info("La cavité est plus étroite que profonde : son effet réel sera nettement plus faible "
                    "que celui calculé en 1D. Le verdict ci-dessus est optimiste.")

    st.caption("Ordres de grandeur issus de règles empiriques (cône d'influence de 45°, 3 profils et "
               "4 stations sur l'anomalie), pas d'une modélisation 3D.")

# --- PROFONDEUR LIMITE ---
with st.expander("🔎 Jusqu'à quelle profondeur cette cavité serait-elle visible ?"):
    st.write("La même cavité (même remplissage, même hauteur, même terrain) est placée à des "
             "profondeurs de 2 à 60 m : la courbe montre l'écart qu'elle produit à chaque profondeur.")

    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(PROFONDEURS_TEST, np.maximum(anomalies_prof, 1e-3), color="#8e44ad", linewidth=2.5,
             marker="o", markersize=5, label="Écart maximal")
    ax4.axhline(seuil_pct, color="red", linestyle="--", linewidth=1.8,
                label=f"Seuil d'interprétation ({seuil_pct:.0f} %)")
    ax4.axvline(z_toit, color="#27ae60", linestyle=":", linewidth=2,
                label=f"Cavité étudiée ({z_toit:.1f} m)")
    if z_limite is not None and z_limite < PROFONDEURS_TEST[-1]:
        ax4.axvline(z_limite, color="#c0392b", linestyle="-.", linewidth=2,
                    label=f"Détectable jusqu'à ≈ {z_limite:.0f} m")
    ax4.set_xlabel("Profondeur du toit de la cavité (m)")
    ax4.set_ylabel("Écart maximal (%)")
    ax4.set_yscale("log")
    ax4.grid(True, which="both", linestyle=":", alpha=0.7)
    ax4.legend(fontsize="small", frameon=False)

    st.pyplot(fig4)
    st.caption("La détection dépend aussi du bruit : une cavité peut dépasser le seuil en % "
               "et rester invisible si le signal est lui-même noyé dans le bruit.")

    buf4 = io.BytesIO()
    fig4.savefig(buf4, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Profondeur limite", data=buf4.getvalue(),
                       file_name="tdem_profondeur_limite.png", mime="image/png")

# --- HYPOTHÈSES ---
with st.expander("⚙️ Hypothèses du calcul"):
    st.markdown(f"""
- **Appareil** : GroundTEM Trek, boucles de {COTE_TX} m de côté ({TOURS_TX} spires en émission,
  {TOURS_RX} en réception), courant de {COURANT:.0f} A.
- **Temps de mesure** : {N_GATES} mesures entre {T_MIN * 1e6:.0f} µs et {T_MAX * 1e3:.0f} ms après la coupure.
- **Terrain en couches horizontales (1D)** : la cavité est traitée comme une couche qui s'étend à
  l'infini sur les côtés. Une vraie cavité de quelques mètres de large donnera un **écart plus faible** :
  le résultat est un **cas favorable**.
- **Résistivités** : valeurs typiques des matériaux, elles peuvent varier fortement d'un site à l'autre.
- **Bruit** : « {site} » = {bruit_1ms_nv:.0f} nV/m² à 1 ms (ordre de grandeur, à remplacer par une mesure sur site si possible), et un écart d'au moins {seuil_pct:.0f} % pour être interprété.
""")
