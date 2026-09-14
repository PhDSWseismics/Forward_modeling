import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LogNorm
import io

try:
    import empymod
    EMPYMOD_OK = True
except ImportError:
    EMPYMOD_OK = False


# ==========================================
# MOTEUR PHYSIQUE : TDEM (forward 1D)
# ==========================================
MU0 = 4e-7 * np.pi
RES_AIR = 2e14          # résistivité de l'air (demi-espace supérieur)
FACTOR_NV = 1e9         # conversion V -> nV


def construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                      karst_actif, z_toit, ep_karst, res_karst):
    """
    Construit le modèle 1D en couches attendu par empymod.
    Retourne (res, depth) avec :
      res   = [air, couche1, couche2, ...]
      depth = profondeurs des interfaces (la première est 0 = surface)
    Le karst est inséré comme une couche dans le substratum si z_toit > ep_recouvrement,
    sinon il est inséré dans le recouvrement.
    """
    if not karst_actif:
        return [RES_AIR, res_recouvrement, res_substratum], [0.0, ep_recouvrement]

    z_base = z_toit + ep_karst

    if z_toit >= ep_recouvrement:
        # karst dans le substratum
        res = [RES_AIR, res_recouvrement, res_substratum, res_karst, res_substratum]
        depth = [0.0, ep_recouvrement, z_toit, z_base]
    else:
        # karst (ou décompression) déjà présent dans le recouvrement
        res = [RES_AIR, res_recouvrement, res_karst, res_recouvrement, res_substratum]
        depth = [0.0, z_toit, z_base, max(ep_recouvrement, z_base + 0.1)]

    return res, depth


def reponse_tdem(res, depth, times, offset, courant,
                 aire_tx, tours_tx, aire_rx, tours_rx, config):
    """
    Réponse transitoire dB/dt d'un dispositif à boucles horizontales sur un modèle 1D.
    - config "offset"  : TX et RX séparés de `offset` (GroundTEM Trek, moving loop)
    - config "centrale": RX au centre de TX (offset nul, valeur plancher imposée)
    Retourne la tension induite aux bornes du récepteur, en volts.
    """
    d = 0.1 if config == "centrale" else offset   # 0 exact -> singularité numérique

    out = empymod.loop(
        src=[0.0, 0.0, 0.0, 0.0, 90.0],      # boucle horizontale (dip 90°)
        rec=[d, 0.0, 0.0, 0.0, 90.0],
        depth=depth,
        res=res,
        freqtime=times,
        signal=-1,                            # coupure du courant (switch-off) -> dB/dt
        mrec=True,
        verb=0,
    )
    moment_tx = courant * aire_tx * tours_tx
    aire_eff_rx = aire_rx * tours_rx
    return np.asarray(out, dtype=float) * moment_tx * aire_eff_rx


def niveau_bruit(times, bruit_1ms_nv, pente_bruit):
    """
    Modèle de bruit usuel en TDEM : décroissance en loi de puissance du temps.
    bruit(t) = bruit(1 ms) * (t / 1 ms) ** (-pente)
    Retourne le bruit en volts.
    """
    return (bruit_1ms_nv / FACTOR_NV) * (times / 1e-3) ** (-abs(pente_bruit))


def resistivite_apparente_tardive(dbdt, times, moment_tx, aire_eff_rx):
    """
    Résistivité apparente aux temps tardifs (approximation asymptotique, boucle centrale).
    INDICATIF uniquement : non valable aux temps courts ni pour un fort offset.
    """
    dbdt_norm = np.abs(dbdt) / aire_eff_rx           # V/m^2
    dbdt_norm = np.where(dbdt_norm <= 0, np.nan, dbdt_norm)
    terme = (2.0 * MU0 * moment_tx) / (5.0 * (times ** 2.5) * dbdt_norm)
    return (MU0 / (4.0 * np.pi)) * terme ** (2.0 / 3.0)


def doi_spies(res_moyenne, moment_tx, bruit_v, aire_eff_rx):
    """
    Profondeur d'investigation approchée (Spies, 1989) pour un dispositif de type boucle.
    Approximation empirique : à recouper avec une DOI par sensibilité cumulée (Christiansen
    & Auken, 2012) si l'inversion est faite sous AarhusInv / Workbench.
    """
    eta = max(bruit_v / aire_eff_rx, 1e-18)          # bruit ramené en V/m^2
    return 0.55 * ((moment_tx * res_moyenne) / eta) ** 0.2


def anomalie_relative(sig_avec, sig_sans):
    """Écart relatif (%) entre la réponse avec cible et la réponse de référence."""
    base = np.where(np.abs(sig_sans) <= 0, np.nan, np.abs(sig_sans))
    return np.abs(sig_avec - sig_sans) / base * 100.0


# ==========================================
# INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Module TDEM Maxxé", layout="wide")

st.title("Module TDEM Maxxé — Karst")

st.markdown(
    "Modélisation directe (forward) 1D d'un sondage TDEM au sol : justification méthodologique "
    "de la détectabilité d'une cible karstique (vide, remplissage argileux, zone décomprimée) "
    "à partir des dimensions connues par sondages, avec test de détectabilité (SNR) et "
    "profondeur limite de détection."
)

if not EMPYMOD_OK:
    st.error(
        "Le module **empymod** est requis pour le calcul forward. "
        "Installation : `pip install empymod`"
    )
    st.stop()

# --- BARRE LATÉRALE : CONTEXTE GÉOLOGIQUE ---
st.sidebar.header("Contexte géologique")

res_encaissants = {
    "Alluvions sablo-argileuses": 30.0,
    "Argile": 10.0,
    "Limon": 40.0,
    "Sable sec": 500.0,
    "Sable saturé": 100.0,
    "Marno-calcaire": 120.0,
    "Calcaire compact": 800.0,
    "Craie": 80.0,
    "Granite": 3000.0,
}
res_remplissages = {
    "Vide (air)": 5000.0,
    "Zone décomprimée (fracturée)": 40.0,
    "Remplissage argileux": 10.0,
    "Remplissage sableux saturé": 80.0,
    "Eau": 30.0,
    "Remblai / béton": 200.0,
}

recouvrement = st.sidebar.selectbox(
    "Recouvrement (couche 1)", list(res_encaissants.keys()), index=0)
substratum = st.sidebar.selectbox(
    "Substratum (encaissant de la cible)", list(res_encaissants.keys()), index=5)
remplissage = st.sidebar.selectbox(
    "Remplissage de la cible", list(res_remplissages.keys()), index=1)

res_recouvrement = st.sidebar.slider(
    "ρ recouvrement (Ω·m)", 1.0, 3000.0, float(res_encaissants[recouvrement]), 1.0)
ep_recouvrement = st.sidebar.slider(
    "Épaisseur du recouvrement (m)", 1.0, 60.0, 18.0, 0.5)
res_substratum = st.sidebar.slider(
    "ρ substratum (Ω·m)", 1.0, 5000.0, float(res_encaissants[substratum]), 1.0)

st.sidebar.caption(
    f"Contraste cible / substratum : {res_remplissages[remplissage]:.0f} Ω·m "
    f"vs {res_substratum:.0f} Ω·m"
)

# --- BARRE LATÉRALE : CIBLE ---
st.sidebar.header("Cible karstique")
karst_actif = st.sidebar.checkbox("Insérer la cible dans le modèle", value=True)
z_toit = st.sidebar.slider("Profondeur du toit (m)", 1.0, 60.0, 20.5, 0.5)
ep_karst = st.sidebar.slider("Épaisseur de la cible (m)", 0.5, 15.0, 2.5, 0.1)
res_karst = st.sidebar.slider(
    "ρ de la cible (Ω·m)", 1.0, 10000.0, float(res_remplissages[remplissage]), 1.0)

if res_karst > res_substratum:
    st.sidebar.warning(
        "⚠️ Cible **résistante** par rapport à l'encaissant : le TDEM y est peu sensible "
        "(les courants de Foucault se développent mal dans un résistant). Vérifier que "
        "l'anomalie dépasse bien le bruit ci-dessous avant de conclure."
    )

# --- BARRE LATÉRALE : DISPOSITIF ---
st.sidebar.header("Dispositif (GroundTEM Trek)")
config = st.sidebar.selectbox("Configuration", ["offset", "centrale"], index=0,
                              format_func=lambda x: {
                                  "offset": "Offset / moving loop (TX et RX séparés)",
                                  "centrale": "Boucle centrale (RX au centre de TX)",
                              }[x])
cote_tx = st.sidebar.number_input("Côté boucle TX (m)", min_value=0.1, value=0.65, step=0.05)
tours_tx = st.sidebar.number_input("Nombre de tours TX", min_value=1, value=4, step=1)
cote_rx = st.sidebar.number_input("Côté boucle RX (m)", min_value=0.1, value=0.65, step=0.05)
tours_rx = st.sidebar.number_input("Nombre de tours RX", min_value=1, value=53, step=1)
offset = st.sidebar.slider("Offset TX–RX (m)", 1.0, 50.0, 15.0, 0.5)
moment_choix = st.sidebar.radio("Moment", ["Low (1 A)", "High (10 A)"], index=1, horizontal=True)
courant = 1.0 if moment_choix.startswith("Low") else 10.0

aire_tx = cote_tx ** 2
aire_rx = cote_rx ** 2
moment_tx = courant * aire_tx * tours_tx
aire_eff_rx = aire_rx * tours_rx

st.sidebar.caption(
    f"Moment TX = {moment_tx:.2f} A·m² · Aire effective RX = {aire_eff_rx:.1f} m²"
)

# --- BARRE LATÉRALE : FENÊTRES TEMPORELLES ---
st.sidebar.header("Fenêtres temporelles (gates)")
t_min_us = st.sidebar.number_input("Premier gate (µs)", min_value=1.0, value=5.0, step=1.0)
t_max_ms = st.sidebar.number_input("Dernier gate (ms)", min_value=0.01, value=5.0, step=0.5)
n_gates = st.sidebar.slider("Nombre de gates", 10, 60, 30, 1)

times = np.logspace(np.log10(t_min_us * 1e-6), np.log10(t_max_ms * 1e-3), int(n_gates))

# --- BARRE LATÉRALE : BRUIT ---
st.sidebar.header("Bruit & détectabilité")
bruit_1ms_nv = st.sidebar.number_input(
    "Niveau de bruit à 1 ms (nV)", min_value=0.001, value=0.5, step=0.1, format="%.3f")
pente_bruit = st.sidebar.slider("Pente de décroissance du bruit", 0.0, 1.5, 0.5, 0.1)
bruit_relatif_pct = st.sidebar.slider(
    "Plancher de bruit relatif (%)", 0.0, 20.0, 3.0, 0.5,
    help="Erreur systématique résiduelle (calibration, géométrie, dérive) : une anomalie "
         "inférieure à ce seuil n'est pas interprétable même si elle dépasse le bruit absolu.")

bruit_v = niveau_bruit(times, bruit_1ms_nv, pente_bruit)

# --- CALCUL DES DONNÉES ---
res_ref, dep_ref = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                     False, z_toit, ep_karst, res_karst)
res_cib, dep_cib = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                     karst_actif, z_toit, ep_karst, res_karst)

with st.spinner("Calcul des réponses transitoires..."):
    sig_sans = reponse_tdem(res_ref, dep_ref, times, offset, courant,
                            aire_tx, tours_tx, aire_rx, tours_rx, config)
    sig_avec = reponse_tdem(res_cib, dep_cib, times, offset, courant,
                            aire_tx, tours_tx, aire_rx, tours_rx, config)

ecart_abs = np.abs(sig_avec - sig_sans)
ecart_rel = anomalie_relative(sig_avec, sig_sans)

# --- SNR / DÉTECTABILITÉ ---
snr_par_gate = np.where(bruit_v > 0, ecart_abs / bruit_v, 0.0)
gates_valides = (snr_par_gate >= 1.0) & (ecart_rel >= bruit_relatif_pct)
n_gates_valides = int(np.sum(gates_valides))

snr = float(np.nanmax(snr_par_gate)) if len(snr_par_gate) else 0.0
idx_pic = int(np.nanargmax(np.nan_to_num(ecart_rel))) if len(ecart_rel) else 0
anomalie_max_pct = float(np.nan_to_num(ecart_rel)[idx_pic])
t_pic = times[idx_pic]

if snr >= 5 and n_gates_valides >= 3:
    verdict_snr, couleur_snr = "Détectable", "green"
elif snr >= 2 and n_gates_valides >= 1:
    verdict_snr, couleur_snr = "Marginal", "orange"
else:
    verdict_snr, couleur_snr = "Non détectable / noyé dans le bruit", "red"

res_moyenne = float(np.mean([res_recouvrement, res_substratum]))
doi = doi_spies(res_moyenne, moment_tx, float(np.min(bruit_v)), aire_eff_rx)

if karst_actif and (z_toit + ep_karst) > doi:
    st.sidebar.warning(
        f"⚠️ La base de la cible ({z_toit + ep_karst:.1f} m) dépasse la profondeur "
        f"d'investigation estimée ({doi:.1f} m)."
    )

# --- ZONE D'AFFICHAGE ET POP-UP THÉORIQUE ---
col_title, col_help = st.columns([0.85, 0.15])

with col_help:
    with st.popover("📖 Résumé Théorique"):
        st.markdown("### Fondamentaux du Module")
        st.subheader("1. Principe du TDEM")
        st.write(
            "Coupure brutale du courant primaire → courants de Foucault induits dans le sol → "
            "champ secondaire transitoire mesuré au récepteur. Les temps courts renseignent "
            "le proche, les temps longs le profond (diffusion type « smoke ring »).")
        st.latex(r"\varepsilon(t) = -\,n_{rx}\,A_{rx}\,\frac{\partial B_z(t)}{\partial t}")
        st.subheader("2. Moment d'émission")
        st.latex(r"M = I \cdot A_{tx} \cdot n_{tx}")
        st.subheader("3. Anomalie relative")
        st.latex(r"a(t) = \frac{\left|\varepsilon_{cible}(t) - \varepsilon_{ref}(t)\right|}"
                 r"{\left|\varepsilon_{ref}(t)\right|} \times 100")
        st.subheader("4. Détectabilité (SNR)")
        st.latex(r"SNR(t) = \frac{\left|\Delta\varepsilon(t)\right|}{\eta(t)}"
                 r"\qquad \eta(t) = \eta_{1ms}\left(\frac{t}{1\,ms}\right)^{-p}")
        st.write("SNR ≥ 5 sur ≥ 3 gates : détectable · SNR ≥ 2 : marginal · sinon : noyé dans le bruit.")
        st.subheader("5. Profondeur d'investigation (Spies, 1989)")
        st.latex(r"DOI \approx 0.55 \left(\frac{M\,\rho}{\eta}\right)^{1/5}")
        st.caption("Approximation empirique — à recouper avec une DOI par sensibilité cumulée "
                   "(Christiansen & Auken, 2012) si l'inversion est faite sous AarhusInv.")
        st.subheader("6. Sensibilité conducteur / résistant")
        st.write("Le TDEM répond fortement aux cibles conductrices et faiblement aux cibles "
                 "résistantes : un vide d'air produit une anomalie bien plus faible qu'un "
                 "remplissage argileux de même géométrie.")

# --- AFFICHAGE DES RÉSULTATS (KPIs) ---
col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
col_stat1.metric("Anomalie relative max", f"{anomalie_max_pct:.1f} %")
col_stat2.metric("Gate du pic d'anomalie", f"{t_pic * 1e6:.0f} µs")
col_stat3.metric("SNR max", f"{snr:.1f}", delta=f"{n_gates_valides} gates exploitables")
col_stat4.metric("DOI estimée", f"{doi:.1f} m")
col_stat5.markdown(f"**Verdict**")
col_stat5.markdown(f":{couleur_snr}[{verdict_snr}]")

# --- GRAPHIQUES ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Courbes de décroissance")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(times * 1e6, np.abs(sig_sans) * FACTOR_NV, label="Modèle de référence (sans cible)",
              color="#3498db", linewidth=2)
    ax.loglog(times * 1e6, np.abs(sig_avec) * FACTOR_NV, label="Modèle avec cible",
              color="#e67e22", linewidth=2, linestyle="--", marker="o", markersize=4)
    ax.loglog(times * 1e6, bruit_v * FACTOR_NV, label="Niveau de bruit",
              color="gray", linewidth=1.5, linestyle=":")
    ax.fill_between(times * 1e6, 1e-6, bruit_v * FACTOR_NV, color="gray", alpha=0.15,
                    label="Domaine non exploitable")

    ax.set_xlabel("Temps après coupure (µs)")
    ax.set_ylabel("Tension induite |ε| (nV)")
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    ax.set_ylim(bottom=max(np.min(bruit_v * FACTOR_NV) * 0.1, 1e-6))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize="small", frameon=False)
    fig.subplots_adjust(bottom=0.3)

    st.pyplot(fig)

    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Décroissance", data=buf1.getvalue(),
                       file_name="tdem_decroissance.png", mime="image/png",
                       use_container_width=True)

with col2:
    st.subheader("Modèle de résistivité")
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    prof_max_affichage = max(40.0, (z_toit + ep_karst) * 1.5, ep_recouvrement * 1.5)
    interfaces = list(dep_cib) + [prof_max_affichage]
    couleurs = plt.cm.viridis_r(
        np.log10(np.clip(res_cib[1:], 1, 1e4)) / 4.0)

    for i in range(len(res_cib) - 1):
        haut = interfaces[i]
        bas = interfaces[i + 1]
        ax2.add_patch(Rectangle((0, haut), 1, bas - haut, facecolor=couleurs[i],
                                edgecolor="black", linewidth=1.2, zorder=2))
        ax2.text(1.05, (haut + bas) / 2, f"{res_cib[i + 1]:.0f} Ω·m",
                 va="center", fontsize=10, fontweight="bold")

    if karst_actif:
        ax2.add_patch(Rectangle((0, z_toit), 1, ep_karst, facecolor="none",
                                edgecolor="red", linewidth=2.5, linestyle="--", zorder=5))
        ax2.text(0.5, z_toit + ep_karst / 2, "CIBLE", ha="center", va="center",
                 color="red", fontweight="bold", fontsize=11, zorder=6)

    ax2.axhline(0, color="#5d4037", linewidth=3, zorder=4)
    ax2.axhline(doi, color="#c0392b", linewidth=1.5, linestyle="-.", zorder=4)
    ax2.text(0.02, doi - 1, f"DOI ≈ {doi:.0f} m", color="#c0392b", fontsize=9, fontweight="bold")

    # dispositif en surface
    ax2.add_patch(Polygon([[0.3, 0], [0.25, -prof_max_affichage * 0.03],
                           [0.35, -prof_max_affichage * 0.03]],
                          closed=True, facecolor="#e67e22", zorder=6))
    ax2.add_patch(Polygon([[0.7, 0], [0.65, -prof_max_affichage * 0.03],
                           [0.75, -prof_max_affichage * 0.03]],
                          closed=True, facecolor="#2c3e50", zorder=6))
    ax2.text(0.3, -prof_max_affichage * 0.05, "TX", ha="center", fontsize=9, fontweight="bold")
    ax2.text(0.7, -prof_max_affichage * 0.05, "RX", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xlim(0, 1.6)
    ax2.set_ylim(prof_max_affichage, -prof_max_affichage * 0.08)
    ax2.set_xticks([])
    ax2.set_ylabel("Profondeur (m)")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    st.pyplot(fig2)

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
    st.download_button(label="📸 Snapshot Modèle 1D", data=buf2.getvalue(),
                       file_name="tdem_modele.png", mime="image/png",
                       use_container_width=True)

# --- FIGURE CLÉ : ANOMALIE RELATIVE VS BRUIT ---
st.subheader("Anomalie relative et seuil de détectabilité")
st.caption("Figure de justification méthodologique : l'anomalie est exploitable là où elle "
           "dépasse simultanément le bruit absolu et le plancher de bruit relatif.")

fig3, ax3 = plt.subplots(figsize=(12, 4.5))
ax3.semilogx(times * 1e6, np.nan_to_num(ecart_rel), color="#8e44ad", linewidth=2.5,
             marker="o", markersize=5, label="Anomalie relative")
ax3.axhline(bruit_relatif_pct, color="red", linestyle="--", linewidth=1.8,
            label=f"Plancher de bruit relatif ({bruit_relatif_pct:.1f} %)")
ax3.fill_between(times * 1e6, 0, bruit_relatif_pct, color="red", alpha=0.08)

if n_gates_valides > 0:
    ax3.fill_between(times * 1e6, 0, np.nan_to_num(ecart_rel), where=gates_valides,
                     color="#27ae60", alpha=0.2, label="Gates exploitables")

ax3.axvline(t_pic * 1e6, color="#2c3e50", linestyle=":", linewidth=1.5)
ax3.annotate(f"pic : {anomalie_max_pct:.1f} % à {t_pic * 1e6:.0f} µs",
             xy=(t_pic * 1e6, anomalie_max_pct),
             xytext=(t_pic * 1e6 * 1.6, anomalie_max_pct * 0.85),
             fontsize=10, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#2c3e50"))

ax3.set_xlabel("Temps après coupure (µs)")
ax3.set_ylabel("Écart relatif (%)")
ax3.grid(True, which="both", linestyle=":", alpha=0.7)
ax3.legend(loc="upper right", fontsize="small", frameon=False)

st.pyplot(fig3)

buf3 = io.BytesIO()
fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight")
st.download_button(label="📸 Snapshot Anomalie relative", data=buf3.getvalue(),
                   file_name="tdem_anomalie.png", mime="image/png")

# --- BALAYAGES DE SENSIBILITÉ ---
with st.expander("🔎 Balayage de sensibilité (profondeur limite de détection)"):
    st.write("Calcule l'anomalie maximale en fonction de la profondeur du toit de la cible, "
             "toutes autres caractéristiques inchangées. L'intersection avec le seuil donne "
             "la profondeur limite de détection pour ce dispositif et cette géologie.")

    col_b1, col_b2, col_b3 = st.columns(3)
    z_bal_min = col_b1.number_input("Profondeur min (m)", min_value=0.5, value=5.0, step=1.0)
    z_bal_max = col_b2.number_input("Profondeur max (m)", min_value=2.0, value=50.0, step=1.0)
    n_bal = col_b3.slider("Nombre de pas", 5, 40, 20, 1)

    if st.button("Lancer le balayage", use_container_width=True):
        z_balayage = np.linspace(z_bal_min, z_bal_max, int(n_bal))
        anomalies_max = []
        snr_max = []

        barre = st.progress(0.0)
        for i, z in enumerate(z_balayage):
            r_s, d_s = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                         False, z, ep_karst, res_karst)
            r_a, d_a = construire_modele(res_recouvrement, ep_recouvrement, res_substratum,
                                         True, z, ep_karst, res_karst)
            s_s = reponse_tdem(r_s, d_s, times, offset, courant,
                               aire_tx, tours_tx, aire_rx, tours_rx, config)
            s_a = reponse_tdem(r_a, d_a, times, offset, courant,
                               aire_tx, tours_tx, aire_rx, tours_rx, config)
            anomalies_max.append(float(np.nanmax(np.nan_to_num(anomalie_relative(s_a, s_s)))))
            snr_max.append(float(np.nanmax(np.abs(s_a - s_s) / bruit_v)))
            barre.progress((i + 1) / len(z_balayage))
        barre.empty()

        anomalies_max = np.array(anomalies_max)
        snr_max = np.array(snr_max)

        sous_seuil = np.where(anomalies_max < bruit_relatif_pct)[0]
        z_limite = z_balayage[sous_seuil[0]] if len(sous_seuil) else None

        fig4, ax4 = plt.subplots(figsize=(12, 4.5))
        ax4.plot(z_balayage, anomalies_max, color="#8e44ad", linewidth=2.5, marker="o",
                 markersize=5, label="Anomalie relative max")
        ax4.axhline(bruit_relatif_pct, color="red", linestyle="--", linewidth=1.8,
                    label=f"Seuil de détectabilité ({bruit_relatif_pct:.1f} %)")
        ax4.axvline(z_toit, color="#27ae60", linestyle=":", linewidth=2,
                    label=f"Cible étudiée ({z_toit:.1f} m)")
        if z_limite is not None:
            ax4.axvline(z_limite, color="#c0392b", linestyle="-.", linewidth=2,
                        label=f"Profondeur limite ≈ {z_limite:.1f} m")
        ax4.set_xlabel("Profondeur du toit de la cible (m)")
        ax4.set_ylabel("Anomalie relative max (%)")
        ax4.set_yscale("log")
        ax4.grid(True, which="both", linestyle=":", alpha=0.7)
        ax4.legend(fontsize="small", frameon=False)

        st.pyplot(fig4)

        if z_limite is not None:
            st.success(f"Profondeur limite de détection estimée : **{z_limite:.1f} m** "
                       f"(cible d'épaisseur {ep_karst:.1f} m à {res_karst:.0f} Ω·m).")
        else:
            st.success("La cible reste détectable sur toute la gamme de profondeurs testée.")

        buf4 = io.BytesIO()
        fig4.savefig(buf4, format="png", dpi=300, bbox_inches="tight")
        st.download_button(label="📸 Snapshot Balayage", data=buf4.getvalue(),
                           file_name="tdem_balayage.png", mime="image/png")

# --- RÉSISTIVITÉ APPARENTE (INDICATIF) ---
with st.expander("📉 Résistivité apparente (indicatif — temps tardifs)"):
    st.caption("Approximation asymptotique aux temps tardifs en boucle centrale : valeur "
               "indicative pour la lecture, non utilisable telle quelle en interprétation "
               "quantitative, surtout en configuration offset.")

    rho_sans = resistivite_apparente_tardive(sig_sans, times, moment_tx, aire_eff_rx)
    rho_avec = resistivite_apparente_tardive(sig_avec, times, moment_tx, aire_eff_rx)

    fig5, ax5 = plt.subplots(figsize=(12, 4.5))
    ax5.loglog(times * 1e6, rho_sans, color="#3498db", linewidth=2, label="Sans cible")
    ax5.loglog(times * 1e6, rho_avec, color="#e67e22", linewidth=2, linestyle="--",
               marker="o", markersize=4, label="Avec cible")
    ax5.axhline(res_recouvrement, color="gray", linestyle=":", linewidth=1.2)
    ax5.axhline(res_substratum, color="gray", linestyle=":", linewidth=1.2)
    ax5.set_xlabel("Temps après coupure (µs)")
    ax5.set_ylabel("ρ apparente (Ω·m)")
    ax5.grid(True, which="both", linestyle=":", alpha=0.7)
    ax5.legend(fontsize="small", frameon=False)

    st.pyplot(fig5)
