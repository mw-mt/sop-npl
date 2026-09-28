"""
Gemeinsames Stil-Modul für alle SoP-Visualisierungsskripte
=============================================================
Zentrale Quelle für Farben, Schriftgrössen und Speicher-Einstellungen, damit
alle erzeugten Diagramme (Kategorie-Charts, Wordclouds, Cosine-Similarity-
Heatmaps) einheitlich aussehen. Wird importiert von:
    - visualize_categories_v2.py
    - generate_wordclouds.py
    - cosine_similarity_sites.py (und darüber auch sentence_cosine_similarity.py)

Farblogik
---------
SITE_COLORS:
    Eine feste Farbe pro Standort. Die 10 Standorte bilden 5 Paare gleichen
    Landschaftstyps (siehe SITE_ORDER in visualize_categories_v2.py:
    Oeschinensee/Seealpsee, River Thur/River Reuss, Robenhuserriet/Ägeriried,
    Ufschötti/Zürichhorn, Lägern/Pfannenstiel). Jedes Paar bekommt denselben
    Grundfarbton in zwei Helligkeitsstufen (dunkel/hell), damit man Paare auf
    einen Blick erkennt, ohne dass die 10 Farben insgesamt schwerer
    unterscheidbar werden. Die 5 Grundtöne sind an die Okabe-Ito-Palette
    angelehnt (Okabe & Ito 2008 — eine der Standard-Referenzpaletten für
    Farbfehlsichtigkeits-sichere qualitative Farbgebung), mit reduzierter
    Sättigung für einen gedeckteren Gesamteindruck.

CATEGORY_COLORS:
    Eigene Palette für die 14 SoP-Kategorien (aktuell nur in den
    Wordclouds verwendet), bewusst mit anderen Grundfarbtönen als
    SITE_COLORS (damit Kategorie-Farbe nicht mit Standort-Farbe verwechselt
    wird), aber gleicher "Vibe" (ähnliche Sättigung/Helligkeit, gedeckt/
    pastell). Die Zuordnung Kategorie→Farbton ist bewusst verschränkt
    (nicht in Hue-Reihenfolge), damit in CATEGORY_ORDER benachbarte
    Kategorien in Legenden maximal unterschiedliche Töne bekommen.

HEATMAP_CMAP:
    Gedeckte sequentielle Farbskala (ersetzt viridis) für alle Heatmaps
    (Verteilungs-Heatmaps 9c/9d, Cosine-Similarity-Heatmaps 12/13/14/15).
    Verlauf von cremeweiss zu dunklem Blau — derselbe Blauton wie
    SITE_COLORS["oeschinensee"] (Mountain-Lakes-Familie), damit Heatmaps und
    Balkendiagramme farblich zusammengehören. WICHTIG: die Skala geht
    HELL→DUNKEL (niedrig→hoch) — umgekehrt zu viridis (dunkel→hell). Für
    Zellbeschriftungen daher heatmap_text_color() verwenden, nicht den
    alten viridis-Schwellenwert.

Schriftgrössen
--------------
RCPARAMS: Titel max. 16pt, sonst proportional gestuft. Wird per
    plt.rcParams.update(RCPARAMS) beim Start jedes Skripts angewendet;
    einzelne Plot-Funktionen dürfen für Detailelemente (z.B. Subplot-Titel
    in Grids, Zellbeschriftungen) davon abweichende, aber konsistente
    explizite fontsize-Werte setzen (siehe FONT_SIZES).
"""

import colorsys
from pathlib import Path
import matplotlib.colors as mcolors


# ── Schriftgrössen ────────────────────────────────────────────────────────────

RCPARAMS = {
    "font.size":             12,
    "axes.titlesize":        16,
    "axes.labelsize":        14,
    "xtick.labelsize":       12,
    "ytick.labelsize":       12,
    "legend.fontsize":       12,
    "legend.title_fontsize": 13,
    "figure.titlesize":      16,
}

# Explizite Grössen für Detailelemente, die rcParams nicht abdeckt
# (Subplot-Titel in Grids, Zellbeschriftungen, Balkenbeschriftungen etc.),
# konsistent zur RCPARAMS-Hierarchie oben (Titel <= 16).
FONT_SIZES = {
    "suptitle":       16,
    "subplot_title":  14,
    "bar_label":      10,
    "heatmap_cell":   11,
    "cbar_label":     13,
    "cbar_ticks":     12,
}


# ── Speicher-Einstellungen ─────────────────────────────────────────────────────

DPI = 150


def save_figure(fig, output_path, dpi: int = DPI) -> None:
    """Speichert eine Figure einheitlich (dpi + bbox_inches='tight') und schliesst sie."""
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── Standort-Palette (5 Landschaftstyp-Paare, colorblind-orientiert) ──────────
# Herleitung s. Docstring oben. Werte einmalig berechnet (Okabe-Ito-Ankerfarbe
# je Landschaftstyp, ±Helligkeit für das Paar, Sättigung *0.85) und als
# konkrete Hex-Werte fixiert, damit die Palette stabil bleibt.

SITE_COLORS = {
    "oeschinensee":           "#0A557F",  # Mountain Lakes, dunkel
    "seealpsee":              "#1088CA",  # Mountain Lakes, hell
    "thur_thurauen":          "#9F4E0D",  # Rivers, dunkel
    "reuss_bremgarten":       "#EB7213",  # Rivers, hell
    "robenhuserriet":         "#096C51",  # Moors, dunkel
    "aegeriried":             "#0FB88A",  # Moors, hell
    "ufschoetti":             "#AF8F0E",  # Urban Lakes, dunkel
    "zuerichhorn":            "#EDC422",  # Urban Lakes, hell
    "hochwacht_laegern":      "#B96393",  # Hills, dunkel
    "hochwacht_pfannenstiel": "#D29BBA",  # Hills, hell
}

# Landschaftstyp je Standort (für evtl. spätere Legenden-Gruppierung)
SITE_LANDSCAPE_TYPE = {
    "oeschinensee":           "Mountain Lakes",
    "seealpsee":              "Mountain Lakes",
    "thur_thurauen":          "Rivers",
    "reuss_bremgarten":       "Rivers",
    "robenhuserriet":         "Moors",
    "aegeriried":             "Moors",
    "ufschoetti":             "Urban Lakes",
    "zuerichhorn":            "Urban Lakes",
    "hochwacht_laegern":      "Hills",
    "hochwacht_pfannenstiel": "Hills",
}


# ── Kategorie-Palette (14 Farben) ──────────────────────────────────────────────
# Wiederverwendet grösstenteils die SITE_COLORS-Töne (10 von 14) plus 2 neue
# Farbfamilien (Crimson, Indigo), die dieselben 4 restlichen Kategorien
# abdecken. Zuordnung so gewählt, dass keine zwei in CATEGORY_ORDER benachbarten
# Kategorien denselben Grundton teilen — und v.a. dass die 2 grünlichen Töne
# (Erlebnisintensität, Bedrohung und Besorgnis) weit auseinander liegen und
# nicht mit weiteren Grüntönen verwechselt werden können (frühere Version
# hatte 5 zu ähnliche Grüntöne, das war die reine Kategorie-Palette ohne
# Standort-Wiederverwendung).

CATEGORY_COLORS = {
    "Ästhetische Qualität":           "#0A557F",  # = SITE_COLORS Mountain Lakes dunkel
    "Überwältigung und Staunen":      "#9F4E0D",  # = SITE_COLORS Rivers dunkel
    "Erlebnisintensität":             "#096C51",  # = SITE_COLORS Moors dunkel
    "Wert und Bedeutsamkeit":         "#B96393",  # = SITE_COLORS Hills dunkel
    "Wildheit und Ursprünglichkeit":  "#AF8F0E",  # = SITE_COLORS Urban Lakes dunkel
    "Naturverbundenheit":             "#922A37",  # neu: Crimson dunkel
    "Erholung und Entschleunigung":   "#3F2A92",  # neu: Indigo dunkel
    "Ruhe und Stille":                "#EB7213",  # = SITE_COLORS Rivers hell
    "Persönliche emotionale Bindung": "#D29BBA",  # = SITE_COLORS Hills hell
    "Allgemeine positive Emotion":    "#C94555",  # neu: Crimson hell
    "Soziale Dimension":              "#EDC422",  # = SITE_COLORS Urban Lakes hell
    "Kollektive Wertschätzung":       "#1088CA",  # = SITE_COLORS Mountain Lakes hell
    "Konflikt durch Popularität":     "#6045C9",  # neu: Indigo hell
    "Bedrohung und Besorgnis":        "#0FB88A",  # = SITE_COLORS Moors hell
}


# ── Heatmap-Farbskala (ersetzt viridis) ───────────────────────────────────────

_HEATMAP_LOW  = "#F7F3EC"        # cremeweiss (niedrige Werte)
_HEATMAP_MID  = "#8FB8D6"
_HEATMAP_HIGH = SITE_COLORS["oeschinensee"]  # dunkles Blau (hohe Werte)

HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "sop_muted_blues", [_HEATMAP_LOW, _HEATMAP_MID, _HEATMAP_HIGH]
)


def heatmap_text_color(value: float, vmax: float) -> str:
    """
    Textfarbe für Zellbeschriftungen in HEATMAP_CMAP-Heatmaps.
    HEATMAP_CMAP geht hell->dunkel (niedrig->hoch) — umgekehrt zu viridis!
    Niedrige Werte (heller Hintergrund) -> schwarzer Text.
    Hohe Werte (dunkler Hintergrund) -> weisser Text.
    """
    if vmax <= 0:
        return "black"
    return "white" if value > vmax * 0.6 else "black"


# ── Wordcloud-Schriftart ──────────────────────────────────────────────────────
# Segoe UI: moderne, serifenlose System-Schrift (auf Windows Standard).

WORDCLOUD_FONT_PATH = r"C:\Windows\Fonts\segoeui.ttf"


def get_wordcloud_font_path() -> str | None:
    """Gibt den Font-Pfad zurück falls vorhanden, sonst None (WordCloud-Default)."""
    if Path(WORDCLOUD_FONT_PATH).exists():
        return WORDCLOUD_FONT_PATH
    print(f"  ⚠ Schriftart nicht gefunden: {WORDCLOUD_FONT_PATH} — verwende WordCloud-Standard")
    return None
