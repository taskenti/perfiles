# ═══════════════════════════════════════════════════════════════════════════════
#  GPX ALTIMETRY STUDIO PRO  ·  v3.2
#  Features: HTML embed, slope gradient fill, social sizes, slope subgraph,
#  sector table, km bands, WP shortcode, danger zones, 3D shadow, ITRA score,
#  dual GPX compare, mini-map, auto-pass + auto-villages detection,
#  GeoJSON export, batch ZIP, save custom presets
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import gpxpy
import gpxpy.gpx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import io, re, textwrap, hashlib, json, zipfile, os, math
from pathlib import Path

import plotly.graph_objects as go
import plotly.subplots as psubplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPX Altimetry Studio Pro",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Fuentes con display=swap para no bloquear render aunque la red sea lenta */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', system-ui, sans-serif; }
h1 { font-weight: 800; letter-spacing: -0.03em; }
.stMetric { background: #f8fafc; border-radius: 10px; padding: 12px 16px; border: 1px solid #e2e8f0; }
.stMetric label { font-size: 0.72rem !important; color: #64748b !important; text-transform: uppercase; letter-spacing:.06em; }
div[data-testid="stSidebar"] { background: #0c1220; }
div[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
div[data-testid="stSidebar"] h1,div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-weight:700; }
.stButton button { border-radius: 8px; font-weight: 600; letter-spacing:.02em; }
.wp-chip { display:inline-flex;align-items:center;gap:6px;background:#f1f5f9;
           border:1px solid #e2e8f0;border-radius:20px;padding:4px 12px;
           font-size:.8rem;color:#334155;margin:3px; }
.sec { font-size:.95rem;font-weight:700;color:#0f172a;border-left:3px solid #ef4444;
       padding-left:10px;margin:16px 0 8px 0; }
.badge-green  { background:#dcfce7;color:#166534;border-radius:6px;padding:2px 8px;font-size:.78rem;font-weight:600; }
.badge-yellow { background:#fef9c3;color:#854d0e;border-radius:6px;padding:2px 8px;font-size:.78rem;font-weight:600; }
.badge-orange { background:#ffedd5;color:#9a3412;border-radius:6px;padding:2px 8px;font-size:.78rem;font-weight:600; }
.badge-red    { background:#fee2e2;color:#991b1b;border-radius:6px;padding:2px 8px;font-size:.78rem;font-weight:600; }
/* Shortcode box usa monospace del sistema — sin fuente web extra */
.shortcode-box { background:#1e293b;color:#7dd3fc;
                 font-family:ui-monospace,'Cascadia Code','Fira Code',monospace;
                 font-size:.8rem;padding:10px 14px;border-radius:8px;word-break:break-all; }
/* MIDE */
.mide-card { background:#f8fafc;border:2px solid #e2e8f0;border-radius:14px;
             padding:18px 22px;margin:10px 0; }
.mide-pill { display:inline-block;border-radius:50%;width:36px;height:36px;line-height:36px;
             text-align:center;font-weight:800;font-size:1rem;color:#fff;margin:3px; }
.mide-row  { display:flex;align-items:center;gap:12px;margin:6px 0; }
.mide-label{ font-size:.75rem;text-transform:uppercase;letter-spacing:.07em;
             color:#64748b;font-weight:600;min-width:110px; }
.mide-sym  { background:#1e293b;color:#f8fafc;border-radius:4px;
             padding:2px 7px;font-size:.85rem;font-weight:700;margin:2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WAYPOINT_DEFS = {
    "📍 Genérico":        {"emoji":"📍","marker":"o", "color":"#EF4444","edge":"#991B1B","size":12},
    "💧 Fuente":          {"emoji":"💧","marker":"o", "color":"#06B6D4","edge":"#0E7490","size":12},
    "🏠 Refugio":         {"emoji":"🏠","marker":"s", "color":"#92400E","edge":"#451A03","size":13},
    "🏘️ Pueblo":          {"emoji":"🏘️","marker":"h", "color":"#F97316","edge":"#9A3412","size":14},
    "🌉 Puente":          {"emoji":"🌉","marker":"d", "color":"#64748B","edge":"#334155","size":12},
    "🥪 Avituallamiento": {"emoji":"🥪","marker":"P", "color":"#22C55E","edge":"#15803D","size":13},
    "📷 Foto":            {"emoji":"📷","marker":"p", "color":"#A855F7","edge":"#6B21A8","size":12},
    "🚩 Cima":            {"emoji":"🚩","marker":"^", "color":"#DC2626","edge":"#7F1D1D","size":14},
    "⛰️ Puerto":          {"emoji":"⛰️","marker":"D", "color":"#64748B","edge":"#1E293B","size":13},
    "⚠️ Peligro":         {"emoji":"⚠️","marker":"X", "color":"#FBBF24","edge":"#B45309","size":13},
    "🅿️ Parking":         {"emoji":"🅿️","marker":"s", "color":"#3B82F6","edge":"#1E40AF","size":12},
    "🌲 Bosque":          {"emoji":"🌲","marker":"^", "color":"#16A34A","edge":"#14532D","size":12},
    "⛪ Iglesia":          {"emoji":"⛪","marker":"P", "color":"#8B5CF6","edge":"#5B21B6","size":13},
    "🏔️ Mirador":         {"emoji":"🏔️","marker":"*", "color":"#F59E0B","edge":"#92400E","size":15},
    "🚰 Punto de Agua":   {"emoji":"🚰","marker":"o", "color":"#0EA5E9","edge":"#0369A1","size":12},
    "🏥 Primeros Aux.":   {"emoji":"🏥","marker":"+", "color":"#F43F5E","edge":"#9F1239","size":14},
}

# Paleta de pendiente: 4 categorías limpias, colores 2026
SLOPE_PALETTE = [
    (-99,  3, "#22C55E"),   # llano / bajadas → verde
    (  3,  8, "#F59E0B"),   # moderado → ámbar
    (  8, 18, "#EF4444"),   # empinado → rojo
    ( 18, 99, "#7C3AED"),   # extremo  → violeta oscuro
]

# Ventana de suavizado para el heatmap de pendiente (evita "arcoíris")
_SLOPE_SMOOTH_W = 15   # puntos a promediar antes de clasificar color

def _smoothed_slopes(dist: np.ndarray, ele: np.ndarray) -> np.ndarray:
    """Pendiente suavizada por ventana móvil para coloreado limpio."""
    raw = slopes_array(dist, ele)
    w   = min(_SLOPE_SMOOTH_W, max(3, len(raw) // 50))
    return pd.Series(raw).rolling(window=w, center=True, min_periods=1).mean().to_numpy()


SOCIAL_SIZES = {
    "Web / og:image (1200×630)": (1200, 630),
    "Instagram cuadrado (1080×1080)": (1080, 1080),
    "Twitter/X banner (1200×400)": (1200, 400),
    "Cabecera web (1400×350)": (1400, 350),
    "Personalizado": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# PURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hex_to_rgba(h: str, a: float) -> str:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return f"rgba(239,68,68,{a})"
    try:
        r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
        return f"rgba({r},{g},{b},{a})"
    except ValueError:
        return f"rgba(239,68,68,{a})"


def file_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def repair_gpx(s: str) -> str:
    return re.sub(r'(</?)[a-zA-Z0-9]+:(?!gpxtpx|gpxx|ns3)', r'\1', s)


def slope_color(s: float) -> str:
    for lo, hi, col in SLOPE_PALETTE:
        if lo <= s < hi:
            return col
    return "#22C55E"


def slopes_array(dist: np.ndarray, ele: np.ndarray) -> np.ndarray:
    dd = np.diff(dist) * 1000
    de = np.diff(ele)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(dd > 0, (de / dd) * 100, 0.0)
    return np.clip(s, -40, 40)


def compute_slope_colors_arr(dist: np.ndarray, ele: np.ndarray) -> np.ndarray:
    return np.array([slope_color(s) for s in slopes_array(dist, ele)])


# ─── FEATURE 2: Gradient fill bajo la curva (polígonos por segmento) ───────
def draw_slope_gradient_fill(ax, dist: np.ndarray, ele: np.ndarray,
                              base: float, alpha: float = 0.82):
    """Rellena el área bajo la curva con colores de pendiente (estilo Komoot)."""
    slopes = slopes_array(dist, ele)
    for i in range(len(dist) - 1):
        col = slope_color(slopes[i])
        xs = [dist[i], dist[i+1], dist[i+1], dist[i]]
        ys = [ele[i], ele[i+1], base, base]
        ax.fill(xs, ys, color=col, alpha=alpha, linewidth=0, zorder=2)


# ─── FEATURE 5: Tabla de sectores ────────────────────────────────────────────
def compute_sectors(dist: np.ndarray, ele: np.ndarray,
                    n_sectors: int = 10) -> pd.DataFrame:
    """Divide la ruta en N sectores y calcula stats por sector."""
    total = dist[-1]
    sector_len = total / n_sectors
    rows = []
    for i in range(n_sectors):
        km_start = i * sector_len
        km_end   = (i + 1) * sector_len
        mask = (dist >= km_start) & (dist <= km_end)
        if mask.sum() < 2:
            continue
        seg_ele  = ele[mask]
        seg_dist = dist[mask]
        diffs    = np.diff(seg_ele)
        d_plus   = float(diffs[diffs > 0.5].sum())
        d_minus  = float(abs(diffs[diffs < -0.5].sum()))
        length   = float(seg_dist[-1] - seg_dist[0])
        avg_slp  = (seg_ele[-1] - seg_ele[0]) / (length * 1000) * 100 if length > 0 else 0
        difficulty = ("🟢 Fácil" if abs(avg_slp) < 3
                      else "🟡 Moderado" if abs(avg_slp) < 7
                      else "🟠 Difícil" if abs(avg_slp) < 12
                      else "🔴 Extremo")
        rows.append({
            "Sector": f"{i+1}",
            "Km inicio": f"{km_start:.1f}",
            "Km fin":    f"{km_end:.1f}",
            "Longitud":  f"{length:.2f} km",
            "D+":        f"{d_plus:.0f} m",
            "D-":        f"{d_minus:.0f} m",
            "Pend. media": f"{avg_slp:.1f}%",
            "Dificultad": difficulty,
        })
    return pd.DataFrame(rows)


# ─── FEATURE 8: Zonas de peligro automáticas ────────────────────────────────
def detect_danger_zones(dist: np.ndarray, ele: np.ndarray,
                        threshold: float = 15.0, min_len_km: float = 0.1):
    """Detecta segmentos con pendiente > threshold%. Devuelve lista de (km_start, km_end)."""
    slopes = slopes_array(dist, ele)
    zones, in_zone, z_start = [], False, 0.0
    for i, s in enumerate(slopes):
        if abs(s) >= threshold and not in_zone:
            in_zone, z_start = True, dist[i]
        elif abs(s) < threshold and in_zone:
            if dist[i] - z_start >= min_len_km:
                zones.append((z_start, dist[i]))
            in_zone = False
    if in_zone and dist[-1] - z_start >= min_len_km:
        zones.append((z_start, dist[-1]))
    return zones


# ─── FEATURE 10: ITRA Score ──────────────────────────────────────────────────
def compute_itra(dist_km: float, gain_m: float, loss_m: float) -> dict:
    """
    Calcula el Effort Distance ITRA (ED) y puntuación aproximada.
    ED = dist + gain/100 + loss/200  (fórmula oficial ITRA)
    """
    ed = dist_km + gain_m / 100 + loss_m / 200
    # Clasificación ITRA por ED
    if ed < 25:    cat = ("XS", "badge-green")
    elif ed < 45:  cat = ("S",  "badge-green")
    elif ed < 75:  cat = ("M",  "badge-yellow")
    elif ed < 115: cat = ("L",  "badge-yellow")
    elif ed < 160: cat = ("XL", "badge-orange")
    else:          cat = ("XXL","badge-red")
    # Puntos ITRA (0-1000 aprox lineal)
    points = min(1000, int(ed * 4.5))
    return {"ed": round(ed, 1), "category": cat[0], "badge": cat[1], "points": points}


# ─── MÓDULO MIDE ─────────────────────────────────────────────────────────────
# Sistema de valoración de excursiones del montañismo español (MIDE)
# Fuente: Gobierno de Aragón / PRAMES

MIDE_TERRAIN_SPEEDS = {
    "Pista / camino": 5.0,
    "Senda":          4.0,
    "Terreno abrupto / sin camino": 3.0,
}

MIDE_RISK_FACTORS = [
    "Exposición a desprendimientos o aludes",
    "Pasos que requieren el uso de manos (III o más)",
    "Cruce de torrentes sin puente",
    "Glaciares o neveros permanentes",
    "Zona a >1h de auxilio organizado",
    "Temperaturas extremas previsibles (<-10°C o >40°C)",
    "Vientos fuertes habituales (>80 km/h)",
    "Niebla o visibilidad reducida frecuente",
    "Alta exposición en aristas o cornisas",
    "Sin cobertura móvil en >50% del recorrido",
    "Riesgo de tormenta eléctrica habitual",
    "Terreno glaciar con riesgo de grietas",
]

MIDE_ORIENTATION_OPTS = [
    ("1 – Caminos bien definidos, señalización completa",                     1),
    ("2 – Sendas marcadas, alguna pérdida de señal puntual",                  2),
    ("3 – Sendas mal definidas, orientación por mapa/GPS necesaria",          3),
    ("4 – Sin camino, orientación continua por mapa/brújula/GPS",             4),
    ("5 – Navegación interrumpida por obstáculos o terreno complejo",         5),
]

MIDE_DISPLACEMENT_OPTS = [
    ("1 – Superficie lisa, sin desnivel significativo",                        1),
    ("2 – Camino con desnivel, terreno fácil",                                 2),
    ("3 – Senda con desnivel importante o terreno irregular",                  3),
    ("4 – Terreno muy irregular, uso de manos puntual (I-II UIAA)",           4),
    ("5 – Pasos de escalada continuos (III+ UIAA) o rápel necesario",         5),
]

def compute_mide_effort(gain_m: float, loss_m: float,
                         dist_km: float, speed_kmh: float) -> dict:
    """
    Calcula el tiempo estimado MIDE y el nivel de Esfuerzo (1-5).
    Fórmula oficial MIDE (Gobierno de Aragón):
      T_h = subida/400 + bajada/600
      T_d = distancia / velocidad_terreno
      T_total = max(T_h, T_d) + min(T_h, T_d) / 2
    """
    t_h = gain_m / 400.0 + loss_m / 600.0          # horas
    t_d = dist_km / speed_kmh                        # horas
    t_total = max(t_h, t_d) + min(t_h, t_d) / 2.0  # horas

    if   t_total < 1:  effort = 1
    elif t_total < 3:  effort = 2
    elif t_total < 6:  effort = 3
    elif t_total < 10: effort = 4
    else:              effort = 5

    h = int(t_total)
    m = int(round((t_total - h) * 60))
    return {
        "t_h":     round(t_h, 2),
        "t_d":     round(t_d, 2),
        "t_total": round(t_total, 2),
        "hhmm":    f"{h}h {m:02d}min",
        "effort":  effort,
    }

def mide_medio_score(n_factors: int) -> int:
    """Convierte número de factores de riesgo en nivel de Medio (1-5)."""
    if   n_factors == 0: return 1
    elif n_factors <= 1: return 1
    elif n_factors <= 3: return 2
    elif n_factors <= 6: return 3
    elif n_factors <= 10: return 4
    else:                return 5

def mide_score_label(score: int) -> str:
    return ["", "Bajo", "Moderado", "Alto", "Muy alto", "Extremo"][score]

def mide_score_color(score: int) -> str:
    return ["", "#22C55E", "#84CC16", "#F59E0B", "#EF4444", "#7C3AED"][score]


def build_mide_figure(
    route_name: str,
    trip_type: str,
    dist_km: float, gain_m: float, loss_m: float,
    t_hhmm: str,
    medio: int, orientacion: int, desplazamiento: int, esfuerzo: int,
    sym_T: bool, sym_R: bool, sym_N: bool,
    rapel_m: float, nieve_deg: float,
    dist_arr, ele_display,
    waypoints: list,
) -> plt.Figure:
    """
    Genera la ficha MIDE oficial como figura Matplotlib (PNG/JPG/SVG).
    Reproduce el diseño de la cartografía oficial del Gobierno de Aragón.
    """
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap

    # ── Dimensiones ──────────────────────────────────────────────────────
    FW, FH = 11.5, 9.0   # pulgadas  → ~1380×1080 px a 120 dpi
    DPI     = 120
    BG      = "#E8E8E8"   # gris claro fondo exterior
    CARD    = "#F2F2F2"   # interior de la tarjeta
    WHITE   = "#FFFFFF"
    BLACK   = "#1A1A1A"
    DGRAY   = "#4A4A4A"
    LGRAY   = "#D0D0D0"
    BORDER  = "#888888"
    RED     = "#CC0000"   # color acento para alertas

    fig = plt.figure(figsize=(FW, FH), dpi=DPI, facecolor=BG)

    # Layout: header 14% | tabla 32% | perfil 54%
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.13, 0.32, 0.55],
        hspace=0.0,
        left=0.04, right=0.96, top=0.97, bottom=0.03,
    )

    ax_hdr  = fig.add_subplot(gs[0])
    ax_tbl  = fig.add_subplot(gs[1])
    ax_prof = fig.add_subplot(gs[2])

    for ax in (ax_hdr, ax_tbl, ax_prof):
        ax.set_facecolor(CARD)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
            sp.set_linewidth(1.5)

    # ════════════════════════════════════════════════════════════════════
    # HEADER — "MIDE" + nombre ruta
    # ════════════════════════════════════════════════════════════════════
    ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")
    ax_hdr.set_facecolor(CARD)

    # Rectángulo "MIDE" con borde negro grueso
    ax_hdr.add_patch(FancyBboxPatch(
        (0.01, 0.08), 0.14, 0.84,
        boxstyle="round,pad=0.01",
        facecolor=WHITE, edgecolor=BLACK, linewidth=3,
        transform=ax_hdr.transAxes, zorder=2,
    ))
    ax_hdr.text(0.08, 0.5, "MIDE",
                ha="center", va="center",
                fontsize=34, fontweight="black",
                color=BLACK, fontfamily="DejaVu Sans",
                transform=ax_hdr.transAxes, zorder=3)

    # Nombre de la ruta
    _rname = route_name or "Ruta sin nombre"
    if len(_rname) > 60: _rname = _rname[:57] + "…"
    ax_hdr.text(0.17, 0.5, _rname,
                ha="left", va="center",
                fontsize=13, fontweight="bold", color=DGRAY,
                transform=ax_hdr.transAxes)

    # ════════════════════════════════════════════════════════════════════
    # TABLA — izquierda (valoraciones) + derecha (datos de referencia)
    # ════════════════════════════════════════════════════════════════════
    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)
    ax_tbl.axis("off")

    # Línea divisoria vertical
    ax_tbl.axvline(0.50, color=BORDER, linewidth=1.2, ymin=0.0, ymax=1.0)
    # Líneas horizontales separadoras (5 filas en cada columna)
    N_ROWS = 5
    for r in range(N_ROWS + 1):
        y = r / N_ROWS
        ax_tbl.axhline(1 - y, color=LGRAY, linewidth=0.8, xmin=0.0, xmax=1.0)

    # ── Definir contenido filas izquierda ──
    # Col izq: (etiqueta, valor, símbolo_texto)
    sym_extra = ""
    if sym_T: sym_extra += "T"
    if sym_R: sym_extra += "R"
    if sym_N: sym_extra += "N"

    left_rows = [
        ("severidad del medio natural",    str(medio),        "⚠"),
        ("orientación en el itinerario",   str(orientacion),  "⊙"),
        ("dificultad en el desplazamiento",str(desplazamiento),"⚙"),
        ("cantidad de esfuerzo necesario", str(esfuerzo),     "↻"),
    ]
    # Quinta fila: simbología técnica
    if sym_T or sym_R or sym_N:
        tech_parts = []
        if sym_T: tech_parts.append(f"T (escalada)")
        if sym_R: tech_parts.append(f"R ({int(rapel_m)}m rápel)")
        if sym_N: tech_parts.append(f"N ({int(nieve_deg)}°)")
        left_rows.append(("dificultad técnica", " · ".join(tech_parts), "△"))
    else:
        left_rows.append(("dificultad técnica", "—", "△"))

    # Col derecha: (icono_texto, etiqueta, valor_bold)
    right_rows = [
        ("⊙", "horario",              t_hhmm),
        ("△", "desnivel de subida",  f"{int(gain_m):,} m".replace(",",".")),
        ("▽", "desnivel de bajada",  f"{int(loss_m):,} m".replace(",",".")),
        ("═", "distancia horizontal",f"{dist_km:.1f} km"),
        ("↺", "tipo de recorrido",   trip_type),
    ]

    ROW_H = 1.0 / N_ROWS
    for i, (lbl, val, sym) in enumerate(left_rows):
        y_c = 1.0 - (i + 0.5) * ROW_H  # centro vertical de la fila
        # Etiqueta
        ax_tbl.text(0.02, y_c, lbl,
                    ha="left", va="center", fontsize=8.5, color=BLACK)
        # Número en negrita
        ax_tbl.text(0.35, y_c, val,
                    ha="right", va="center", fontsize=11, fontweight="bold", color=BLACK)
        # Símbolo MIDE (aproximado con texto)
        ax_tbl.text(0.41, y_c, sym,
                    ha="center", va="center", fontsize=14,
                    color=DGRAY,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=LGRAY,
                              edgecolor=BORDER, linewidth=1))

    for i, (sym, lbl, val) in enumerate(right_rows):
        y_c = 1.0 - (i + 0.5) * ROW_H
        # Icono circular
        ax_tbl.text(0.53, y_c, sym,
                    ha="center", va="center", fontsize=13, color=DGRAY,
                    bbox=dict(boxstyle="circle,pad=0.25", facecolor=LGRAY,
                              edgecolor=BORDER, linewidth=1))
        # Etiqueta
        ax_tbl.text(0.57, y_c, lbl,
                    ha="left", va="center", fontsize=8.5, color=BLACK)
        # Valor en negrita
        ax_tbl.text(0.98, y_c, val,
                    ha="right", va="center", fontsize=10, fontweight="bold", color=BLACK)

    # ════════════════════════════════════════════════════════════════════
    # PERFIL ALTIMÉTRICO  — reproduce el diseño MIDE oficial
    # ════════════════════════════════════════════════════════════════════
    ax_prof.tick_params(colors=DGRAY, labelsize=8)
    for sp in ax_prof.spines.values():
        sp.set_edgecolor(BORDER); sp.set_linewidth(1.2)
    ax_prof.set_facecolor(CARD)

    # Etiqueta "perfil ────"
    ax_prof.text(0.018, 0.97, "— perfil",
                 transform=ax_prof.transAxes,
                 ha="left", va="top", fontsize=11, fontstyle="italic",
                 fontweight="bold", color=BLACK)

    # Eje X: "m 0 ... N km"
    ax_prof.set_xlabel("", labelpad=2)
    ax_prof.set_xlim(0, dist_arr[-1])

    xticks = np.arange(0, dist_arr[-1] + 0.01, 1.0)
    ax_prof.set_xticks(xticks)
    _xlabels = []
    for _x in xticks:
        if _x == 0:         _xlabels.append(f"m  0")
        elif _x == xticks[-1]: _xlabels.append(f"{int(round(_x))} km")
        else:               _xlabels.append(str(int(round(_x))))
    ax_prof.set_xticklabels(_xlabels, fontsize=8, color=DGRAY)

    # Eje Y: altitudes en múltiplos de 200
    _mn = ele_display.min()
    _mx = ele_display.max()
    _pad_p = (_mx - _mn) * 0.18
    _y_bot = _mn - _pad_p * 0.5
    _y_top = _mx + _pad_p * 2.2   # espacio para waypoint labels
    ax_prof.set_ylim(_y_bot, _y_top)

    _y200 = np.arange(int(_mn // 200) * 200, _mx + 200, 200)
    ax_prof.set_yticks(_y200)
    ax_prof.set_yticklabels([f"{int(y):,}".replace(",",".") for y in _y200],
                             fontsize=8, color=DGRAY)
    ax_prof.yaxis.set_label_coords(-0.05, 0.5)

    # Rejilla ligera
    ax_prof.set_axisbelow(True)
    ax_prof.yaxis.grid(True, color=LGRAY, linewidth=0.7, linestyle="-")
    ax_prof.xaxis.grid(True, color=LGRAY, linewidth=0.6, linestyle="-")

    # ── Relleno degradado oscuro→claro (efecto pendiente MIDE) ──
    _base = _y_bot
    _N    = len(dist_arr)
    # Usamos un mapa de grises inverso: más oscuro arriba, claro en la base
    _cmap_mide = LinearSegmentedColormap.from_list(
        "mide_fill", ["#AAAAAA", "#1A1A1A"])

    for _i in range(_N - 1):
        _x0, _x1 = dist_arr[_i], dist_arr[_i + 1]
        _e0, _e1 = ele_display[_i], ele_display[_i + 1]
        _t = (_e0 - _mn) / max(_mx - _mn, 1)  # 0=min, 1=max
        _col = _cmap_mide(_t)
        ax_prof.fill_between(
            [_x0, _x1], [_base, _base], [_e0, _e1],
            color=_col, alpha=0.85, linewidth=0, zorder=2,
        )

    # Línea del perfil encima
    ax_prof.plot(dist_arr, ele_display,
                 color=BLACK, linewidth=1.8, zorder=3,
                 solid_capstyle="round", solid_joinstyle="round")

    # ── Waypoints ──
    for wp in waypoints:
        _xi = int(np.argmin(np.abs(dist_arr - wp["km"])))
        _xe = float(dist_arr[_xi])
        _ye = float(ele_display[_xi])
        # Marcador círculo estilo MIDE
        ax_prof.plot(_xe, _ye, "o",
                     color=WHITE, markersize=7,
                     markeredgecolor=BLACK, markeredgewidth=1.8, zorder=5)
        # Línea vertical fina al eje
        ax_prof.plot([_xe, _xe], [_base, _ye],
                     color=BLACK, linewidth=0.6, linestyle="--",
                     alpha=0.4, zorder=3)
        # Texto del waypoint — dos líneas si es largo
        _lbl = wp.get("label", "")
        _lines = textwrap.wrap(_lbl, 14)
        _txt = "\n".join(_lines[:3])
        ax_prof.text(_xe, _ye + _pad_p * 0.55, _txt,
                     ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color=BLACK,
                     multialignment="center",
                     zorder=6)

    # Leyenda "punto de interés"
    ax_prof.plot([], [], "o",
                 color=WHITE, markersize=6,
                 markeredgecolor=BLACK, markeredgewidth=1.5,
                 label="punto de interés")
    ax_prof.legend(loc="upper left", fontsize=7.5, framealpha=0,
                   handlelength=1, borderpad=0.3,
                   labelcolor=DGRAY)

    fig.tight_layout(pad=0.3)
    return fig

def _smooth_for_peaks(ele: np.ndarray) -> np.ndarray:
    """Suavizado estándar para find_peaks."""
    w = max(5, len(ele) // 200)
    return pd.Series(ele).rolling(window=w, center=True, min_periods=1).mean().values


def detect_passes(dist: np.ndarray, ele: np.ndarray,
                  min_gain: float = 100, min_dist_km: float = 1.0) -> list:
    """
    Detecta máximos locales significativos (puertos/collados).
    Criterio: cimas con prominencia >= min_gain metros respecto al entorno.
    """
    from scipy.signal import find_peaks
    smooth = _smooth_for_peaks(ele)
    step_km = dist[-1] / len(dist) if len(dist) > 0 else 0.01
    min_dist_pts = max(1, int(min_dist_km / step_km))
    peaks, _ = find_peaks(smooth, prominence=min_gain, distance=min_dist_pts)
    return [
        {
            "km":       round(float(dist[p]), 2),
            "ele":      round(float(ele[p]), 1),
            "label":    f"Puerto {i+1}",
            "icon":     "⛰️",
            "icon_key": "⛰️ Puerto",
        }
        for i, p in enumerate(peaks)
    ]


def detect_villages(dist: np.ndarray, ele: np.ndarray,
                    min_drop: float = 40.0, min_dist_km: float = 1.5) -> list:
    """
    Detecta mínimos locales significativos que corresponden a valles habitados
    (pueblos, aldeas). Los pueblos suelen estar en el fondo de valles y quebradas.

    Criterio: mínimos locales donde la elevación cae al menos min_drop metros
    respecto al entorno cercano. Separados al menos min_dist_km entre sí.

    Returns lista de {'km', 'ele', 'label', 'icon', 'icon_key'}.
    """
    from scipy.signal import find_peaks
    smooth = _smooth_for_peaks(ele)
    # Invertimos para buscar mínimos como máximos del negativo
    inv = -smooth
    step_km = dist[-1] / len(dist) if len(dist) > 0 else 0.01
    min_dist_pts = max(1, int(min_dist_km / step_km))
    valleys, props = find_peaks(inv, prominence=min_drop, distance=min_dist_pts)

    # Filtrar: excluir el punto de inicio y fin (primeros/últimos 2%)
    margin = max(1, int(len(dist) * 0.02))
    valleys = valleys[(valleys > margin) & (valleys < len(dist) - margin)]

    return [
        {
            "km":       round(float(dist[v]), 2),
            "ele":      round(float(ele[v]), 1),
            "label":    f"Pueblo {i+1}",
            "icon":     "🏘️",
            "icon_key": "🏘️ Pueblo",
        }
        for i, v in enumerate(valleys)
    ]


# ─── FEATURE 14: GeoJSON export ──────────────────────────────────────────────
def build_geojson(df: pd.DataFrame, ele: np.ndarray, dist: np.ndarray) -> str:
    slopes = slopes_array(dist, ele)
    features = []
    for i in range(len(df) - 1):
        s = float(slopes[i]) if i < len(slopes) else 0.0
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(df.iloc[i]['lon']),   float(df.iloc[i]['lat']),   float(ele[i])],
                    [float(df.iloc[i+1]['lon']), float(df.iloc[i+1]['lat']), float(ele[i+1])],
                ]
            },
            "properties": {
                "slope_pct":   round(s, 2),
                "slope_color": slope_color(s),
                "km_start":    round(float(dist[i]), 3),
                "km_end":      round(float(dist[i+1]), 3),
                "ele_start":   round(float(ele[i]), 1),
                "ele_end":     round(float(ele[i+1]), 1),
            }
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def parse_gpx_cached(file_bytes: bytes):
    """Returns (df, total_km, gpx_waypoints) — cached by content hash."""
    for attempt in range(2):
        try:
            s = file_bytes.decode('utf-8', errors='ignore' if attempt else 'strict')
            if attempt:
                s = repair_gpx(s)
            gpx = gpxpy.parse(s)
            break
        except Exception:
            if attempt == 1:
                return None, 0.0, []

    points = []
    for track in gpx.tracks:
        for seg in track.segments:
            for pt in seg.points:
                points.append({'lat': pt.latitude, 'lon': pt.longitude, 'ele': pt.elevation})
    if not points:
        for route in gpx.routes:
            for pt in route.points:
                points.append({'lat': pt.latitude, 'lon': pt.longitude, 'ele': pt.elevation})
    if not points:
        return None, 0.0, []

    df = pd.DataFrame(points)
    corrupted = df['ele'].isnull().all()
    df['ele'] = df['ele'].interpolate().fillna(0)
    df['ele_corrupted'] = corrupted

    # Distancias
    dists = [0.0]
    for i in range(1, len(df)):
        d = gpxpy.geo.haversine_distance(
            df.iloc[i-1]['lat'], df.iloc[i-1]['lon'],
            df.iloc[i]['lat'],   df.iloc[i]['lon'])
        dists.append(dists[-1] + d / 1000.0)
    df['dist'] = dists
    total_km = dists[-1]

    # Decimar si >20k puntos
    if len(df) > 20_000:
        step = max(2, len(df) // 10_000)
        df = df.iloc[::step].reset_index(drop=True)
        dists2 = [0.0]
        for i in range(1, len(df)):
            d = gpxpy.geo.haversine_distance(
                df.iloc[i-1]['lat'], df.iloc[i-1]['lon'],
                df.iloc[i]['lat'],   df.iloc[i]['lon'])
            dists2.append(dists2[-1] + d / 1000.0)
        df['dist'] = dists2
        total_km = dists2[-1]

    # Waypoints nativos <wpt>
    gpx_wpts = []
    for wpt in gpx.waypoints:
        if not (wpt.latitude and wpt.longitude):
            continue
        dists_sq = ((df['lat'] - wpt.latitude)**2 + (df['lon'] - wpt.longitude)**2).values
        idx = int(np.argmin(dists_sq))
        gpx_wpts.append({
            "km":      round(float(df.iloc[idx]['dist']), 2),
            "label":   wpt.name or "Waypoint",
            "ele":     round(float(wpt.elevation or df.iloc[idx]['ele']), 1),
            "icon":    "📍", "icon_key": "📍 Genérico",
        })

    return df, total_km, gpx_wpts


@st.cache_data(show_spinner=False)
def smooth_cached(ele: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return ele.copy()
    w = window + (1 - window % 2)  # forzar impar
    return pd.Series(ele).rolling(window=w, center=True, min_periods=1).mean().to_numpy()


def calc_stats(dist: np.ndarray, ele: np.ndarray):
    diffs = np.diff(ele)
    gain  = float(diffs[diffs >  0.5].sum())
    loss  = float(abs(diffs[diffs < -0.5].sum()))
    dd    = np.diff(dist) * 1000
    with np.errstate(divide='ignore', invalid='ignore'):
        slp = np.where(dd > 0, (diffs / dd) * 100, 0.0)
    valid = slp[np.abs(slp) < 35]
    return float(ele.min()), float(ele.max()), gain, loss, (float(valid.max()) if len(valid) else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STATIC FIGURE  (shared by all export formats)
# ─────────────────────────────────────────────────────────────────────────────

def build_static_fig(
    dist_arr, ele_display, df_raw, total_km,
    min_ele, max_ele, gain, loss, max_slope,
    line_color, fill_color, bg_color, text_color,
    line_width, fill_alpha, fill_area,
    show_slope_heat, slope_fill_style,
    show_grid, show_km_markers, km_interval,
    label_rotation, aspect_ratio,
    start_loc, end_loc, chart_title,
    waypoints, show_danger_zones, danger_threshold,
    show_slope_subgraph, show_slope_legend,
    fig_w_px=None, fig_h_px=None,
    extra_top_factor=None,
):
    """Construye y devuelve el figure de matplotlib con todas las capas activas."""
    padding = (max_ele - min_ele) * 0.15

    if fig_w_px and fig_h_px:
        dpi    = 150
        fw     = fig_w_px / dpi
        fh     = fig_h_px / dpi
    else:
        fh = 5.0
        fw = fh * aspect_ratio

    if extra_top_factor is None:
        extra_top_factor = 4.5 if label_rotation == "Vertical" else 2.2

    # ── Layout: 1 o 2 filas ──
    if show_slope_subgraph:
        fig, (ax, ax_s) = plt.subplots(
            2, 1, figsize=(fw, fh),
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(fw, fh))
        ax_s = None

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # ══════════════════════════════════════════════════════
    # CAPA 1 — relleno de área (slope gradient o color plano)
    # ══════════════════════════════════════════════════════
    base_fill = min_ele - padding
    if show_slope_heat and slope_fill_style == "Relleno por pendiente (Komoot)":
        draw_slope_gradient_fill(ax, dist_arr, ele_display, base_fill, fill_alpha)
    elif fill_area:
        ax.fill_between(dist_arr, ele_display, base_fill,
                        color=fill_color, alpha=fill_alpha, zorder=2)

    # ══════════════════════════════════════════════════════
    # CAPA 2 — línea del perfil (slope coloreada o sólida)
    # ══════════════════════════════════════════════════════
    if show_slope_heat and slope_fill_style == "Línea coloreada por pendiente":
        seg_cols = [slope_color(s) for s in _smoothed_slopes(dist_arr, ele_display)]
        for i in range(len(dist_arr) - 1):
            ax.plot([dist_arr[i], dist_arr[i+1]],
                    [ele_display[i], ele_display[i+1]],
                    color=seg_cols[i], linewidth=line_width,  # usa el grosor configurado
                    solid_capstyle="round", zorder=3)
    else:
        # ── FEATURE 9: Sombra 3D bajo la línea ──
        ax.plot(dist_arr + dist_arr[-1]*0.003, ele_display - (max_ele-min_ele)*0.018,
                color="#00000022", linewidth=line_width+3, zorder=2,
                solid_capstyle="round", solid_joinstyle="round")
        ax.plot(dist_arr, ele_display,
                color=line_color, linewidth=line_width, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")

    # ══════════════════════════════════════════════════════
    # CAPA 3 — Zonas de peligro (FEATURE 8)
    # ══════════════════════════════════════════════════════
    if show_danger_zones:
        zones = detect_danger_zones(dist_arr, ele_display, threshold=danger_threshold)
        for z_start, z_end in zones:
            ax.axvspan(z_start, z_end, ymin=0, ymax=1,
                       color="#DC2626", alpha=0.13, zorder=1, linewidth=0)
            ax.text((z_start + z_end) / 2, max_ele + padding * 0.3,
                    f"⚠️ {danger_threshold:.0f}%+",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#DC2626", fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="#DC2626",
                              linewidth=0.8, pad=2, boxstyle="round,pad=0.25"),
                    zorder=5)

    # ══════════════════════════════════════════════════════
    # CAPA 4 — Marcadores de km (FEATURE 6)
    # ══════════════════════════════════════════════════════
    if show_km_markers:
        for km_mark in np.arange(km_interval, total_km, km_interval):
            idx_m = int(np.argmin(np.abs(dist_arr - km_mark)))
            xv = float(dist_arr[idx_m])
            ax.axvline(x=xv, color=text_color, linestyle=":", linewidth=0.6, alpha=0.3, zorder=1)
            # Banda alterna
            if int(km_mark // km_interval) % 2 == 0:
                x0 = float(dist_arr[max(0, int(np.argmin(np.abs(dist_arr - (km_mark - km_interval)))) )])
                ax.axvspan(x0, xv, color=text_color, alpha=0.025, zorder=0)
            # Pendiente media del sector
            mask = (dist_arr >= (km_mark - km_interval)) & (dist_arr <= km_mark)
            if mask.sum() > 1:
                seg_e = ele_display[mask]
                seg_d = dist_arr[mask]
                seg_len = (seg_d[-1] - seg_d[0]) * 1000
                avg_s = (seg_e[-1] - seg_e[0]) / seg_len * 100 if seg_len > 0 else 0
                ax.text(xv - km_interval/2, min_ele - padding * 0.55,
                        f"{avg_s:+.1f}%", ha="center", va="top",
                        fontsize=6.5, color=slope_color(avg_s), alpha=0.9, fontweight="600")
            ax.text(xv, min_ele - padding * 0.25,
                    f"{km_mark:.0f}km", ha="center", va="top",
                    fontsize=6.5, color=text_color, alpha=0.55)

    # ══════════════════════════════════════════════════════
    # CAPA 5 — Etiquetas de localidades (vertical, negro, fondo transparente)
    # ══════════════════════════════════════════════════════
    # Offset mínimo (~0.3%): solo para no pisar el eje Y, pegado a su posición real
    _x_inset   = total_km * 0.003
    _y_off_loc = padding * 0.8

    if start_loc:
        ax.text(
            _x_inset,
            float(ele_display[0]) + _y_off_loc,
            start_loc,
            ha="left", va="bottom",
            rotation=90,
            fontsize=9, fontweight="bold", color="#000000",
            bbox=dict(facecolor="none", edgecolor="none", pad=2),
            zorder=5,
        )

    if end_loc:
        ax.text(
            total_km - _x_inset,
            float(ele_display[-1]) + _y_off_loc,
            end_loc,
            ha="right", va="bottom",
            rotation=90,
            fontsize=9, fontweight="bold", color="#000000",
            bbox=dict(facecolor="none", edgecolor="none", pad=2),
            zorder=5,
        )

    # Usado por waypoints intermedios
    rotation_deg   = 90 if label_rotation == "Vertical" else 0
    y_offset_label = padding * (1.0 if label_rotation == "Vertical" else 0.5)

    # ══════════════════════════════════════════════════════
    # CAPA 6 — Waypoints (respeta atributos por-waypoint del editor)
    # ══════════════════════════════════════════════════════
    for wp in waypoints:
        # Atributos individuales (con fallback al global si no editado aún)
        wp_rot  = wp.get("rotation", label_rotation)
        wp_fs   = int(wp.get("fontsize", 8))
        wp_vpos = wp.get("vpos", "Arriba")   # "Arriba" | "Abajo"

        wp_rot_deg = 90 if wp_rot == "Vertical" else 0
        wp_y_off   = padding * (1.0 if wp_rot == "Vertical" else 0.5)
        # Posición vertical de la etiqueta
        if wp_vpos == "Abajo":
            txt_y  = wp["ele"] - wp_y_off
            txt_va = "top"
        else:
            txt_y  = wp["ele"] + wp_y_off
            txt_va = "bottom"

        # Línea vertical punteada hasta la base
        ax.plot([wp["km"], wp["km"]], [base_fill, wp["ele"]],
                color=text_color, linestyle=":", linewidth=0.9, alpha=0.35, zorder=4)

        # Marcador con estilo
        st_ = WAYPOINT_DEFS.get(wp["icon_key"], WAYPOINT_DEFS["📍 Genérico"])
        ax.plot(wp["km"], wp["ele"],
                marker=st_["marker"], color=st_["color"],
                markersize=st_["size"], markeredgecolor=st_["edge"],
                markeredgewidth=2.0, zorder=6)

        # Etiqueta con atributos individuales
        wrapped = "\n".join(textwrap.wrap(wp["label"], width=14))
        ax.text(wp["km"], txt_y, wrapped,
                ha="center", va=txt_va, rotation=wp_rot_deg,
                fontsize=wp_fs, fontweight="bold", color=text_color,
                bbox=dict(facecolor=bg_color, alpha=0.92, edgecolor="none",
                          pad=3, boxstyle="round,pad=0.35"), zorder=5)

    # ══════════════════════════════════════════════════════
    # CAPA 7 — Stats embed + título
    # ══════════════════════════════════════════════════════
    stats_txt = (f"↑ {gain:.0f}m  ↓ {loss:.0f}m  "
                 f"⬆ {max_ele:.0f}m  ⬇ {min_ele:.0f}m  "
                 f"≈ {total_km:.1f}km  max {max_slope:.1f}%")
    # Posición: dentro del área del gráfico, esquina inferior derecha,
    # por ENCIMA del eje X para no solapar etiquetas de km
    ax.text(0.99, 0.04, stats_txt, transform=ax.transAxes,
            fontsize=6.5, color=text_color, alpha=0.65,
            va="bottom", ha="right", fontfamily="monospace")

    if chart_title:
        ax.set_title(chart_title, fontsize=13, fontweight="bold",
                     color=text_color, pad=6)

    # ── Leyenda pendiente (4 categorías) ──
    if show_slope_heat and show_slope_legend:
        legend_patches = [
            mpatches.Patch(color="#22C55E", label="Llano / bajada (<3%)"),
            mpatches.Patch(color="#F59E0B", label="Moderado (3–8%)"),
            mpatches.Patch(color="#EF4444", label="Empinado (8–18%)"),
            mpatches.Patch(color="#7C3AED", label="Extremo (>18%)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
                  framealpha=0.85, edgecolor="#e2e8f0", ncol=2)

    # ══════════════════════════════════════════════════════
    # EJES PRINCIPALES
    # ══════════════════════════════════════════════════════
    ax.set_ylabel("Altitud (m)", color=text_color, fontsize=10, fontweight="bold")
    ax.tick_params(colors=text_color, labelsize=8)
    ax.set_ylim(base_fill, max_ele + padding * extra_top_factor)
    right_margin = 1.06 if end_loc else 1.02
    ax.set_xlim(0, total_km * right_margin)

    if not show_slope_subgraph:
        ax.set_xlabel("Distancia (km)", color=text_color, fontsize=10, fontweight="bold")
    else:
        ax.set_xticklabels([])

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_edgecolor(text_color)
        ax.spines[sp].set_linewidth(1.0)

    if show_grid:
        ax.grid(True, color="#9ca3af", linestyle="-", linewidth=0.6, alpha=0.4, zorder=0)

    # ══════════════════════════════════════════════════════
    # SUBGRÁFICO DE PENDIENTE (FEATURE 4)
    # ══════════════════════════════════════════════════════
    if ax_s is not None:
        ax_s.set_facecolor(bg_color)
        slps = slopes_array(dist_arr, ele_display)
        seg_mid = (dist_arr[:-1] + dist_arr[1:]) / 2
        bar_colors = [slope_color(s) for s in slps]
        ax_s.bar(seg_mid, slps, width=np.diff(dist_arr),
                 color=bar_colors, alpha=0.9, linewidth=0, align="center")
        ax_s.axhline(0, color=text_color, linewidth=0.7, alpha=0.4)
        ax_s.set_ylabel("Pend. %", color=text_color, fontsize=7)
        ax_s.set_xlabel("Distancia (km)", color=text_color, fontsize=9, fontweight="bold")
        ax_s.set_xlim(0, total_km * right_margin)
        ax_s.tick_params(colors=text_color, labelsize=7)
        for sp in ax_s.spines.values():
            sp.set_visible(False)
        ax_s.spines["bottom"].set_visible(True)
        ax_s.spines["bottom"].set_edgecolor(text_color)

    plt.tight_layout(pad=0.7)
    fig.subplots_adjust(top=0.87 if chart_title else 0.93)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: HTML INTERACTIVO EMBED
# ─────────────────────────────────────────────────────────────────────────────

def build_html_embed(dist_arr, ele_display, df_raw, total_km,
                     min_ele, max_ele, gain, loss, max_slope,
                     line_color, fill_color, bg_color, text_color,
                     line_width, fill_alpha, fill_area,
                     show_slope_heat, slope_fill_style,
                     show_grid, show_km_markers, km_interval,
                     waypoints, start_loc, end_loc, chart_title,
                     show_danger_zones, danger_threshold,
                     show_slope_subgraph, embed_width, embed_height) -> str:
    """Genera HTML autocontenido con Plotly CDN, listo para incrustar en WordPress."""

    padding = (max_ele - min_ele) * 0.15

    if show_slope_subgraph:
        fig = psubplots.make_subplots(
            rows=2, cols=1,
            row_heights=[0.78, 0.22],
            shared_xaxes=True,
            vertical_spacing=0.04,
        )
        slope_row = 2
    else:
        fig = go.Figure()
        slope_row = None

    def _add(trace, **kw):
        if show_slope_subgraph:
            fig.add_trace(trace, row=1, col=1, **kw)
        else:
            fig.add_trace(trace)

    # ── Relleno / línea ──
    if show_slope_heat and slope_fill_style == "Relleno por pendiente (Komoot)":
        slopes = slopes_array(dist_arr, ele_display)
        for i in range(len(dist_arr) - 1):
            col = slope_color(slopes[i])
            col_t = hex_to_rgba(col, fill_alpha)
            xs = [dist_arr[i], dist_arr[i+1], dist_arr[i+1], dist_arr[i], dist_arr[i]]
            ys = [ele_display[i], ele_display[i+1],
                  min_ele - padding, min_ele - padding, ele_display[i]]
            _add(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=col_t,
                            line=dict(width=0), mode="lines",
                            showlegend=False, hoverinfo="skip"))
        _add(go.Scatter(x=dist_arr.tolist(), y=ele_display.tolist(),
                        mode="lines", line=dict(color=line_color, width=line_width),
                        showlegend=False,
                        hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>"))
    elif show_slope_heat and slope_fill_style == "Línea coloreada por pendiente":
        slopes = slopes_array(dist_arr, ele_display)
        for i in range(len(dist_arr) - 1):
            col = slope_color(slopes[i])
            _add(go.Scatter(x=[dist_arr[i], dist_arr[i+1]],
                            y=[ele_display[i], ele_display[i+1]],
                            mode="lines", line=dict(color=col, width=line_width),
                            showlegend=False, hoverinfo="skip"))
    else:
        _add(go.Scatter(
            x=dist_arr.tolist(), y=ele_display.tolist(), mode="lines",
            line=dict(color=line_color, width=line_width),
            fill="tozeroy" if fill_area else "none",
            fillcolor=hex_to_rgba(fill_color, fill_alpha) if fill_area else None,
            hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>",
            showlegend=False,
        ))

    # ── Zonas de peligro ──
    if show_danger_zones:
        zones = detect_danger_zones(dist_arr, ele_display, threshold=danger_threshold)
        for z_start, z_end in zones:
            if show_slope_subgraph:
                fig.add_vrect(x0=z_start, x1=z_end,
                              fillcolor="rgba(220,38,38,0.12)", layer="below",
                              line_width=0, row=1, col=1)
            else:
                fig.add_vrect(x0=z_start, x1=z_end,
                              fillcolor="rgba(220,38,38,0.12)", layer="below",
                              line_width=0)
            fig.add_annotation(x=(z_start+z_end)/2,
                               y=max_ele + padding*0.3,
                               text=f"⚠️ {danger_threshold:.0f}%+",
                               showarrow=False,
                               font=dict(size=9, color="#DC2626"),
                               row=1 if show_slope_subgraph else None,
                               col=1 if show_slope_subgraph else None)

    # ── Waypoints ──
    for wp in waypoints:
        st_ = WAYPOINT_DEFS.get(wp["icon_key"], WAYPOINT_DEFS["📍 Genérico"])
        wlab = "<br>".join(textwrap.wrap(wp["label"], 15))
        _add(go.Scatter(
            x=[wp["km"]], y=[wp["ele"]],
            mode="markers+text",
            text=[f'{st_["emoji"]} {wlab}'],
            textposition="top center",
            marker=dict(color=st_["color"], size=10, symbol="circle",
                        line=dict(color=st_["edge"], width=2)),
            showlegend=False,
            hovertemplate=f"<b>{wp['label']}</b><br>{wp['km']:.1f} km · {wp['ele']:.0f} m<extra></extra>",
        ))

    # ── Subgráfico pendiente ──
    if show_slope_subgraph:
        slps = slopes_array(dist_arr, ele_display)
        seg_mid = (dist_arr[:-1] + dist_arr[1:]) / 2
        fig.add_trace(go.Bar(
            x=seg_mid.tolist(), y=slps.tolist(),
            marker_color=[slope_color(s) for s in slps],
            showlegend=False,
            hovertemplate="km %{x:.2f}<br>%{y:.1f}%<extra></extra>",
        ), row=2, col=1)

    # ── Marcadores km ──
    if show_km_markers:
        for km_mark in np.arange(km_interval, total_km, km_interval):
            fig.add_vline(x=float(km_mark), line=dict(color="#94a3b8", width=1, dash="dot"))

    # ── Start / End — texto vertical con add_annotation, sin icono ──
    for x_pos, name, col in [
        (0.0,      start_loc, "#16A34A"),
        (total_km, end_loc,   "#0F172A"),
    ]:
        if not name:
            continue
        idx = int(np.argmin(np.abs(dist_arr - x_pos)))
        # Marcador simple de punto
        _add(go.Scatter(
            x=[float(dist_arr[idx])], y=[float(ele_display[idx])],
            mode="markers",
            marker=dict(color=col, size=9, symbol="circle",
                        line=dict(color="white", width=2)),
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>{ele_display[idx]:.0f} m<extra></extra>",
        ))
        # Anotación vertical — sin diamante, sin icono, siempre vertical
        _is_start = (x_pos == 0.0)
        _ann_kw = dict(row=1, col=1) if show_slope_subgraph else {}
        fig.add_annotation(
            x=float(dist_arr[idx]),
            y=float(ele_display[idx]),
            text=name,
            showarrow=False,
            textangle=-90,
            xshift=14 if _is_start else -14,
            xanchor="left" if _is_start else "right",
            yanchor="bottom",
            font=dict(size=11, color=col),
            bgcolor=bg_color,
            opacity=0.92,
            **_ann_kw,
        )

    # ── Layout ──
    grid_col = "#e2e8f0" if show_grid else "rgba(0,0,0,0)"
    yrange = [min_ele - padding, max_ele + padding * 2.2]

    layout_kw = dict(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=0, r=0, t=45 if chart_title else 20, b=0),
        hovermode="x unified",
        title=dict(text=chart_title or "", font=dict(size=14), x=0.5) if chart_title else {},
    )
    if show_slope_subgraph:
        fig.update_layout(**layout_kw)
        fig.update_xaxes(showgrid=show_grid, gridcolor=grid_col, zeroline=False, row=2, col=1,
                         title_text="Distancia (km)", title_font=dict(color=text_color))
        fig.update_yaxes(showgrid=show_grid, gridcolor=grid_col, zeroline=False, row=1, col=1,
                         title_text="Altitud (m)", title_font=dict(color=text_color),
                         range=yrange)
        fig.update_yaxes(title_text="Pend. %", row=2, col=1,
                         title_font=dict(color=text_color, size=9))
    else:
        fig.update_layout(
            **layout_kw,
            xaxis=dict(title="Distancia (km)", showgrid=show_grid, gridcolor=grid_col,
                       zeroline=False, title_font=dict(color=text_color)),
            yaxis=dict(title="Altitud (m)", showgrid=show_grid, gridcolor=grid_col,
                       zeroline=False, range=yrange, title_font=dict(color=text_color)),
        )

    total_height = embed_height
    html_str = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={"displayModeBar": True, "responsive": True},
        default_height=f"{total_height}px",
        default_width="100%",
    )
    return html_str


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Carga del archivo — primera prioridad ──
    st.markdown("### 📁 Archivo GPX")
    uploaded_file = st.file_uploader("GPX principal", type=["gpx"], label_visibility="collapsed")

    if uploaded_file:
        st.markdown("### 📍 Localidades")
        start_loc   = st.text_input("Salida",   placeholder="Ej. Segovia")
        end_loc     = st.text_input("Llegada",  placeholder="Ej. Madrid")
        chart_title = st.text_input("Título del perfil", placeholder="Ej. Ruta de los Picos")

    st.markdown("### 🎨 Diseño")

    # ── Opciones del gráfico PRIMERO (más usadas) ──
    with st.expander("📊 Opciones del gráfico", expanded=True):
        smooth_curve    = st.checkbox("Suavizado",         value=True)
        smooth_strength = st.slider("Intensidad suavizado", min_value=3, max_value=51,
                                    value=7, step=2) if smooth_curve else 3
        show_grid       = st.checkbox("Rejilla",           value=True)
        fill_area       = st.checkbox("Rellenar área",     value=True)

        st.markdown("**Pendiente**")
        show_slope_heat = st.checkbox("Heatmap de pendiente", value=False)
        slope_fill_style = "Relleno por pendiente (Komoot)"
        if show_slope_heat:
            slope_fill_style = st.radio(
                "Estilo heatmap",
                ["Relleno por pendiente (Komoot)", "Línea coloreada por pendiente"],
                index=0,
            )
            show_slope_legend = st.checkbox("Mostrar leyenda pendiente", value=True)
        else:
            show_slope_legend = False

        show_danger_zones  = st.checkbox("Zonas de peligro automáticas", value=False)
        danger_threshold   = st.slider("Umbral peligro (%)", 5.0, 30.0, 15.0, 0.5) if show_danger_zones else 15.0

        show_slope_subgraph = st.checkbox("Subgráfico de pendiente", value=False)
        show_km_markers     = st.checkbox("Marcadores de km", value=True)
        km_interval         = st.slider("Cada N km", 1, 20, 5, 1) if show_km_markers else 5

        label_rotation = st.radio("Etiquetas waypoints", ["Horizontal","Vertical"], index=1, horizontal=True)
        aspect_ratio   = st.slider("Proporción W/H", 1.0, 10.0, 4.0, 0.5)

    # ── Colores ──
    with st.expander("🎨 Colores", expanded=False):
        c1, c2 = st.columns(2)
        line_color  = c1.color_picker("Línea",    st.session_state.get("_lc", "#EF4444"), key="cp_lc")
        fill_color  = c2.color_picker("Relleno",  st.session_state.get("_fc", "#FCA5A5"), key="cp_fc")
        bg_color    = c1.color_picker("Fondo",    st.session_state.get("_bc", "#FFFFFF"),  key="cp_bc")
        text_color  = c2.color_picker("Texto",    st.session_state.get("_tc", "#374151"),  key="cp_tc")
        line_width  = st.slider("Grosor línea", 0.5, 12.0, 2.5, 0.5)
        fill_alpha  = st.slider("Opacidad",     0.0,  1.0, 0.65, 0.01)

    # ── Presets de estilo ──
    with st.expander("💾 Presets de estilo", expanded=False):
        FACTORY_PRESETS = {
            "Coral Moderno":   dict(lc="#F43F5E", fc="#FECDD3", bc="#FFF1F2", tc="#1C1C1E"),
            "Océano Profundo": dict(lc="#0EA5E9", fc="#BAE6FD", bc="#0F172A", tc="#E0F2FE"),
            "Bosque Oscuro":   dict(lc="#4ADE80", fc="#166534", bc="#052E16", tc="#DCFCE7"),
            "Arena Desierto":  dict(lc="#D97706", fc="#FDE68A", bc="#1C1917", tc="#FEF3C7"),
            "Neón Urbano":     dict(lc="#A855F7", fc="#3B0764", bc="#09090B", tc="#E9D5FF"),
            "Nieve Alpina":    dict(lc="#64748B", fc="#CBD5E1", bc="#F8FAFC", tc="#0F172A"),
        }
        if "user_presets" not in st.session_state:
            st.session_state["user_presets"] = {}

        all_presets = {"— Sin preset —": None, **FACTORY_PRESETS, **st.session_state["user_presets"]}
        chosen = st.selectbox("Cargar preset", list(all_presets.keys()))
        if st.button("▶ Aplicar preset") and all_presets.get(chosen):
            p = all_presets[chosen]
            st.session_state["_lc"] = p["lc"]
            st.session_state["_fc"] = p["fc"]
            st.session_state["_bc"] = p["bc"]
            st.session_state["_tc"] = p["tc"]
            st.rerun()

        st.markdown("---")
        new_preset_name = st.text_input("Nombre del nuevo preset", placeholder="Mi estilo",
                                         key="new_preset_name_input")
        if st.button("💾 Guardar estilo actual"):
            name = new_preset_name.strip()
            if not name:
                st.warning("Escribe un nombre.")
            elif name in FACTORY_PRESETS:
                st.warning("Nombre reservado, elige otro.")
            else:
                st.session_state["user_presets"][name] = dict(
                    lc=line_color, fc=fill_color, bc=bg_color, tc=text_color)
                st.success(f"✅ «{name}» guardado")
                st.rerun()

        user_preset_names = list(st.session_state["user_presets"].keys())
        if user_preset_names:
            del_name = st.selectbox("Borrar preset", ["— No borrar —"] + user_preset_names,
                                     key="del_preset_sel")
            if st.button("🗑 Borrar") and del_name != "— No borrar —":
                del st.session_state["user_presets"][del_name]
                st.rerun()

    with st.expander("🌐 Embed WordPress", expanded=False):
        embed_width  = st.number_input("Ancho embed (px)", 400, 2000, 900, 50)
        embed_height = st.number_input("Alto embed (px)",  200,  800, 420, 20)
        wp_iframe_base_url = st.text_input("URL base",
                                           placeholder="https://tudominio.com/wp-content/uploads/")

    with st.expander("📐 Tamaños sociales", expanded=False):
        social_preset = st.selectbox("Formato", list(SOCIAL_SIZES.keys()))
        if SOCIAL_SIZES[social_preset] is None:
            cw1, cw2 = st.columns(2)
            custom_w = cw1.number_input("Ancho px", 400, 4000, 1200, 50)
            custom_h = cw2.number_input("Alto px",  200, 4000,  630, 50)
        else:
            custom_w, custom_h = SOCIAL_SIZES[social_preset]

    # ── Comparar rutas — discreta, al fondo ──
    with st.expander("🔀 Comparar con otra ruta", expanded=False):
        uploaded_file2 = st.file_uploader("GPX secundario", type=["gpx"],
                                          label_visibility="collapsed")

    # ── Branding — al final, discreto ──
    st.markdown("---")
    st.caption("🏔️ GPX Altimetry Studio Pro · v3.3")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏔️ GPX Altimetry Studio Pro")

if uploaded_file is None:
    st.info("👆 Carga un archivo GPX en el menú lateral para empezar.")
    st.stop()

# ── Detectar cambio de archivo ──
cur_hash = file_hash(uploaded_file.getvalue())
if st.session_state.get("last_hash") != cur_hash:
    st.session_state.update({
        "last_hash": cur_hash,
        "waypoints": [],
        "gpx_wpts_imported": False,
    })

# ── Parsear ──
with st.spinner("Procesando ruta…"):
    df_raw, total_km, gpx_native_wpts = parse_gpx_cached(uploaded_file.getvalue())

if df_raw is None:
    st.error("No se pudo leer el archivo GPX.")
    st.stop()

if df_raw['ele_corrupted'].iloc[0]:
    st.warning("⚠️ Elevaciones corruptas o ausentes en el GPX. Se han rellenado con 0 m.")

# ── Parsear GPX secundario (FEATURE 11) ──
df_raw2, total_km2, _ = (None, 0.0, []) if uploaded_file2 is None else parse_gpx_cached(uploaded_file2.getvalue())

# ── Suavizado ──
dist_arr  = df_raw['dist'].to_numpy()
ele_raw   = df_raw['ele'].to_numpy()
ele_display = smooth_cached(ele_raw, smooth_strength) if smooth_curve else ele_raw.copy()

# ── Suavizado GPX 2 ──
if df_raw2 is not None:
    dist_arr2   = df_raw2['dist'].to_numpy()
    ele_raw2    = df_raw2['ele'].to_numpy()
    ele_display2 = smooth_cached(ele_raw2, smooth_strength) if smooth_curve else ele_raw2.copy()
else:
    dist_arr2 = ele_display2 = None

# ── Stats ──
min_ele, max_ele, gain, loss, max_slope = calc_stats(dist_arr, ele_display)
padding = (max_ele - min_ele) * 0.15

# ── Importar waypoints nativos ──
if gpx_native_wpts and not st.session_state.get("gpx_wpts_imported"):
    ci1, ci2 = st.columns([3,1])
    ci1.info(f"🗂️ El GPX contiene **{len(gpx_native_wpts)} waypoint(s)** nativos. ¿Importarlos?")
    if ci2.button("📥 Importar"):
        for w in gpx_native_wpts:
            if not any(abs(e["km"]-w["km"]) < 0.05 for e in st.session_state["waypoints"]):
                st.session_state["waypoints"].append(w)
        st.session_state["gpx_wpts_imported"] = True
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS + ITRA  — 2 filas de 4
# ─────────────────────────────────────────────────────────────────────────────
itra = compute_itra(total_km, gain, loss)

row1 = st.columns(4)
row1[0].metric("📏 Distancia",  f"{total_km:.2f} km")
row1[1].metric("⬆️ Desnivel +", f"{gain:.0f} m")
row1[2].metric("⬇️ Desnivel −", f"{loss:.0f} m")
row1[3].metric("📐 Pend. Máx",  f"{max_slope:.1f} %")

row2 = st.columns(4)
row2[0].metric("🔝 Alt. Máx",   f"{max_ele:.0f} m")
row2[1].metric("🔽 Alt. Mín",   f"{min_ele:.0f} m")

# ITRA con info expandida al pasar cursor
_itra_help = (
    f"Effort Distance ITRA: {itra['ed']} · Categoría: {itra['category']} · "
    f"≈{itra['points']} puntos\n\n"
    "La ED (Effort Distance) es la fórmula oficial ITRA para calcular el esfuerzo de una carrera de montaña:\n"
    "ED = distancia_km + desnivel+/100 + desnivel−/200\n\n"
    "Categorías: XS<25 · S<45 · M<75 · L<115 · XL<160 · XXL≥160"
)
row2[2].metric("🏅 ITRA ED",
               f"{itra['ed']} · {itra['category']}",
               help=_itra_help)
row2[3].metric("🏆 Puntos ITRA",
               f"{itra['points']} pts",
               help=_itra_help)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINTS — inicializar session state
# ─────────────────────────────────────────────────────────────────────────────
if "waypoints" not in st.session_state:
    st.session_state["waypoints"] = []

_sl_v  = start_loc   if "start_loc"   in dir() else ""
_el_v  = end_loc     if "end_loc"     in dir() else ""
_ct_v  = chart_title if "chart_title" in dir() else ""

# ─────────────────────────────────────────────────────────────────────────────
# VISTA PREVIA INTERACTIVA  — PRIMERO, sin scroll
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">👁️ Vista Previa Interactiva</div>', unsafe_allow_html=True)

# Botones auto-waypoints encima del perfil (compactos en una fila)
_qa1, _qa2, _qa3, _qa4, _qa5, _qa6 = st.columns(6)
if _qa1.button("🏆 Cima", use_container_width=True):
    peak_idx = int(np.argmax(ele_display))
    if not any("Cima" in w["label"] for w in st.session_state["waypoints"]):
        st.session_state["waypoints"].append({
            "km": float(dist_arr[peak_idx]), "label": "Cima",
            "ele": float(ele_display[peak_idx]), "icon": "🚩", "icon_key": "🚩 Cima",
        })
        st.rerun()

if _qa2.button("📉 Valle", use_container_width=True):
    vi = int(np.argmin(ele_display))
    st.session_state["waypoints"].append({
        "km": float(dist_arr[vi]), "label": "Valle",
        "ele": float(ele_display[vi]), "icon": "📍", "icon_key": "📍 Genérico",
    })
    st.rerun()

if _qa3.button("⛰️ Puertos", use_container_width=True):
    passes = detect_passes(dist_arr, ele_display)
    added = sum(1 for p in passes
                if not any(abs(w["km"]-p["km"]) < 0.5 for w in st.session_state["waypoints"])
                and not st.session_state["waypoints"].append(p))
    if added: st.rerun()
    else: st.toast("No se detectaron puertos significativos")

if _qa4.button("🏘️ Pueblos", use_container_width=True):
    villages = detect_villages(dist_arr, ele_display,
                               min_drop=st.session_state.get("village_min_drop", 40.0),
                               min_dist_km=st.session_state.get("village_min_dist", 1.5))
    added = sum(1 for v in villages
                if not any(abs(w["km"]-v["km"]) < 0.5 for w in st.session_state["waypoints"])
                and not st.session_state["waypoints"].append(v))
    if added: st.rerun()
    else: st.toast("No se detectaron valles habitados")

if _qa5.button("🔀 Ordenar", use_container_width=True):
    st.session_state["waypoints"].sort(key=lambda x: x["km"])
    st.rerun()

if _qa6.button("🗑️ Limpiar", use_container_width=True):
    st.session_state["waypoints"] = []
    st.rerun()

# ── Figura Plotly directa ──
_pad_v = (max_ele - min_ele) * 0.15

if show_slope_subgraph:
    fig_prev = psubplots.make_subplots(
        rows=2, cols=1, row_heights=[0.78, 0.22],
        shared_xaxes=True, vertical_spacing=0.05,
    )
    def _padd(tr): fig_prev.add_trace(tr, row=1, col=1)
else:
    fig_prev = go.Figure()
    def _padd(tr): fig_prev.add_trace(tr)
_sl_v  = start_loc   if "start_loc"   in dir() else ""
_el_v  = end_loc     if "end_loc"     in dir() else ""
_ct_v  = chart_title if "chart_title" in dir() else ""

# Construimos la figura Plotly directamente (no via HTML) para que Streamlit
# la renderice sin el iframe que trunca los márgenes.
_pad_v = (max_ele - min_ele) * 0.15

if show_slope_subgraph:
    fig_prev = psubplots.make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    def _padd(tr): fig_prev.add_trace(tr, row=1, col=1)
else:
    fig_prev = go.Figure()
    def _padd(tr): fig_prev.add_trace(tr)

# ── Relleno/línea ──
if show_slope_heat and slope_fill_style == "Relleno por pendiente (Komoot)":
    _slopes_v = slopes_array(dist_arr, ele_display)
    for _i in range(len(dist_arr) - 1):
        _col = slope_color(_slopes_v[_i])
        _xs  = [dist_arr[_i], dist_arr[_i+1], dist_arr[_i+1], dist_arr[_i], dist_arr[_i]]
        _ys  = [ele_display[_i], ele_display[_i+1],
                min_ele - _pad_v, min_ele - _pad_v, ele_display[_i]]
        _padd(go.Scatter(x=_xs, y=_ys, fill="toself",
                         fillcolor=hex_to_rgba(_col, fill_alpha),
                         line=dict(width=0), mode="lines",
                         showlegend=False, hoverinfo="skip"))
    _padd(go.Scatter(x=dist_arr.tolist(), y=ele_display.tolist(),
                     mode="lines", line=dict(color=line_color, width=line_width),
                     showlegend=False,
                     hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>"))
elif show_slope_heat and slope_fill_style == "Línea coloreada por pendiente":
    _slopes_v = slopes_array(dist_arr, ele_display)
    for _i in range(len(dist_arr) - 1):
        _padd(go.Scatter(
            x=[dist_arr[_i], dist_arr[_i+1]],
            y=[ele_display[_i], ele_display[_i+1]],
            mode="lines", line=dict(color=slope_color(_slopes_v[_i]), width=line_width),
            showlegend=False, hoverinfo="skip"))
else:
    _padd(go.Scatter(
        x=dist_arr.tolist(), y=ele_display.tolist(), mode="lines",
        line=dict(color=line_color, width=line_width),
        fill="tozeroy" if fill_area else "none",
        fillcolor=hex_to_rgba(fill_color, fill_alpha) if fill_area else None,
        hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>",
        showlegend=False,
    ))

# ── Zonas de peligro ──
if show_danger_zones:
    for _zs, _ze in detect_danger_zones(dist_arr, ele_display, threshold=danger_threshold):
        if show_slope_subgraph:
            fig_prev.add_vrect(x0=_zs, x1=_ze, fillcolor="rgba(220,38,38,0.12)",
                               layer="below", line_width=0, row=1, col=1)
        else:
            fig_prev.add_vrect(x0=_zs, x1=_ze, fillcolor="rgba(220,38,38,0.12)",
                               layer="below", line_width=0)

# ── Waypoints — con posición de texto por-waypoint ──
for _wp in st.session_state["waypoints"]:
    _st   = WAYPOINT_DEFS.get(_wp["icon_key"], WAYPOINT_DEFS["📍 Genérico"])
    _wlab = "<br>".join(textwrap.wrap(_wp["label"], 15))
    _vpos = _wp.get("vpos", "Arriba")
    _tpos = "bottom center" if _vpos == "Abajo" else "top center"
    _fs   = int(_wp.get("fontsize", 11))  # Plotly usa tamaños distintos a MPL
    _padd(go.Scatter(
        x=[_wp["km"]], y=[_wp["ele"]],
        mode="markers+text",
        text=[f'{_st["emoji"]} {_wlab}'],
        textposition=_tpos,
        textfont=dict(size=_fs, color=_st["color"]),
        marker=dict(color=_st["color"], size=10, symbol="circle",
                    line=dict(color=_st["edge"], width=2)),
        showlegend=False,
        hovertemplate=f"<b>{_wp['label']}</b><br>{_wp['km']:.1f} km · {_wp['ele']:.0f} m<extra></extra>",
    ))

# ── Start / End ──
for _xp, _nm, _nc in [(0.0, _sl_v, "#16A34A"), (total_km, _el_v, "#0F172A")]:
    if not _nm:
        continue
    _idx = int(np.argmin(np.abs(dist_arr - _xp)))
    # Solo marcador de punto — sin texto inline (que sale horizontal en Plotly)
    _padd(go.Scatter(
        x=[float(dist_arr[_idx])], y=[float(ele_display[_idx])],
        mode="markers",
        marker=dict(color=_nc, size=10, symbol="circle",
                    line=dict(color="white", width=2)),
        showlegend=False,
        hovertemplate=f"<b>{_nm}</b><br>{ele_display[_idx]:.0f} m<extra></extra>",
    ))
    # Texto vertical con add_annotation (textangle=-90 = vertical)
    # Salida: ax ancla a la derecha del punto; Llegada: a la izquierda
    _is_start = (_xp == 0.0)
    _ann_x    = float(dist_arr[_idx])
    _ann_xshift = 14 if _is_start else -14   # px hacia el interior
    _ann_xanchor = "left" if _is_start else "right"
    _ann_kw = dict(row=1, col=1) if show_slope_subgraph else {}
    fig_prev.add_annotation(
        x=_ann_x,
        y=float(ele_display[_idx]),
        text=_nm,
        showarrow=False,
        textangle=-90,           # VERTICAL
        xshift=_ann_xshift,
        xanchor=_ann_xanchor,
        yanchor="bottom",
        font=dict(size=11, color=_nc, family="Syne, system-ui, sans-serif"),
        bgcolor=bg_color,
        opacity=0.92,
        **_ann_kw,
    )

# ── Marcadores km ──
if show_km_markers:
    for _km in np.arange(km_interval, total_km, km_interval):
        fig_prev.add_vline(x=float(_km),
                           line=dict(color="#94a3b8", width=1, dash="dot"))

# ── Subgráfico pendiente ──
if show_slope_subgraph:
    _slps = slopes_array(dist_arr, ele_display)
    _smid = (dist_arr[:-1] + dist_arr[1:]) / 2
    fig_prev.add_trace(go.Bar(
        x=_smid.tolist(), y=_slps.tolist(),
        marker_color=[slope_color(s) for s in _slps],
        showlegend=False,
        hovertemplate="km %{x:.2f}<br>%{y:.1f}%<extra></extra>",
    ), row=2, col=1)

# ── Layout — márgenes generosos para que no se recorte nada ──
_grid_col = "#e2e8f0" if show_grid else "rgba(0,0,0,0)"
_yrange   = [min_ele - _pad_v * 1.2, max_ele + _pad_v * 2.5]
_prev_h   = max(350, int(750 / max(aspect_ratio, 1.0)))
if show_slope_subgraph:
    _prev_h += 140

_layout_prev = dict(
    height=_prev_h,
    paper_bgcolor=bg_color,
    plot_bgcolor=bg_color,
    # Márgenes con espacio real para etiquetas de waypoints arriba y ejes abajo
    margin=dict(l=60, r=30, t=60 if _ct_v else 30, b=60),
    hovermode="x unified",
    title=dict(text=_ct_v, font=dict(size=14, color=text_color), x=0.5) if _ct_v else {},
)

if show_slope_subgraph:
    fig_prev.update_layout(**_layout_prev)
    fig_prev.update_xaxes(showgrid=show_grid, gridcolor=_grid_col, zeroline=False,
                          row=2, col=1,
                          title_text="Distancia (km)",
                          title_font=dict(color=text_color),
                          tickfont=dict(color=text_color))
    fig_prev.update_yaxes(showgrid=show_grid, gridcolor=_grid_col, zeroline=False,
                          row=1, col=1,
                          title_text="Altitud (m)",
                          title_font=dict(color=text_color),
                          tickfont=dict(color=text_color),
                          range=_yrange)
    fig_prev.update_yaxes(title_text="Pend. %", row=2, col=1,
                          title_font=dict(color=text_color, size=9),
                          tickfont=dict(color=text_color))
    fig_prev.update_xaxes(showgrid=show_grid, gridcolor=_grid_col, zeroline=False,
                          row=1, col=1, tickfont=dict(color=text_color))
else:
    fig_prev.update_layout(
        **_layout_prev,
        xaxis=dict(title="Distancia (km)", showgrid=show_grid, gridcolor=_grid_col,
                   zeroline=False, range=[0, total_km * 1.02],
                   title_font=dict(color=text_color), tickfont=dict(color=text_color)),
        yaxis=dict(title="Altitud (m)", showgrid=show_grid, gridcolor=_grid_col,
                   zeroline=False, range=_yrange,
                   title_font=dict(color=text_color), tickfont=dict(color=text_color)),
    )

st.plotly_chart(fig_prev, use_container_width=True)

# ── Tabs bajo el perfil: Mapa, Añadir waypoint, Editor, Config ──
_tab_map, _tab_add, _tab_edit, _tab_cfg = st.tabs([
    "🗺️ Mapa", "➕ Añadir waypoint", "✏️ Editar waypoints", "⚙️ Config detección"
])

with _tab_map:
    _map_max  = round(float(total_km), 1)
    _map_init = round(round(float(total_km / 2) / 0.1) * 0.1, 1)
    _map_init = min(_map_init, _map_max)
    map_km_sel = st.slider("📍 Posición en ruta (km)", 0.0, _map_max,
                            _map_init, 0.1, key="map_sel")
    idx_map = int(np.argmin(np.abs(dist_arr - map_km_sel)))
    sel_ele = float(ele_display[idx_map])
    sel_lat = float(df_raw.iloc[idx_map]['lat'])
    sel_lon = float(df_raw.iloc[idx_map]['lon'])

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermap(
        mode="lines",
        lon=df_raw['lon'].tolist(), lat=df_raw['lat'].tolist(),
        line={"width": 3, "color": line_color}, name="Ruta", hoverinfo="skip",
    ))
    if st.session_state["waypoints"]:
        wlons = [df_raw.iloc[int(np.argmin(np.abs(dist_arr-w["km"])))]["lon"]
                 for w in st.session_state["waypoints"]]
        wlats = [df_raw.iloc[int(np.argmin(np.abs(dist_arr-w["km"])))]["lat"]
                 for w in st.session_state["waypoints"]]
        fig_map.add_trace(go.Scattermap(
            mode="markers", lon=wlons, lat=wlats,
            marker={"size": 12, "color": "#F97316"}, name="Waypoints",
            hovertext=[f'{w["icon"]} {w["label"]} · {w["km"]:.1f}km' for w in st.session_state["waypoints"]],
            hoverinfo="text",
        ))
    fig_map.add_trace(go.Scattermap(
        mode="markers", lon=[sel_lon], lat=[sel_lat],
        marker={"size": 16, "color": "#EF4444"}, name="Selección",
        hovertemplate=f"km {map_km_sel:.1f} · {sel_ele:.0f} m<extra></extra>",
    ))
    fig_map.update_layout(
        map={"style": "open-street-map",
             "center": {"lon": sel_lon, "lat": sel_lat}, "zoom": 11},
        showlegend=False, margin={"l":0,"r":0,"b":0,"t":0}, height=320,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with _tab_add:
    _map_max2  = round(float(total_km), 1)
    _map_init2 = round(round(float(total_km / 2) / 0.1) * 0.1, 1)
    _add_km = st.slider("📍 Posición (km)", 0.0, _map_max2, _map_init2, 0.1, key="add_km_sel")
    _add_idx = int(np.argmin(np.abs(dist_arr - _add_km)))
    _add_ele = float(ele_display[_add_idx])
    st.caption(f"Altitud: **{_add_ele:.0f} m**")
    _ac1, _ac2, _ac3 = st.columns([2, 2, 1])
    _wp_type = _ac1.selectbox("Tipo", list(WAYPOINT_DEFS.keys()), key="wp_type")
    _wp_name = _ac2.text_input("Nombre", "Punto", key="wp_name")
    if _ac3.button("➕ Añadir", use_container_width=True, key="btn_add"):
        dup = any(abs(w["km"]-_add_km) < 0.05 and w["label"] == _wp_name
                  for w in st.session_state["waypoints"])
        if dup:
            st.warning("Punto duplicado.")
        else:
            st.session_state["waypoints"].append({
                "km": _add_km, "label": _wp_name, "ele": _add_ele,
                "icon": WAYPOINT_DEFS[_wp_type]["emoji"], "icon_key": _wp_type,
            })
            st.success(f"✅ '{_wp_name}' añadido")
            st.rerun()

with _tab_edit:
    if not st.session_state["waypoints"]:
        st.info("No hay waypoints todavía. Usa los botones de arriba o la pestaña 'Añadir'.")
    else:
        if "wp_edit_idx" not in st.session_state:
            st.session_state["wp_edit_idx"] = None
        for i, wp in enumerate(st.session_state["waypoints"]):
            row_cols = st.columns([0.4, 3.5, 1.2, 0.9, 0.9, 0.9])
            row_cols[0].markdown(
                f'<span style="font-size:1.3rem;line-height:2rem">{wp["icon"]}</span>',
                unsafe_allow_html=True)
            row_cols[1].markdown(
                f'<span class="wp-chip"><b>{wp["km"]:.1f} km</b> — {wp["label"]}'
                f' <em style="opacity:.6">({wp["ele"]:.0f} m)</em></span>',
                unsafe_allow_html=True)
            is_editing = st.session_state["wp_edit_idx"] == i
            if row_cols[2].button("🔼 Cerrar" if is_editing else "✏️ Editar",
                                  key=f"edit_btn_{i}", use_container_width=True):
                st.session_state["wp_edit_idx"] = None if is_editing else i
                st.rerun()
            if row_cols[3].button("▲", key=f"up_{i}", use_container_width=True) and i > 0:
                st.session_state["waypoints"][i], st.session_state["waypoints"][i-1] = \
                    st.session_state["waypoints"][i-1], st.session_state["waypoints"][i]
                st.rerun()
            if row_cols[4].button("▼", key=f"dn_{i}", use_container_width=True) \
                    and i < len(st.session_state["waypoints"]) - 1:
                st.session_state["waypoints"][i], st.session_state["waypoints"][i+1] = \
                    st.session_state["waypoints"][i+1], st.session_state["waypoints"][i]
                st.rerun()
            if row_cols[5].button("✕", key=f"del_{i}", use_container_width=True):
                st.session_state["waypoints"].pop(i)
                if st.session_state["wp_edit_idx"] == i:
                    st.session_state["wp_edit_idx"] = None
                st.rerun()
            if is_editing:
                ec1, ec2, ec3 = st.columns([2, 2, 1])
                new_label = ec1.text_input("📝 Nombre", value=wp["label"], key=f"ed_label_{i}")
                icon_keys = list(WAYPOINT_DEFS.keys())
                cur_icon_idx = icon_keys.index(wp["icon_key"]) if wp["icon_key"] in icon_keys else 0
                new_icon_key = ec2.selectbox("🏷️ Tipo", icon_keys, index=cur_icon_idx, key=f"ed_icon_{i}")
                new_km = ec3.number_input("📍 km", 0.0, float(total_km),
                                          float(wp["km"]), 0.1, key=f"ed_km_{i}", format="%.1f")
                ec4, ec5, ec6 = st.columns([2, 2, 1])
                cur_rot = wp.get("rotation", label_rotation)
                new_rot = ec4.radio("↔️ Orientación", ["Horizontal","Vertical"],
                                    index=0 if cur_rot == "Horizontal" else 1,
                                    key=f"ed_rot_{i}", horizontal=True)
                new_fs = ec5.slider("🔤 Tamaño", 6, 16, int(wp.get("fontsize", 8)), 1, key=f"ed_fs_{i}")
                cur_vpos = wp.get("vpos", "Arriba")
                new_vpos = ec6.selectbox("↕️ Pos.", ["Arriba","Abajo"],
                                         index=0 if cur_vpos == "Arriba" else 1, key=f"ed_vpos_{i}")
                if st.button("💾 Aplicar", key=f"ed_save_{i}"):
                    idx_new = int(np.argmin(np.abs(dist_arr - new_km)))
                    st.session_state["waypoints"][i] = {
                        **wp, "label": new_label, "icon_key": new_icon_key,
                        "icon": WAYPOINT_DEFS[new_icon_key]["emoji"],
                        "km": float(new_km), "ele": float(ele_display[idx_new]),
                        "rotation": new_rot, "fontsize": new_fs, "vpos": new_vpos,
                    }
                    st.session_state["wp_edit_idx"] = None
                    st.rerun()

with _tab_cfg:
    c_pa1, c_pa2, c_pa3, c_pa4 = st.columns(4)
    st.session_state["village_min_drop"] = c_pa1.number_input(
        "Pueblos: caída mín. (m)", 10, 300, 40, 10,
        help="Metros de descenso para considerar un valle como pueblo")
    st.session_state["village_min_dist"] = c_pa2.number_input(
        "Pueblos: dist. mín. (km)", 0.5, 10.0, 1.5, 0.5,
        help="Separación mínima entre pueblos")
    c_pa3.number_input("Puertos: prominencia mín. (m)", 50, 500, 100, 25, key="pass_min_gain")
    c_pa4.number_input("Puertos: dist. mín. (km)", 0.5, 10.0, 1.0, 0.5, key="pass_min_dist")

# HTML embed para descarga
html_embed_str = build_html_embed(
    dist_arr, ele_display, df_raw, total_km,
    min_ele, max_ele, gain, loss, max_slope,
    line_color, fill_color, bg_color, text_color,
    line_width, fill_alpha, fill_area,
    show_slope_heat, slope_fill_style,
    show_grid, show_km_markers, km_interval,
    st.session_state["waypoints"],
    _sl_v, _el_v, _ct_v,
    show_danger_zones, danger_threshold,
    show_slope_subgraph, embed_width, embed_height,
)

# ── Comparación de rutas (FEATURE 11) ──
if df_raw2 is not None:
    st.markdown('<div class="sec">🔀 Comparación de Rutas</div>', unsafe_allow_html=True)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=dist_arr.tolist(), y=ele_display.tolist(),
        mode="lines", name=chart_title or "Ruta 1",
        line=dict(color=line_color, width=line_width),
        fill="tozeroy", fillcolor=hex_to_rgba(fill_color, 0.3),
        hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>",
    ))
    fig_cmp.add_trace(go.Scatter(
        x=dist_arr2.tolist(), y=ele_display2.tolist(),
        mode="lines", name="Ruta 2",
        line=dict(color="#F97316", width=line_width, dash="dot"),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.2)",
        hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>",
    ))
    min2, max2, gain2, loss2, ms2 = calc_stats(dist_arr2, ele_display2)
    itra2 = compute_itra(total_km2, gain2, loss2)
    fig_cmp.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=30,b=0), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Distancia (km)", showgrid=show_grid, gridcolor="#e2e8f0"),
        yaxis=dict(title="Altitud (m)", showgrid=show_grid, gridcolor="#e2e8f0"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
    cp1, cp2 = st.columns(2)
    cp1.info(f"**Ruta 1:** {total_km:.1f} km · ↑{gain:.0f}m · ↓{loss:.0f}m · ITRA {itra['category']} ({itra['ed']} ED)")
    cp2.info(f"**Ruta 2:** {total_km2:.1f} km · ↑{gain2:.0f}m · ↓{loss2:.0f}m · ITRA {itra2['category']} ({itra2['ed']} ED)")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# FICHA MIDE  — Sistema de valoración montañismo español
# ═════════════════════════════════════════════════════════════════════════════
# Variables de localidades / título definidas aquí de forma defensiva
# (pueden no existir si el usuario no las ha rellenado en la sidebar)
_ct  = chart_title if "chart_title" in dir() else ""
_sl  = start_loc   if "start_loc"   in dir() else ""
_el  = end_loc     if "end_loc"     in dir() else ""

with st.expander("🏔️ Generar Ficha MIDE", expanded=False):
    st.markdown("**Sistema MIDE** — Modelo de Información de Excursiones (Gobierno de Aragón / PRAMES)")
    st.caption("Los datos de distancia y desnivel se toman automáticamente del GPX cargado.")

    _mc1, _mc2 = st.columns([1, 1])

    # ── Tipo de recorrido ──────────────────────────────────────────────────
    with _mc1:
        _trip_type = st.radio(
            "📍 Tipo de recorrido",
            ["Ida y vuelta", "Circular", "Travesía (A→B)"],
            horizontal=True, key="mide_trip",
        )

        # Para ida y vuelta, doble desnivel
        if _trip_type == "Ida y vuelta":
            _m_gain = gain * 2
            _m_loss = loss * 2
            _m_dist = total_km * 2
            st.caption(f"Ida y vuelta: {_m_dist:.1f} km · ↑{_m_gain:.0f}m · ↓{_m_loss:.0f}m")
        else:
            _m_gain = gain
            _m_loss = loss
            _m_dist = total_km

        # ── Tipo de terreno (afecta velocidad) ──
        _terrain = st.selectbox(
            "🥾 Tipo de terreno predominante",
            list(MIDE_TERRAIN_SPEEDS.keys()), key="mide_terrain",
        )
        _speed = MIDE_TERRAIN_SPEEDS[_terrain]

        # ── Calcular esfuerzo ──
        _mide_e = compute_mide_effort(_m_gain, _m_loss, _m_dist, _speed)

        st.markdown(f"""
        <div class="mide-card">
          <div class="mide-row">
            <span class="mide-label">⏱ Tiempo estimado</span>
            <strong style="font-size:1.2rem">{_mide_e['hhmm']}</strong>
          </div>
          <div class="mide-row" style="font-size:.78rem;color:#64748b;">
            T_desnivel = {_mide_e['t_h']:.1f}h &nbsp;|&nbsp;
            T_distancia = {_mide_e['t_d']:.1f}h
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Orientación y Desplazamiento ──────────────────────────────────────
    with _mc2:
        _orient_label = st.selectbox(
            "🧭 Orientación",
            [o[0] for o in MIDE_ORIENTATION_OPTS], key="mide_orient",
        )
        _orient_score = dict(MIDE_ORIENTATION_OPTS)[_orient_label]

        _displac_label = st.selectbox(
            "🦶 Desplazamiento",
            [d[0] for d in MIDE_DISPLACEMENT_OPTS], key="mide_displac",
        )
        _displac_score = dict(MIDE_DISPLACEMENT_OPTS)[_displac_label]

    # ── Checklist Severidad del Medio ─────────────────────────────────────
    st.markdown("**⚠️ Factores de riesgo del medio** — Marca todos los que apliquen a esta ruta:")
    _risk_cols = st.columns(2)
    _selected_risks = []
    for _ri, _rf in enumerate(MIDE_RISK_FACTORS):
        _col = _risk_cols[_ri % 2]
        if _col.checkbox(_rf, key=f"mide_risk_{_ri}"):
            _selected_risks.append(_rf)

    _medio_score = mide_medio_score(len(_selected_risks))

    # ── Dificultades técnicas específicas ─────────────────────────────────
    _sym_T = _displac_score >= 4
    _sym_R = st.checkbox("🪢 Descenso en rápel necesario", key="mide_rapel")
    _sym_N = any("nieve" in r.lower() or "glaciar" in r.lower() or "nevero" in r.lower()
                 for r in _selected_risks)
    _rapel_m = ""
    _nieve_deg = ""
    if _sym_R:
        _rapel_m = st.number_input("Metros de rápel total", 5, 500, 30, 5, key="mide_rapel_m")
    if _sym_N:
        _nieve_deg = st.number_input("Inclinación máxima de nieve (°)", 15, 80, 35, 5, key="mide_nieve_deg")

    # ── FICHA FINAL ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Ficha MIDE generada")

    _scores = {
        "Esfuerzo":      _mide_e["effort"],
        "Orientación":   _orient_score,
        "Desplazamiento":_displac_score,
        "Medio":         _medio_score,
    }
    _score_icons = {"Esfuerzo": "💪", "Orientación": "🧭",
                    "Desplazamiento": "🦶", "Medio": "⚠️"}

    # Fila de pills de color
    _pill_html = '<div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:12px;">'
    for _sname, _sval in _scores.items():
        _sc = mide_score_color(_sval)
        _pill_html += f"""
        <div style="text-align:center;">
          <div class="mide-pill" style="background:{_sc};">{_sval}</div>
          <div style="font-size:.7rem;color:#64748b;margin-top:2px;">{_sname}</div>
        </div>"""
    _pill_html += "</div>"
    st.markdown(_pill_html, unsafe_allow_html=True)

    # Datos de referencia
    _ref_lines = [
        f"📏 Distancia: {_m_dist:.2f} km",
        f"⬆️ Subida: {_m_gain:.0f} m",
        f"⬇️ Bajada: {_m_loss:.0f} m",
        f"⏱ Tiempo estimado: {_mide_e['hhmm']}",
        f"🗺 Tipo: {_trip_type}",
    ]
    _fi1, _fi2 = st.columns([1.2, 1])
    with _fi1:
        for _rl in _ref_lines:
            st.markdown(f"- {_rl}")

    with _fi2:
        st.markdown("**Niveles:**")
        for _sname, _sval in _scores.items():
            _sc = mide_score_color(_sval)
            _sl_txt = mide_score_label(_sval)
            st.markdown(
                f'<span style="color:{_sc};font-weight:700;">{_score_icons[_sname]} '
                f'{_sname}: {_sval} — {_sl_txt}</span>',
                unsafe_allow_html=True,
            )

        # Símbolos técnicos
        _syms = []
        if _sym_T: _syms.append(f'<span class="mide-sym">T</span> Escalada/UIAA')
        if _sym_R: _syms.append(f'<span class="mide-sym">R</span> Rápel {_rapel_m}m')
        if _sym_N: _syms.append(f'<span class="mide-sym">N</span> Nieve {_nieve_deg}°')
        if _syms:
            st.markdown("**Advertencias técnicas:**")
            for _s in _syms:
                st.markdown(_s, unsafe_allow_html=True)

    # Factores de riesgo marcados
    if _selected_risks:
        st.markdown(f"**Factores de riesgo ({len(_selected_risks)}):** " +
                    " · ".join(f"_{r}_" for r in _selected_risks))

    # ── FICHA VISUAL MATPLOTLIB ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🖼️ Ficha MIDE oficial — exportar imagen")

    # Construir la figura
    _mide_fig = build_mide_figure(
        route_name      = _ct or (uploaded_file.name.replace(".gpx","") if uploaded_file else ""),
        trip_type       = _trip_type,
        dist_km         = _m_dist,
        gain_m          = _m_gain,
        loss_m          = _m_loss,
        t_hhmm          = _mide_e["hhmm"],
        medio           = _medio_score,
        orientacion     = _orient_score,
        desplazamiento  = _displac_score,
        esfuerzo        = _mide_e["effort"],
        sym_T           = _sym_T,
        sym_R           = _sym_R,
        sym_N           = _sym_N,
        rapel_m         = float(_rapel_m) if _rapel_m else 0.0,
        nieve_deg       = float(_nieve_deg) if _nieve_deg else 0.0,
        dist_arr        = dist_arr,
        ele_display     = ele_display,
        waypoints       = st.session_state.get("waypoints", []),
    )

    # Vista previa en Streamlit
    st.pyplot(_mide_fig, use_container_width=True)

    # Generar buffers en memoria
    _fname_base = (_ct or (uploaded_file.name.replace(".gpx","") if uploaded_file else "MIDE")).replace(" ","_")

    _buf_png = io.BytesIO()
    _mide_fig.savefig(_buf_png, format="png", dpi=150, bbox_inches="tight")
    _buf_png.seek(0)

    _buf_jpg = io.BytesIO()
    # JPEG no admite canal alpha — forzamos fondo blanco opaco
    _mide_fig.savefig(_buf_jpg, format="jpeg", dpi=150, bbox_inches="tight",
                      facecolor="#E8E8E8", quality=92)
    _buf_jpg.seek(0)

    _buf_svg = io.BytesIO()
    _mide_fig.savefig(_buf_svg, format="svg", bbox_inches="tight")
    _buf_svg.seek(0)

    plt.close(_mide_fig)

    # Botones de descarga en una fila
    _dl1, _dl2, _dl3, _dl4 = st.columns(4)
    _dl1.download_button(
        "⬇️ PNG",
        _buf_png,
        file_name=f"{_fname_base}_MIDE.png",
        mime="image/png",
        use_container_width=True,
        key="dl_mide_png",
    )
    _dl2.download_button(
        "⬇️ JPG",
        _buf_jpg,
        file_name=f"{_fname_base}_MIDE.jpg",
        mime="image/jpeg",
        use_container_width=True,
        key="dl_mide_jpg",
    )
    _dl3.download_button(
        "⬇️ SVG",
        _buf_svg,
        file_name=f"{_fname_base}_MIDE.svg",
        mime="image/svg+xml",
        use_container_width=True,
        key="dl_mide_svg",
    )

    # TXT resumen
    _ficha_txt = f"""FICHA MIDE — {_ct or (uploaded_file.name if uploaded_file else '')}
{'='*50}
Tipo de recorrido: {_trip_type}
Distancia: {_m_dist:.2f} km | Subida: {_m_gain:.0f}m | Bajada: {_m_loss:.0f}m
Tiempo estimado MIDE: {_mide_e['hhmm']}
Terreno: {_terrain}

VALORACIONES MIDE
  Severidad del medio:    {_medio_score}/5 — {mide_score_label(_medio_score)}
  Orientación:            {_orient_score}/5 — {mide_score_label(_orient_score)}
  Desplazamiento:         {_displac_score}/5 — {mide_score_label(_displac_score)}
  Esfuerzo:               {_mide_e['effort']}/5 — {mide_score_label(_mide_e['effort'])}

{'Símbolos: ' + ' '.join(filter(None,['T' if _sym_T else '','R' if _sym_R else '','N' if _sym_N else ''])) if any([_sym_T,_sym_R,_sym_N]) else ''}
{'Factores de riesgo: ' + ', '.join(_selected_risks) if _selected_risks else ''}
"""
    _dl4.download_button(
        "⬇️ TXT",
        _ficha_txt.strip(),
        file_name=f"{_fname_base}_MIDE.txt",
        mime="text/plain",
        use_container_width=True,
        key="dl_mide_txt",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN ESTÁTICA  (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">💾 Exportar Imagen</div>', unsafe_allow_html=True)

extra_top = 4.5 if label_rotation == "Vertical" else 2.2
if "start_loc" in dir() and start_loc:
    extra_top = max(extra_top, 4.0)

common_kwargs = dict(
    dist_arr=dist_arr, ele_display=ele_display, df_raw=df_raw,
    total_km=total_km, min_ele=min_ele, max_ele=max_ele,
    gain=gain, loss=loss, max_slope=max_slope,
    line_color=line_color, fill_color=fill_color,
    bg_color=bg_color, text_color=text_color,
    line_width=line_width, fill_alpha=fill_alpha, fill_area=fill_area,
    show_slope_heat=show_slope_heat, slope_fill_style=slope_fill_style,
    show_grid=show_grid, show_km_markers=show_km_markers, km_interval=km_interval,
    label_rotation=label_rotation, aspect_ratio=aspect_ratio,
    start_loc=_sl, end_loc=_el, chart_title=_ct,
    waypoints=st.session_state["waypoints"],
    show_danger_zones=show_danger_zones, danger_threshold=danger_threshold,
    show_slope_subgraph=show_slope_subgraph, show_slope_legend=show_slope_legend,
    extra_top_factor=extra_top,
)

fn = uploaded_file.name.replace(".gpx", "")

tab_std, tab_social, tab_html, tab_geojson, tab_batch = st.tabs([
    "🖼️ PNG / JPG / SVG",
    "📱 Tamaños Sociales",
    "🌐 HTML WordPress",
    "🗺️ GeoJSON",
    "📦 Batch ZIP",
])

# ── TAB 1: Estándar ──
with tab_std:
    fig_s = build_static_fig(**common_kwargs)
    for fmt, mime, key in [("png","image/png","dl_png"),
                            ("jpg","image/jpeg","dl_jpg"),
                            ("svg","image/svg+xml","dl_svg")]:
        buf = io.BytesIO()
        kw  = dict(format=fmt, dpi=200, bbox_inches="tight",
                   facecolor=bg_color, edgecolor="none")
        if fmt == "jpg":
            kw["pil_kwargs"] = {"quality": 95}
        fig_s.savefig(buf, **kw)
        buf.seek(0)
        st.download_button(f"💾 {fmt.upper()}", buf.getvalue(),
                           f"{fn}_perfil.{fmt}", mime, key=key)
    plt.close(fig_s)

# ── TAB 2: Social sizes ──
with tab_social:
    sw = custom_w if SOCIAL_SIZES[social_preset] is None else SOCIAL_SIZES[social_preset][0]
    sh = custom_h if SOCIAL_SIZES[social_preset] is None else SOCIAL_SIZES[social_preset][1]
    st.caption(f"Tamaño: {sw} × {sh} px")
    fig_soc = build_static_fig(**{**common_kwargs, "fig_w_px": sw, "fig_h_px": sh})
    buf_soc = io.BytesIO()
    fig_soc.savefig(buf_soc, format="png", dpi=150, bbox_inches="tight",
                    facecolor=bg_color, edgecolor="none")
    buf_soc.seek(0)
    plt.close(fig_soc)
    st.download_button(f"💾 PNG {sw}×{sh}", buf_soc.getvalue(),
                       f"{fn}_{sw}x{sh}.png", "image/png", key="dl_social")
    st.info(f"Formato **{social_preset}** listo para usar en redes sociales.")

# ── TAB 3: HTML WordPress (FEATURE 1 + 7) ──
with tab_html:
    buf_html = html_embed_str.encode("utf-8")
    st.download_button(
        "💾 Descargar HTML interactivo", buf_html,
        f"{fn}_embed.html", "text/html", key="dl_html",
    )
    st.markdown("**Cómo usarlo en WordPress:**")
    st.markdown("""
    1. Sube el `.html` a **Media** de WordPress (o a tu servidor FTP)
    2. Copia la URL del archivo subido
    3. Añade en tu entrada/página un bloque **HTML personalizado** con:
    """)
    iframe_url = wp_iframe_base_url.rstrip("/") + "/" + f"{fn}_embed.html" if wp_iframe_base_url else "https://tudominio.com/ruta.html"
    shortcode = f'<iframe src="{iframe_url}" width="100%" height="{embed_height}px" style="border:none;border-radius:8px;" loading="lazy" title="{_ct or fn}"></iframe>'
    st.markdown(f'<div class="shortcode-box">{shortcode}</div>', unsafe_allow_html=True)

    # FEATURE 7: shortcode copy-ready
    buf_sc = shortcode.encode("utf-8")
    st.download_button("📋 Descargar shortcode .txt", buf_sc,
                       f"{fn}_shortcode.txt", "text/plain", key="dl_sc")
    st.caption("💡 Tip: Instala el plugin 'iframe' en WordPress para usar `[iframe]` shortcodes nativos.")

# ── TAB 4: GeoJSON (FEATURE 14) ──
with tab_geojson:
    geojson_str = build_geojson(df_raw, ele_display, dist_arr)
    st.download_button("💾 Descargar GeoJSON", geojson_str,
                       f"{fn}_slope.geojson", "application/json", key="dl_geojson")
    st.markdown("""
    **Uso con Leaflet (WordPress / web):**
    ```js
    fetch('ruta_slope.geojson')
      .then(r => r.json())
      .then(data => {
        L.geoJSON(data, {
          style: f => ({ color: f.properties.slope_color, weight: 4 })
        }).addTo(map);
      });
    ```
    Cada segmento lleva: `slope_pct`, `slope_color`, `km_start`, `km_end`, `ele_start`, `ele_end`.
    """)

# ── TAB 5: Batch ZIP (FEATURE 16) ──
with tab_batch:
    st.markdown("**Procesar múltiples GPX a la vez**")
    batch_files = st.file_uploader("Sube varios GPX", type=["gpx"],
                                   accept_multiple_files=True, key="batch_up")
    if batch_files and st.button("🚀 Generar todos", key="btn_batch"):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            prog = st.progress(0)
            for bi, bf in enumerate(batch_files):
                df_b, km_b, _ = parse_gpx_cached(bf.getvalue())
                if df_b is None:
                    continue
                ele_b = smooth_cached(df_b['ele'].to_numpy(), smooth_strength) if smooth_curve else df_b['ele'].to_numpy()
                dist_b = df_b['dist'].to_numpy()
                mn_b, mx_b, gn_b, ls_b, ms_b = calc_stats(dist_b, ele_b)
                fig_b = build_static_fig(
                    dist_arr=dist_b, ele_display=ele_b, df_raw=df_b,
                    total_km=km_b, min_ele=mn_b, max_ele=mx_b,
                    gain=gn_b, loss=ls_b, max_slope=ms_b,
                    **{k:v for k,v in common_kwargs.items()
                       if k not in ("dist_arr","ele_display","df_raw","total_km",
                                    "min_ele","max_ele","gain","loss","max_slope",
                                    "start_loc","end_loc","chart_title","waypoints")},
                    start_loc="", end_loc="",
                    chart_title=bf.name.replace(".gpx",""),
                    waypoints=[],
                )
                buf_b = io.BytesIO()
                fig_b.savefig(buf_b, format="png", dpi=150, bbox_inches="tight",
                              facecolor=bg_color, edgecolor="none")
                buf_b.seek(0)
                plt.close(fig_b)
                zf.writestr(bf.name.replace(".gpx","_perfil.png"), buf_b.getvalue())
                prog.progress((bi+1)/len(batch_files))
        zip_buf.seek(0)
        st.download_button("📦 Descargar ZIP", zip_buf.getvalue(),
                           "perfiles_batch.zip", "application/zip", key="dl_zip")
        st.success(f"✅ {len(batch_files)} perfiles generados")

st.divider()
st.caption("GPX Altimetry Studio Pro v3.2 · Python + Streamlit + Matplotlib + Plotly")
