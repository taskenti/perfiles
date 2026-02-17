# ═══════════════════════════════════════════════════════════════════════════════
#  GPX ALTIMETRY STUDIO PRO  ·  v3.0
#  Features: HTML embed, slope gradient fill, social sizes, slope subgraph,
#  sector table, km bands, WP shortcode, danger zones, 3D shadow, ITRA score,
#  dual GPX compare, mini-map, auto-pass detection, GeoJSON export, batch ZIP
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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
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
.shortcode-box { background:#1e293b;color:#7dd3fc;font-family:'DM Mono',monospace;
                 font-size:.8rem;padding:10px 14px;border-radius:8px;word-break:break-all; }
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

# Paleta de pendiente estándar industria (% → color hex)
SLOPE_PALETTE = [
    (-99, -12, "#1D4ED8"),   # bajada pronunciada
    (-12,  -6, "#60A5FA"),   # bajada suave
    ( -6,   3, "#22C55E"),   # llano / suave
    (  3,   8, "#FBBF24"),   # moderado
    (  8,  12, "#F97316"),   # empinado
    ( 12,  99, "#DC2626"),   # muy empinado / peligroso
]

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


# ─── FEATURE 13: Detección automática de puertos ────────────────────────────
def detect_passes(dist: np.ndarray, ele: np.ndarray,
                  min_gain: float = 100, min_dist_km: float = 1.0) -> list:
    """
    Detecta máximos locales significativos (puertos/collados).
    Retorna lista de {'km', 'ele', 'label'}.
    """
    from scipy.signal import find_peaks
    # Suavizar para evitar ruido
    smooth = pd.Series(ele).rolling(window=max(5, len(ele)//200), center=True, min_periods=1).mean().values
    peaks, props = find_peaks(smooth, prominence=min_gain, distance=int(min_dist_km / (dist[-1]/len(dist))))
    passes = []
    for i, p in enumerate(peaks):
        passes.append({
            "km":    round(float(dist[p]), 2),
            "ele":   round(float(ele[p]), 1),
            "label": f"Puerto {i+1}",
            "icon":  "⛰️",
            "icon_key": "⛰️ Puerto",
        })
    return passes


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
        seg_cols = compute_slope_colors_arr(dist_arr, ele_display)
        for i in range(len(dist_arr) - 1):
            ax.plot([dist_arr[i], dist_arr[i+1]],
                    [ele_display[i], ele_display[i+1]],
                    color=seg_cols[i], linewidth=line_width,
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
    # CAPA 5 — Etiquetas de localidades
    # ══════════════════════════════════════════════════════
    rotation_deg   = 90 if label_rotation == "Vertical" else 0
    y_offset_label = padding * (1.0 if label_rotation == "Vertical" else 0.5)

    def _label(x, y, text, ha="center"):
        ax.text(x, y + y_offset_label, text,
                ha=ha, va="bottom", rotation=90, fontsize=9,
                fontweight="bold", color=text_color,
                bbox=dict(facecolor=bg_color, alpha=0.92, edgecolor="none",
                          pad=3, boxstyle="round,pad=0.35"), zorder=5)

    if start_loc:
        ax.plot(0, float(ele_display[0]), marker="D", color="#16A34A",
                markersize=9, markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        _label(0, float(ele_display[0]), start_loc, ha="left")

    if end_loc:
        ax.plot(total_km, float(ele_display[-1]), marker="D", color="#0F172A",
                markersize=9, markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        _label(total_km, float(ele_display[-1]), end_loc, ha="right")

    # ══════════════════════════════════════════════════════
    # CAPA 6 — Waypoints
    # ══════════════════════════════════════════════════════
    for wp in waypoints:
        ax.plot([wp["km"], wp["km"]], [base_fill, wp["ele"]],
                color=text_color, linestyle=":", linewidth=0.9, alpha=0.35, zorder=4)
        st_ = WAYPOINT_DEFS.get(wp["icon_key"], WAYPOINT_DEFS["📍 Genérico"])
        ax.plot(wp["km"], wp["ele"],
                marker=st_["marker"], color=st_["color"],
                markersize=st_["size"], markeredgecolor=st_["edge"],
                markeredgewidth=2.0, zorder=6)
        wrapped = "\n".join(textwrap.wrap(wp["label"], width=14))
        ax.text(wp["km"], wp["ele"] + y_offset_label, wrapped,
                ha="center", va="bottom", rotation=rotation_deg,
                fontsize=8, fontweight="bold", color=text_color,
                bbox=dict(facecolor=bg_color, alpha=0.92, edgecolor="none",
                          pad=3, boxstyle="round,pad=0.35"), zorder=5)

    # ══════════════════════════════════════════════════════
    # CAPA 7 — Stats embed + título
    # ══════════════════════════════════════════════════════
    stats_txt = (f"↑ {gain:.0f}m  ↓ {loss:.0f}m  "
                 f"⬆ {max_ele:.0f}m  ⬇ {min_ele:.0f}m  "
                 f"≈ {total_km:.1f}km  max {max_slope:.1f}%")
    ax.text(0.01, 0.02, stats_txt, transform=ax.transAxes,
            fontsize=6.5, color=text_color, alpha=0.65,
            va="bottom", fontfamily="monospace")

    if chart_title:
        ax.set_title(chart_title, fontsize=13, fontweight="bold",
                     color=text_color, pad=6)

    # ── Leyenda pendiente ──
    if show_slope_heat and show_slope_legend:
        legend_patches = [
            mpatches.Patch(color="#1D4ED8", label="Bajada pronunciada"),
            mpatches.Patch(color="#60A5FA", label="Bajada suave"),
            mpatches.Patch(color="#22C55E", label="Llano / suave (<3%)"),
            mpatches.Patch(color="#FBBF24", label="Moderado (3–8%)"),
            mpatches.Patch(color="#F97316", label="Empinado (8–12%)"),
            mpatches.Patch(color="#DC2626", label="Muy empinado (>12%)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=6.5,
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

    # ── Start / End ──
    for x_pos, name, col in [
        (0.0,      start_loc, "#16A34A"),
        (total_km, end_loc,   "#0F172A"),
    ]:
        if not name:
            continue
        idx = int(np.argmin(np.abs(dist_arr - x_pos)))
        _add(go.Scatter(
            x=[float(dist_arr[idx])], y=[float(ele_display[idx])],
            mode="markers+text", text=[name], textposition="top center",
            marker=dict(color=col, size=12, symbol="diamond",
                        line=dict(color="white", width=2)),
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>{ele_display[idx]:.0f} m<extra></extra>",
        ))

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
    st.markdown("## 🏔️ GPX Altimetry Studio")
    st.caption("v3.0 · Pro Edition")
    st.markdown("---")

    st.markdown("### 📁 Archivo principal")
    uploaded_file = st.file_uploader("GPX principal", type=["gpx"], label_visibility="collapsed")

    st.markdown("### 🔀 Comparar rutas (opcional)")
    uploaded_file2 = st.file_uploader("GPX secundario", type=["gpx"], label_visibility="collapsed",
                                       help="Superpone un segundo perfil para comparar")

    if uploaded_file:
        st.markdown("### 📍 Localidades")
        start_loc   = st.text_input("Salida",   placeholder="Ej. Segovia")
        end_loc     = st.text_input("Llegada",  placeholder="Ej. Madrid")
        chart_title = st.text_input("Título del perfil", placeholder="Ej. Ruta de los Picos")

    st.markdown("### 🎨 Diseño")

    with st.expander("Colores", expanded=True):
        c1, c2 = st.columns(2)
        line_color  = c1.color_picker("Línea",    "#EF4444")
        fill_color  = c2.color_picker("Relleno",  "#FCA5A5")
        bg_color    = c1.color_picker("Fondo",    "#FFFFFF")
        text_color  = c2.color_picker("Texto",    "#374151")
        line_width  = st.slider("Grosor línea", 0.5, 12.0, 2.5, 0.5)
        fill_alpha  = st.slider("Opacidad",     0.0,  1.0, 0.65, 0.05)

    with st.expander("Opciones del gráfico", expanded=True):
        smooth_curve    = st.checkbox("Suavizado",         value=True)
        smooth_strength = st.slider("Intensidad", 3, 51, 7, 2) if smooth_curve else 3
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

    with st.expander("💾 Presets de estilo"):
        presets = {
            "— Sin preset —": None,
            "Montaña Clásica": dict(lc="#EF4444", fc="#FCA5A5", bc="#FFFFFF", tc="#374151"),
            "Minimalista B&N": dict(lc="#000000", fc="#CCCCCC", bc="#FFFFFF", tc="#111111"),
            "Night Mode":      dict(lc="#60A5FA", fc="#1E3A5F", bc="#0F172A", tc="#E2E8F0"),
            "Naturaleza":      dict(lc="#16A34A", fc="#BBF7D0", bc="#F0FDF4", tc="#14532D"),
            "Desierto":        dict(lc="#D97706", fc="#FDE68A", bc="#FFFBEB", tc="#92400E"),
        }
        chosen = st.selectbox("Preset", list(presets.keys()))
        if st.button("Aplicar preset") and presets[chosen]:
            p = presets[chosen]
            line_color, fill_color, bg_color, text_color = p["lc"], p["fc"], p["bc"], p["tc"]

    with st.expander("🌐 Opciones embed WordPress"):
        embed_width  = st.number_input("Ancho embed (px)", 400, 2000, 900, 50)
        embed_height = st.number_input("Alto embed (px)",  200,  800, 420, 20)
        wp_iframe_base_url = st.text_input("URL base (donde subirás el HTML)",
                                           placeholder="https://tudominio.com/wp-content/uploads/")

    with st.expander("📐 Tamaños sociales"):
        social_preset = st.selectbox("Formato", list(SOCIAL_SIZES.keys()))
        if SOCIAL_SIZES[social_preset] is None:
            cw1, cw2 = st.columns(2)
            custom_w = cw1.number_input("Ancho px", 400, 4000, 1200, 50)
            custom_h = cw2.number_input("Alto px",  200, 4000,  630, 50)
        else:
            custom_w, custom_h = SOCIAL_SIZES[social_preset]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏔️ GPX Altimetry Studio Pro")
st.markdown("Generador profesional de perfiles altimétricos para web, redes sociales y WordPress")

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
# MÉTRICAS + ITRA
# ─────────────────────────────────────────────────────────────────────────────
itra = compute_itra(total_km, gain, loss)

mcols = st.columns(7)
mcols[0].metric("📏 Distancia",   f"{total_km:.2f} km")
mcols[1].metric("⬆️ Desnivel +",  f"{gain:.0f} m")
mcols[2].metric("⬇️ Desnivel −",  f"{loss:.0f} m")
mcols[3].metric("🔝 Alt. Máx",    f"{max_ele:.0f} m")
mcols[4].metric("🔽 Alt. Mín",    f"{min_ele:.0f} m")
mcols[5].metric("📐 Pend. Máx",   f"{max_slope:.1f} %")
mcols[6].metric("🏅 ITRA ED",     f"{itra['ed']} ({itra['category']})")

st.markdown(
    f'ITRA: <span class="{itra["badge"]}">{itra["category"]} · {itra["points"]} puntos · ED {itra["ed"]}</span>',
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# WAYPOINTS
# ─────────────────────────────────────────────────────────────────────────────
if "waypoints" not in st.session_state:
    st.session_state["waypoints"] = []

st.markdown('<div class="sec">📍 Puntos de Interés</div>', unsafe_allow_html=True)

qa1, qa2, qa3, qa4, qa5 = st.columns(5)
if qa1.button("🏆 Auto-Cima", use_container_width=True):
    peak_idx = int(np.argmax(ele_display))
    if not any("Cima" in w["label"] for w in st.session_state["waypoints"]):
        st.session_state["waypoints"].append({
            "km": float(dist_arr[peak_idx]), "label": "Cima",
            "ele": float(ele_display[peak_idx]), "icon": "🚩", "icon_key": "🚩 Cima",
        })
        st.rerun()

if qa2.button("📉 Auto-Valle", use_container_width=True):
    vi = int(np.argmin(ele_display))
    st.session_state["waypoints"].append({
        "km": float(dist_arr[vi]), "label": "Valle",
        "ele": float(ele_display[vi]), "icon": "📍", "icon_key": "📍 Genérico",
    })
    st.rerun()

if qa3.button("⛰️ Auto-Puertos", use_container_width=True):
    passes = detect_passes(dist_arr, ele_display)
    added  = 0
    for p in passes:
        if not any(abs(w["km"]-p["km"]) < 0.5 for w in st.session_state["waypoints"]):
            st.session_state["waypoints"].append(p)
            added += 1
    if added:
        st.success(f"✅ {added} puerto(s) detectados y añadidos")
        st.rerun()
    else:
        st.info("No se detectaron puertos significativos.")

if qa4.button("🔀 Ordenar km", use_container_width=True):
    st.session_state["waypoints"].sort(key=lambda x: x["km"])
    st.rerun()

if qa5.button("🗑️ Limpiar todos", use_container_width=True):
    st.session_state["waypoints"] = []
    st.rerun()

# ── Selector + mapa ──
map_km_sel = st.slider("📍 Posición en ruta (km)", 0.0, float(total_km),
                        float(total_km/2), 0.1, key="map_sel")
idx_map = int(np.argmin(np.abs(dist_arr - map_km_sel)))
sel_ele = float(ele_display[idx_map])
sel_lat = float(df_raw.iloc[idx_map]['lat'])
sel_lon = float(df_raw.iloc[idx_map]['lon'])

col_map, col_add = st.columns([3, 1])
with col_map:
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermap(
        mode="lines",
        lon=df_raw['lon'].tolist(), lat=df_raw['lat'].tolist(),
        line={"width": 3, "color": "#3B82F6"}, name="Ruta", hoverinfo="skip",
    ))
    if df_raw2 is not None:
        fig_map.add_trace(go.Scattermap(
            mode="lines",
            lon=df_raw2['lon'].tolist(), lat=df_raw2['lat'].tolist(),
            line={"width": 2, "color": "#F97316", "dash": "dot"}, name="Ruta 2",
        ))
    if st.session_state["waypoints"]:
        wlons = [df_raw.iloc[int(np.argmin(np.abs(dist_arr-w["km"])))]["lon"]
                 for w in st.session_state["waypoints"]]
        wlats = [df_raw.iloc[int(np.argmin(np.abs(dist_arr-w["km"])))]["lat"]
                 for w in st.session_state["waypoints"]]
        fig_map.add_trace(go.Scattermap(
            mode="markers+text",
            lon=wlons, lat=wlats,
            text=[f'{w["icon"]} {w["label"]}' for w in st.session_state["waypoints"]],
            textposition="top right",
            marker={"size": 11, "color": "#F97316"}, name="Waypoints",
        ))
    fig_map.add_trace(go.Scattermap(
        mode="markers", lon=[sel_lon], lat=[sel_lat],
        marker={"size": 16, "color": "#EF4444"}, name="Selección",
        hovertemplate=f"km {map_km_sel:.1f} · {sel_ele:.0f} m<extra></extra>",
    ))
    fig_map.update_layout(
        map={"style": "open-street-map",
             "center": {"lon": sel_lon, "lat": sel_lat}, "zoom": 11},
        showlegend=False, margin={"l":0,"r":0,"b":0,"t":0}, height=310,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_add:
    st.markdown(f"**km {map_km_sel:.1f}** · {sel_ele:.0f} m")
    wp_type   = st.selectbox("Tipo", list(WAYPOINT_DEFS.keys()), key="wp_type")
    wp_name   = st.text_input("Nombre", "Punto", key="wp_name")
    if st.button("➕ Añadir", use_container_width=True, key="btn_add"):
        dup = any(abs(w["km"]-map_km_sel) < 0.05 and w["label"] == wp_name
                  for w in st.session_state["waypoints"])
        if dup:
            st.warning("Punto duplicado.")
        else:
            st.session_state["waypoints"].append({
                "km": map_km_sel, "label": wp_name, "ele": sel_ele,
                "icon": WAYPOINT_DEFS[wp_type]["emoji"], "icon_key": wp_type,
            })
            st.success(f"✅ '{wp_name}' añadido")
            st.rerun()

# Lista con borrar individual
if st.session_state["waypoints"]:
    st.markdown("**Waypoints activos:**")
    for i, wp in enumerate(st.session_state["waypoints"]):
        cc1, cc2 = st.columns([6, 1])
        cc1.markdown(
            f'<span class="wp-chip">{wp["icon"]} <b>{wp["km"]:.1f} km</b>'
            f' — {wp["label"]} <em>({wp["ele"]:.0f} m)</em></span>',
            unsafe_allow_html=True,
        )
        if cc2.button("✕", key=f"del_{i}"):
            st.session_state["waypoints"].pop(i)
            st.rerun()

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# VISTA PREVIA INTERACTIVA  (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">👁️ Vista Previa Interactiva</div>', unsafe_allow_html=True)

html_embed_str = build_html_embed(
    dist_arr, ele_display, df_raw, total_km,
    min_ele, max_ele, gain, loss, max_slope,
    line_color, fill_color, bg_color, text_color,
    line_width, fill_alpha, fill_area,
    show_slope_heat, slope_fill_style,
    show_grid, show_km_markers, km_interval,
    st.session_state["waypoints"],
    start_loc if "start_loc" in dir() else "",
    end_loc   if "end_loc"   in dir() else "",
    chart_title if "chart_title" in dir() else "",
    show_danger_zones, danger_threshold,
    show_slope_subgraph, embed_width, embed_height,
)

# Mostrar en Streamlit como componente HTML
preview_height = max(300, int(800 / aspect_ratio))
st.components.v1.html(html_embed_str, height=preview_height + (120 if show_slope_subgraph else 0))

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

# ─────────────────────────────────────────────────────────────────────────────
# TABLA DE SECTORES  (FEATURE 5)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">📊 Tabla de Sectores</div>', unsafe_allow_html=True)
n_sec = st.slider("Número de sectores", 4, 20, 10, 1)
df_sectors = compute_sectors(dist_arr, ele_display, n_sectors=n_sec)
st.dataframe(df_sectors, use_container_width=True, hide_index=True)

buf_csv = io.StringIO()
df_sectors.to_csv(buf_csv, index=False)
st.download_button("📥 Descargar tabla CSV", buf_csv.getvalue(),
                   "sectores.csv", "text/csv", key="dl_csv")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN ESTÁTICA  (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">💾 Exportar Imagen</div>', unsafe_allow_html=True)

extra_top = 4.5 if label_rotation == "Vertical" else 2.2
if "start_loc" in dir() and start_loc:
    extra_top = max(extra_top, 4.0)

_sl  = start_loc   if "start_loc"   in dir() else ""
_el  = end_loc     if "end_loc"     in dir() else ""
_ct  = chart_title if "chart_title" in dir() else ""

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
st.caption("GPX Altimetry Studio Pro v3.0 · Python + Streamlit + Matplotlib + Plotly")
