import streamlit as st
import gpxpy
import gpxpy.gpx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import io
import re
import textwrap
import hashlib
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GPX Altimetry Studio Pro",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para una UI más pulida
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stMetric { background: #f8fafc; border-radius: 10px; padding: 12px 16px; border: 1px solid #e2e8f0; }
    .stMetric label { font-size: 0.75rem !important; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .stMetric [data-testid="metric-container"] > div:nth-child(2) { font-size: 1.6rem !important; font-weight: 600; color: #0f172a; }
    div[data-testid="stSidebar"] { background: #0f172a; }
    div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stCheckbox label { color: #94a3b8 !important; font-size: 0.82rem; }
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-weight: 600; }
    .stButton button { border-radius: 8px; font-weight: 500; transition: all 0.15s; }
    .waypoint-chip { display: inline-flex; align-items: center; gap: 6px; background: #f1f5f9;
                     border: 1px solid #e2e8f0; border-radius: 20px; padding: 4px 12px;
                     font-size: 0.82rem; color: #334155; margin: 3px; }
    .section-title { font-size: 1rem; font-weight: 600; color: #0f172a;
                     border-left: 3px solid #ef4444; padding-left: 10px; margin: 12px 0 8px 0; }
    .stat-warning { background: #fef9c3; border: 1px solid #fde047;
                    border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; color: #713f12; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES: WAYPOINTS UNIFICADOS (icono + estilo MPL en una sola estructura)
# ─────────────────────────────────────────────
WAYPOINT_DEFS = {
    "📍 Genérico":        {"emoji": "📍", "marker": "o",  "color": "#EF4444", "edge": "#991B1B",  "size": 12},
    "💧 Fuente":          {"emoji": "💧", "marker": "o",  "color": "#06B6D4", "edge": "#0E7490",  "size": 12},
    "🏠 Refugio":         {"emoji": "🏠", "marker": "s",  "color": "#92400E", "edge": "#451A03",  "size": 13},
    "🏘️ Pueblo":          {"emoji": "🏘️", "marker": "h",  "color": "#F97316", "edge": "#9A3412",  "size": 14},
    "🌉 Puente":          {"emoji": "🌉", "marker": "d",  "color": "#64748B", "edge": "#334155",  "size": 12},
    "🥪 Avituallamiento": {"emoji": "🥪", "marker": "P",  "color": "#22C55E", "edge": "#15803D",  "size": 13},
    "📷 Foto":            {"emoji": "📷", "marker": "p",  "color": "#A855F7", "edge": "#6B21A8",  "size": 12},
    "🚩 Cima":            {"emoji": "🚩", "marker": "^",  "color": "#DC2626", "edge": "#7F1D1D",  "size": 14},
    "⛰️ Puerto":          {"emoji": "⛰️", "marker": "D",  "color": "#64748B", "edge": "#1E293B",  "size": 13},
    "⚠️ Peligro":         {"emoji": "⚠️", "marker": "X",  "color": "#FBBF24", "edge": "#B45309",  "size": 13},
    "🅿️ Parking":         {"emoji": "🅿️", "marker": "s",  "color": "#3B82F6", "edge": "#1E40AF",  "size": 12},
    "🌲 Bosque":          {"emoji": "🌲", "marker": "^",  "color": "#16A34A", "edge": "#14532D",  "size": 12},
    "⛪ Iglesia":          {"emoji": "⛪", "marker": "P",  "color": "#8B5CF6", "edge": "#5B21B6",  "size": 13},
    "🏔️ Mirador":         {"emoji": "🏔️", "marker": "*",  "color": "#F59E0B", "edge": "#92400E",  "size": 15},
    "🚰 Punto de Agua":   {"emoji": "🚰", "marker": "o",  "color": "#0EA5E9", "edge": "#0369A1",  "size": 12},
    "🏥 Primeros Aux.":   {"emoji": "🏥", "marker": "+",  "color": "#F43F5E", "edge": "#9F1239",  "size": 14},
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def hex_to_rgba(hex_code: str, alpha: float) -> str:
    """Convierte '#RRGGBB' → 'rgba(r,g,b,a)' para Plotly. Seguro ante inputs mal formados."""
    hex_code = hex_code.strip().lstrip('#')
    if len(hex_code) != 6:
        return f"rgba(239,68,68,{alpha})"  # fallback rojo
    try:
        r, g, b = int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except ValueError:
        return f"rgba(239,68,68,{alpha})"


def file_hash(uploaded_file) -> str:
    """Devuelve un hash MD5 del contenido del archivo para detectar cambios."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def repair_gpx_content(content_str: str) -> str:
    """
    Elimina prefijos de namespace XML problemáticos.
    Solo actúa sobre namespaces no estándar; preserva los de extensiones Garmin/Suunto
    que contienen datos adicionales (HR, cadencia, etc.).
    """
    # Solo elimina namespaces que no sean gpxtpx, gpxx (Garmin) ni ns3
    return re.sub(r'(</?)[a-zA-Z0-9]+:(?!gpxtpx|gpxx|ns3)', r'\1', content_str)


@st.cache_data(show_spinner=False)
def parse_gpx_cached(file_bytes: bytes) -> tuple[pd.DataFrame | None, float, list]:
    """
    Parsea el GPX y devuelve (df, total_km, waypoints_gpx).
    Cacheado por contenido de bytes para evitar re-parsear en cada interacción.
    También extrae <wpt> del propio GPX si los tiene.
    """
    try:
        content_str = file_bytes.decode('utf-8')
        gpx = gpxpy.parse(content_str)
    except Exception:
        try:
            content_str = file_bytes.decode('utf-8', errors='ignore')
            content_str = repair_gpx_content(content_str)
            gpx = gpxpy.parse(content_str)
        except Exception as e:
            return None, 0.0, []

    # ── Extraer track points ──
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append({'lat': pt.latitude, 'lon': pt.longitude, 'ele': pt.elevation})

    if not points:
        for route in gpx.routes:
            for pt in route.points:
                points.append({'lat': pt.latitude, 'lon': pt.longitude, 'ele': pt.elevation})

    if not points:
        return None, 0.0, []

    df = pd.DataFrame(points)

    # ── Validar elevaciones ──
    null_count = df['ele'].isnull().sum()
    if null_count == len(df):
        # Todas nulas → aviso crítico (lo gestionamos en el caller)
        df['ele'] = 0.0
        df['ele_corrupted'] = True
    else:
        df['ele'] = df['ele'].interpolate().fillna(0)
        df['ele_corrupted'] = False

    # ── Calcular distancias acumuladas ──
    distances = [0.0]
    total_dist = 0.0
    for i in range(1, len(points)):
        d = gpxpy.geo.haversine_distance(
            points[i-1]['lat'], points[i-1]['lon'],
            points[i]['lat'], points[i]['lon']
        )
        total_dist += d / 1000.0
        distances.append(total_dist)
    df['dist'] = distances

    # ── Decimar si la ruta es muy larga (>20.000 puntos) ──
    if len(df) > 20_000:
        step = len(df) // 10_000
        df = df.iloc[::step].reset_index(drop=True)
        # Recalcular distancias tras decimación
        distances2 = [0.0]
        for i in range(1, len(df)):
            d = gpxpy.geo.haversine_distance(
                df.iloc[i-1]['lat'], df.iloc[i-1]['lon'],
                df.iloc[i]['lat'], df.iloc[i]['lon']
            )
            distances2.append(distances2[-1] + d / 1000.0)
        df['dist'] = distances2
        total_dist = distances2[-1]

    # ── Extraer waypoints del GPX (<wpt> tags) ──
    gpx_waypoints = []
    for wpt in gpx.waypoints:
        if wpt.latitude and wpt.longitude:
            # Buscar la distancia más cercana al track
            dists_to_wpt = ((df['lat'] - wpt.latitude)**2 + (df['lon'] - wpt.longitude)**2).values
            idx_closest = int(np.argmin(dists_to_wpt))
            km_pos = df.iloc[idx_closest]['dist']
            ele = wpt.elevation if wpt.elevation else df.iloc[idx_closest]['ele']
            gpx_waypoints.append({
                "km": round(km_pos, 2),
                "label": wpt.name or "Waypoint",
                "ele": round(float(ele), 1),
                "icon": "📍",
                "icon_key": "📍 Genérico",
            })

    return df, total_dist, gpx_waypoints


@st.cache_data(show_spinner=False)
def smooth_data_cached(ele_array: np.ndarray, dist_array: np.ndarray, window_size: int) -> np.ndarray:
    """Suaviza elevaciones. Cacheado por contenido numérico."""
    if window_size < 3:
        return ele_array.copy()
    # Garantizar window_size impar
    if window_size % 2 == 0:
        window_size += 1
    s = pd.Series(ele_array)
    return s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()


def calculate_stats(dist: np.ndarray, ele: np.ndarray, gain_threshold: float = 0.5):
    """
    Calcula estadísticas de la ruta usando los datos actualmente visibles (suavizados o no).
    Retorna: min_ele, max_ele, gain, loss, max_slope
    """
    min_ele = float(ele.min())
    max_ele = float(ele.max())

    diffs = np.diff(ele)
    gain = float(diffs[diffs > gain_threshold].sum())
    loss = float(abs(diffs[diffs < -gain_threshold].sum()))

    dist_diff = np.diff(dist) * 1000  # metros
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.where(dist_diff > 0, (diffs / dist_diff) * 100, 0)
    valid_slopes = slopes[np.abs(slopes) < 35]
    max_slope = float(valid_slopes.max()) if len(valid_slopes) > 0 else 0.0

    return min_ele, max_ele, gain, loss, max_slope


def compute_slope_colors(dist: np.ndarray, ele: np.ndarray) -> np.ndarray:
    """
    Devuelve un array de colores RGBA por segmento basado en la pendiente:
    verde → llano, amarillo → moderado, rojo → empinado, azul → bajada
    """
    dist_diff = np.diff(dist) * 1000
    ele_diff = np.diff(ele)
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.where(dist_diff > 0, (ele_diff / dist_diff) * 100, 0.0)

    colors = []
    for s in slopes:
        if s >= 10:
            colors.append('#DC2626')      # rojo - muy empinado
        elif s >= 6:
            colors.append('#F97316')      # naranja - empinado
        elif s >= 3:
            colors.append('#FBBF24')      # amarillo - moderado
        elif s >= -2:
            colors.append('#22C55E')      # verde - llano
        elif s >= -6:
            colors.append('#60A5FA')      # azul claro - bajada suave
        else:
            colors.append('#1D4ED8')      # azul fuerte - bajada pronunciada
    return np.array(colors)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏔️ GPX Altimetry Studio")
    st.markdown("---")

    st.markdown("### 1 · Archivo")
    uploaded_file = st.file_uploader("Archivo GPX", type=['gpx'], label_visibility="collapsed")

    if uploaded_file:
        st.markdown("### 2 · Localidades")
        start_loc = st.text_input("📍 Salida", placeholder="Ej. Segovia")
        end_loc = st.text_input("🏁 Llegada", placeholder="Ej. Madrid")
        chart_title = st.text_input("📝 Título del perfil", placeholder="Ej. Ruta de los Picos")

    st.markdown("### 3 · Diseño")

    with st.expander("🎨 Colores", expanded=True):
        col_c1, col_c2 = st.columns(2)
        line_color  = col_c1.color_picker("Línea",   "#EF4444")
        fill_color  = col_c2.color_picker("Relleno", "#FCA5A5")
        bg_color    = col_c1.color_picker("Fondo",   "#FFFFFF")
        text_color  = col_c2.color_picker("Texto",   "#374151")
        line_width  = st.slider("Grosor de línea",   0.5, 12.0, 2.5, 0.5)
        fill_alpha  = st.slider("Opacidad relleno",  0.0,  1.0, 0.6, 0.05)

    with st.expander("⚙️ Opciones del gráfico", expanded=True):
        smooth_curve = st.checkbox("Suavizado", value=True)
        smooth_strength = 5
        if smooth_curve:
            smooth_strength = st.slider("Intensidad suavizado", 3, 51, 5, step=2)

        show_grid       = st.checkbox("Rejilla",             value=True)
        fill_area       = st.checkbox("Rellenar área",       value=True)
        show_slope_heat = st.checkbox("Heatmap de pendiente",value=False,
                                      help="Colorea la línea según la inclinación de cada segmento")
        show_km_markers = st.checkbox("Marcadores de km",    value=True)
        km_interval     = st.slider("Intervalo km markers",  1, 20, 5, 1) if show_km_markers else 5

        label_rotation  = st.radio("Etiquetas waypoints", ["Horizontal", "Vertical"],
                                   index=1, horizontal=True)

        st.markdown("**Proporción exportación**")
        aspect_ratio = st.slider("Ancho / Alto", 1.0, 10.0, 4.0, 0.5)
        st.caption(f"Formato {aspect_ratio:.1f} : 1")

    with st.expander("💾 Presets de estilo"):
        preset_names = ["— Sin preset —", "Montaña Clásica", "Minimalista B&N", "Night Mode", "Naturaleza"]
        chosen_preset = st.selectbox("Cargar preset", preset_names)
        if st.button("Aplicar preset") and chosen_preset != "— Sin preset —":
            presets = {
                "Montaña Clásica":  dict(lc="#EF4444", fc="#FCA5A5", bc="#FFFFFF", tc="#374151"),
                "Minimalista B&N":  dict(lc="#000000", fc="#CCCCCC", bc="#FFFFFF", tc="#111111"),
                "Night Mode":       dict(lc="#60A5FA", fc="#1E3A5F", bc="#0F172A", tc="#E2E8F0"),
                "Naturaleza":       dict(lc="#16A34A", fc="#BBF7D0", bc="#F0FDF4", tc="#14532D"),
            }
            p = presets[chosen_preset]
            st.session_state["preset_lc"] = p["lc"]
            st.session_state["preset_fc"] = p["fc"]
            st.session_state["preset_bc"] = p["bc"]
            st.session_state["preset_tc"] = p["tc"]
            st.rerun()

# ─────────────────────────────────────────────
# CUERPO PRINCIPAL
# ─────────────────────────────────────────────
st.title("🏔️ GPX Altimetry Studio Pro")
st.markdown("Generador profesional de perfiles altimétricos · [Carga un GPX en el panel lateral]")

if uploaded_file is None:
    st.info("👆 Carga un archivo GPX en el menú lateral para empezar.")
    st.stop()

# ── Detectar cambio de archivo → resetear waypoints ──
current_hash = file_hash(uploaded_file)
if st.session_state.get("last_file_hash") != current_hash:
    st.session_state["last_file_hash"] = current_hash
    st.session_state["waypoints"] = []
    st.session_state["gpx_wpts_imported"] = False

# ── Parsear GPX ──
with st.spinner("Procesando ruta…"):
    df_raw, total_km, gpx_native_wpts = parse_gpx_cached(uploaded_file.getvalue())

if df_raw is None:
    st.error("No se pudo leer el archivo GPX. Comprueba que contiene tracks o routes con puntos.")
    st.stop()

# Aviso si las elevaciones estaban corruptas
if df_raw['ele_corrupted'].iloc[0]:
    st.warning("⚠️ El archivo GPX no contiene datos de elevación válidos. Se han rellenado con 0 m.")

# ── Suavizado (cacheado por separado) ──
if smooth_curve:
    ele_display = smooth_data_cached(
        df_raw['ele'].to_numpy(), df_raw['dist'].to_numpy(), smooth_strength
    )
else:
    ele_display = df_raw['ele'].to_numpy()

dist_arr = df_raw['dist'].to_numpy()

# ── Estadísticas usando los datos visualizados ──
min_ele, max_ele, gain, loss, max_slope = calculate_stats(dist_arr, ele_display)

# ── Oferta de importar waypoints del GPX nativo ──
if gpx_native_wpts and not st.session_state.get("gpx_wpts_imported"):
    col_imp1, col_imp2 = st.columns([3, 1])
    col_imp1.info(
        f"🗂️ El archivo GPX contiene **{len(gpx_native_wpts)} waypoint(s)** integrados. "
        "¿Deseas importarlos?"
    )
    if col_imp2.button("📥 Importar waypoints GPX"):
        for w in gpx_native_wpts:
            if not any(abs(e["km"] - w["km"]) < 0.05 for e in st.session_state["waypoints"]):
                st.session_state["waypoints"].append(w)
        st.session_state["gpx_wpts_imported"] = True
        st.rerun()

# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("📏 Distancia",     f"{total_km:.2f} km")
m2.metric("⬆️ Desnivel +",    f"{gain:.0f} m")
m3.metric("⬇️ Desnivel −",    f"{loss:.0f} m")
m4.metric("🔝 Alt. Máx",      f"{max_ele:.0f} m")
m5.metric("🔽 Alt. Mín",      f"{min_ele:.0f} m")
m6.metric("📐 Pend. Máx",     f"{max_slope:.1f} %")

st.divider()

# ─────────────────────────────────────────────
# GESTIÓN DE WAYPOINTS
# ─────────────────────────────────────────────
if "waypoints" not in st.session_state:
    st.session_state["waypoints"] = []

st.markdown('<div class="section-title">📍 Puntos de Interés</div>', unsafe_allow_html=True)

# ── Acciones rápidas ──
qa1, qa2, qa3, qa4 = st.columns(4)
with qa1:
    if st.button("🏆 Auto-Cima", use_container_width=True):
        peak_idx = int(np.argmax(ele_display))
        peak_km  = float(dist_arr[peak_idx])
        peak_ele = float(ele_display[peak_idx])
        if not any("Cima" in w["label"] for w in st.session_state["waypoints"]):
            st.session_state["waypoints"].append({
                "km": peak_km, "label": "Cima", "ele": peak_ele,
                "icon": "🚩", "icon_key": "🚩 Cima",
            })
            st.success("✅ Cima añadida automáticamente")
            st.rerun()
        else:
            st.info("La cima ya existe.")

with qa2:
    if st.button("📉 Auto-Valle", use_container_width=True):
        valley_idx = int(np.argmin(ele_display))
        st.session_state["waypoints"].append({
            "km":  float(dist_arr[valley_idx]),
            "label": "Valle",
            "ele": float(ele_display[valley_idx]),
            "icon": "🌊", "icon_key": "📍 Genérico",
        })
        st.rerun()

with qa3:
    if st.button("🗑️ Limpiar todos", use_container_width=True):
        st.session_state["waypoints"] = []
        st.rerun()

with qa4:
    # Ordenar waypoints por km
    if st.button("🔀 Ordenar por km", use_container_width=True):
        st.session_state["waypoints"].sort(key=lambda x: x["km"])
        st.rerun()

st.info("Usa el slider para posicionar el punto en la ruta, luego cúbreme con nombre y tipo.")

# ── Selector de posición ──
map_km_sel = st.slider("📍 Posición en ruta (km)", 0.0, float(total_km), float(total_km / 2), 0.1,
                        key="map_selector")
idx_map    = int(np.argmin(np.abs(dist_arr - map_km_sel)))
sel_ele    = float(ele_display[idx_map])
sel_lat    = float(df_raw.iloc[idx_map]['lat'])
sel_lon    = float(df_raw.iloc[idx_map]['lon'])

# ── Layout: mapa + formulario ──
col_map, col_add = st.columns([3, 1])

with col_map:
    fig_map = go.Figure()
    # Traza de la ruta
    fig_map.add_trace(go.Scattermapbox(
        mode="lines",
        lon=df_raw['lon'].tolist(),
        lat=df_raw['lat'].tolist(),
        line={"width": 3, "color": "#3B82F6"},
        name="Ruta",
        hoverinfo="skip",
    ))
    # Waypoints ya añadidos
    if st.session_state["waypoints"]:
        wlons = [df_raw.iloc[int(np.argmin(np.abs(dist_arr - w["km"])))]["lon"]
                 for w in st.session_state["waypoints"]]
        wlats = [df_raw.iloc[int(np.argmin(np.abs(dist_arr - w["km"])))]["lat"]
                 for w in st.session_state["waypoints"]]
        wtext = [f"{w['icon']} {w['label']}" for w in st.session_state["waypoints"]]
        fig_map.add_trace(go.Scattermapbox(
            mode="markers+text",
            lon=wlons, lat=wlats,
            text=wtext, textposition="top right",
            marker={"size": 11, "color": "#F97316"},
            name="Waypoints",
        ))
    # Punto seleccionado
    fig_map.add_trace(go.Scattermapbox(
        mode="markers",
        lon=[sel_lon], lat=[sel_lat],
        marker={"size": 16, "color": "#EF4444"},
        name="Selección",
        hovertemplate=f"<b>km {map_km_sel:.1f}</b><br>{sel_ele:.0f} m<extra></extra>",
    ))
    fig_map.update_layout(
        mapbox={"style": "open-street-map",
                "center": {"lon": sel_lon, "lat": sel_lat},
                "zoom": 11},
        showlegend=False,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        height=310,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_add:
    st.markdown(f"**km {map_km_sel:.1f}** · {sel_ele:.0f} m")
    map_icon_key  = st.selectbox("Tipo", list(WAYPOINT_DEFS.keys()), key="wp_type_sel")
    label_new     = st.text_input("Nombre", value="Punto", key="wp_name_in")
    if st.button("➕ Añadir punto", key="btn_add_wp", use_container_width=True):
        # Evitar duplicados exactos
        if any(abs(w["km"] - map_km_sel) < 0.05 and w["label"] == label_new
               for w in st.session_state["waypoints"]):
            st.warning("Punto duplicado, no añadido.")
        else:
            st.session_state["waypoints"].append({
                "km":      map_km_sel,
                "label":   label_new,
                "ele":     sel_ele,
                "icon":    WAYPOINT_DEFS[map_icon_key]["emoji"],
                "icon_key": map_icon_key,
            })
            st.success(f"✅ '{label_new}' añadido")
            st.rerun()

# ── Lista de waypoints con botón de eliminar individual ──
if st.session_state["waypoints"]:
    st.markdown("**Waypoints activos:**")
    for i, wp in enumerate(st.session_state["waypoints"]):
        c_info, c_del = st.columns([5, 1])
        with c_info:
            st.markdown(
                f'<span class="waypoint-chip">{wp["icon"]} '
                f'<strong>{wp["km"]:.1f} km</strong> — {wp["label"]} '
                f'<em>({wp["ele"]:.0f} m)</em></span>',
                unsafe_allow_html=True,
            )
        with c_del:
            if st.button("✕", key=f"del_wp_{i}", help=f"Eliminar '{wp['label']}'"):
                st.session_state["waypoints"].pop(i)
                st.rerun()

st.divider()

# ─────────────────────────────────────────────
# VISTA PREVIA INTERACTIVA (Plotly)
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">👁️ Vista Previa Interactiva</div>', unsafe_allow_html=True)

padding    = (max_ele - min_ele) * 0.15
fig_iactive = go.Figure()

if show_slope_heat and not fill_area:
    # Trazar segmento a segmento con color de pendiente
    seg_colors = compute_slope_colors(dist_arr, ele_display)
    for i in range(len(dist_arr) - 1):
        fig_iactive.add_trace(go.Scatter(
            x=[dist_arr[i], dist_arr[i+1]],
            y=[ele_display[i], ele_display[i+1]],
            mode="lines",
            line=dict(color=seg_colors[i], width=line_width),
            showlegend=False,
            hoverinfo="skip",
        ))
else:
    plotly_fill_color = hex_to_rgba(fill_color, fill_alpha) if fill_area else None
    fig_iactive.add_trace(go.Scatter(
        x=dist_arr.tolist(),
        y=ele_display.tolist(),
        mode="lines",
        line=dict(color=line_color, width=line_width),
        fill="tozeroy" if fill_area else "none",
        fillcolor=plotly_fill_color,
        hovertemplate="km %{x:.2f}<br>%{y:.0f} m<extra></extra>",
    ))

# Marcadores de km
if show_km_markers:
    km_marks = np.arange(km_interval, total_km, km_interval)
    for km_mark in km_marks:
        idx_m = int(np.argmin(np.abs(dist_arr - km_mark)))
        fig_iactive.add_vline(
            x=float(dist_arr[idx_m]),
            line=dict(color="#94a3b8", width=1, dash="dot"),
        )
        fig_iactive.add_annotation(
            x=float(dist_arr[idx_m]),
            y=min_ele - padding * 0.4,
            text=f"{km_mark:.0f}",
            showarrow=False,
            font=dict(size=9, color="#94a3b8"),
        )

# Waypoints en el preview con estilo sincronizado
for wp in st.session_state["waypoints"]:
    style   = WAYPOINT_DEFS.get(wp["icon_key"], WAYPOINT_DEFS["📍 Genérico"])
    wrapped = "<br>".join(textwrap.wrap(wp["label"], width=15))
    fig_iactive.add_trace(go.Scatter(
        x=[wp["km"]], y=[wp["ele"]],
        mode="markers+text",
        text=[f'{style["emoji"]} {wrapped}'],
        textposition="top center",
        marker=dict(color=style["color"], size=10, symbol="circle",
                    line=dict(color=style["edge"], width=2)),
        showlegend=False,
        hovertemplate=f"<b>{wp['label']}</b><br>{wp['km']:.1f} km · {wp['ele']:.0f} m<extra></extra>",
    ))

preview_height = max(300, int(800 / aspect_ratio))
fig_iactive.update_layout(
    height=preview_height,
    title=dict(text=chart_title if uploaded_file and "chart_title" in dir() else "",
               font=dict(size=15), x=0.5) if uploaded_file else {},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=40 if (uploaded_file and chart_title) else 20, b=0),
    xaxis=dict(title="Distancia (km)", showgrid=show_grid, gridcolor="#e2e8f0",
               zeroline=False),
    yaxis=dict(title="Altitud (m)", showgrid=show_grid, gridcolor="#e2e8f0",
               zeroline=False,
               range=[min_ele - padding, max_ele + padding * 2]),
    hovermode="x unified",
)
st.plotly_chart(fig_iactive, use_container_width=True)

# ─────────────────────────────────────────────
# LEYENDA DE PENDIENTE
# ─────────────────────────────────────────────
if show_slope_heat:
    st.markdown("""
    **Leyenda pendiente:** &nbsp;
    🟢 Llano (&lt;3%) &nbsp;|&nbsp;
    🟡 Moderado (3–6%) &nbsp;|&nbsp;
    🟠 Empinado (6–10%) &nbsp;|&nbsp;
    🔴 Muy empinado (&gt;10%) &nbsp;|&nbsp;
    🔵 Bajada suave &nbsp;|&nbsp;
    💙 Bajada pronunciada
    """)

st.divider()

# ─────────────────────────────────────────────
# EXPORTACIÓN ESTÁTICA (Matplotlib)
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">💾 Descargar imagen</div>', unsafe_allow_html=True)

base_height = 5.0
fig_width   = base_height * aspect_ratio

# ── Calcular margen superior dinámico según longitud máxima de etiquetas ──
max_label_len = max(
    (len(wp["label"]) for wp in st.session_state["waypoints"]),
    default=0,
)
# En modo vertical necesitamos más espacio proporcional al texto
extra_top_factor = 4.5 if label_rotation == "Vertical" else 2.0
# También añadir margen si hay start_loc/end_loc
if uploaded_file and ("start_loc" in dir() and start_loc):
    extra_top_factor = max(extra_top_factor, 4.0)

fig_static, ax = plt.subplots(figsize=(fig_width, base_height))
fig_static.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# ── Heatmap de pendiente o línea sólida ──
if show_slope_heat:
    seg_colors_mpl = compute_slope_colors(dist_arr, ele_display)
    for i in range(len(dist_arr) - 1):
        ax.plot(
            [dist_arr[i], dist_arr[i+1]],
            [ele_display[i], ele_display[i+1]],
            color=seg_colors_mpl[i],
            linewidth=line_width,
            solid_capstyle="round",
            zorder=3,
        )
else:
    ax.plot(
        dist_arr, ele_display,
        color=line_color, linewidth=line_width,
        zorder=3, solid_capstyle="round", solid_joinstyle="round",
    )

if fill_area:
    ax.fill_between(dist_arr, ele_display, min_ele - padding,
                    color=fill_color, alpha=fill_alpha, zorder=2)

# ── Marcadores de km ──
if show_km_markers:
    km_marks = np.arange(km_interval, total_km, km_interval)
    for km_mark in km_marks:
        idx_m = int(np.argmin(np.abs(dist_arr - km_mark)))
        ax.axvline(x=float(dist_arr[idx_m]), color=text_color,
                   linestyle=":", linewidth=0.7, alpha=0.35, zorder=1)
        ax.text(float(dist_arr[idx_m]), min_ele - padding * 0.6,
                f"{km_mark:.0f}", ha="center", va="top",
                fontsize=7, color=text_color, alpha=0.6)

# ── Configuración de etiquetas ──
rotation_deg = 90 if label_rotation == "Vertical" else 0
y_offset_label = padding * (1.0 if label_rotation == "Vertical" else 0.5)

# ── Localidad de inicio ──
if "start_loc" in dir() and start_loc:
    s_ele = float(ele_display[0])
    ax.plot(0, s_ele, marker="D", color="#16A34A", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.text(0, s_ele + y_offset_label, start_loc,
            ha="left", va="bottom",
            rotation=90,            # inicio siempre vertical para evitar salirse por eje Y
            fontsize=9, fontweight="bold", color=text_color,
            bbox=dict(facecolor=bg_color, alpha=0.92, edgecolor="none",
                      pad=3, boxstyle="round,pad=0.35"),
            zorder=5)

# ── Localidad de fin ──
if "end_loc" in dir() and end_loc:
    e_ele = float(ele_display[-1])
    ax.plot(total_km, e_ele, marker="D", color="#0F172A", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    ax.text(total_km, e_ele + y_offset_label, end_loc,
            ha="right", va="bottom",
            rotation=90,            # fin siempre vertical
            fontsize=9, fontweight="bold", color=text_color,
            bbox=dict(facecolor=bg_color, alpha=0.92, edgecolor="none",
                      pad=3, boxstyle="round,pad=0.35"),
            zorder=5)

# ── Waypoints intermedios ──
for wp in st.session_state["waypoints"]:
    # Línea vertical punteada
    ax.plot([wp["km"], wp["km"]], [min_ele - padding, wp["ele"]],
            color=text_color, linestyle=":", linewidth=1.0, alpha=0.4, zorder=4)
    # Marcador
    style = WAYPOINT_DEFS.get(wp["icon_key"], WAYPOINT_DEFS["📍 Genérico"])
    ax.plot(wp["km"], wp["ele"],
            marker=style["marker"],
            color=style["color"],
            markersize=style["size"],
            markeredgecolor=style["edge"],
            markeredgewidth=2.0,
            zorder=6)
    # Etiqueta
    wrapped = "\n".join(textwrap.wrap(wp["label"], width=14))
    ax.text(wp["km"], wp["ele"] + y_offset_label, wrapped,
            ha="center", va="bottom",
            rotation=rotation_deg,
            fontsize=8, fontweight="bold", color=text_color,
            bbox=dict(facecolor=bg_color, alpha=0.92, edgecolor="none",
                      pad=3, boxstyle="round,pad=0.35"),
            zorder=5)

# ── Estadísticas en la esquina del gráfico ──
stats_text = (
    f"↑ {gain:.0f} m   ↓ {loss:.0f} m   "
    f"⬆ {max_ele:.0f} m   ⬇ {min_ele:.0f} m   "
    f"≈ {total_km:.1f} km   max {max_slope:.1f}%"
)
ax.text(0.01, 0.02, stats_text,
        transform=ax.transAxes, fontsize=7.5,
        color=text_color, alpha=0.7, va="bottom",
        fontfamily="monospace")

# ── Título ──
if "chart_title" in dir() and chart_title:
    ax.set_title(chart_title, fontsize=14, fontweight="bold",
                 color=text_color, pad=8)

# ── Ejes ──
ax.set_xlabel("Distancia (km)", color=text_color, fontsize=11, fontweight="bold")
ax.set_ylabel("Altitud (m)",    color=text_color, fontsize=11, fontweight="bold")
ax.tick_params(colors=text_color, labelsize=9)

# Margen superior dinámico para evitar que las etiquetas verticales se corten
ax.set_ylim(min_ele - padding, max_ele + padding * extra_top_factor)

# Margen lateral: mayor derecho para la etiqueta del punto final
right_margin = 1.06 if ("end_loc" in dir() and end_loc) else 1.02
ax.set_xlim(0, total_km * right_margin)

# ── Bordes ──
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_edgecolor(text_color)
ax.spines["left"].set_edgecolor(text_color)
ax.spines["bottom"].set_linewidth(1.1)
ax.spines["left"].set_linewidth(1.1)

if show_grid:
    ax.grid(True, color="#9ca3af", linestyle="-", linewidth=0.7, alpha=0.45, zorder=0)

# ── Layout y ajuste de márgenes para evitar recortes ──
plt.tight_layout(pad=0.8)
# Margen superior explícito para etiquetas largas
fig_static.subplots_adjust(top=0.88 if ("chart_title" in dir() and chart_title) else 0.92)

# ── Botones de descarga ──
fn = uploaded_file.name.replace(".gpx", "")

buf_png = io.BytesIO()
fig_static.savefig(buf_png, format="png", dpi=200,
                   bbox_inches="tight", facecolor=bg_color, edgecolor="none")
buf_png.seek(0)

buf_jpg = io.BytesIO()
fig_static.savefig(buf_jpg, format="jpg", dpi=200,
                   bbox_inches="tight", facecolor=bg_color, edgecolor="none",
                   pil_kwargs={"quality": 95})
buf_jpg.seek(0)

buf_svg = io.BytesIO()
fig_static.savefig(buf_svg, format="svg",
                   bbox_inches="tight", facecolor=bg_color, edgecolor="none")
buf_svg.seek(0)

plt.close(fig_static)

dl1, dl2, dl3 = st.columns(3)
dl1.download_button("💾 Descargar PNG", buf_png.getvalue(),
                    f"{fn}_perfil.png",  "image/png",       key="dl_png")
dl2.download_button("💾 Descargar JPG", buf_jpg.getvalue(),
                    f"{fn}_perfil.jpg",  "image/jpeg",      key="dl_jpg")
dl3.download_button("💾 Descargar SVG", buf_svg.getvalue(),
                    f"{fn}_perfil.svg",  "image/svg+xml",   key="dl_svg")

st.caption(
    "GPX Altimetry Studio Pro · "
    "Perfiles generados con Python + Streamlit + Matplotlib + Plotly"
)
