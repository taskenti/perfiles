import streamlit as st
import gpxpy
import gpxpy.gpx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re
import textwrap
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="GPX Altimetry Studio Pro", page_icon="🏔️", layout="wide")

# --- CONFIGURACIÓN DE ICONOS Y MARCADORES MEJORADOS ---
WAYPOINT_ICONS = {
    "📍 Genérico": "📍",
    "💧 Fuente": "💧",
    "🏠 Refugio": "🏠",
    "🏘️ Pueblo": "🏘️",
    "🌉 Puente": "🌉",
    "🥪 Avituallamiento": "🥪",
    "📷 Foto": "📷",
    "🚩 Cima": "🚩",
    "⛰️ Puerto": "⛰️",
    "⚠️ Peligro": "⚠️",
    "🅿️ Parking": "🅿️",
    "🌲 Bosque": "🌲",
    "⛪ Iglesia": "⛪",
    "🏔️ Mirador": "🏔️"
}

# Estilos profesionales para Matplotlib con gradientes visuales
MPL_STYLES = {
    "📍 Genérico": {"marker": "o", "color": "#EF4444", "size": 12, "edge": "#991B1B"},
    "💧 Fuente": {"marker": "o", "color": "#06B6D4", "size": 12, "edge": "#0E7490"},
    "🏠 Refugio": {"marker": "s", "color": "#92400E", "size": 13, "edge": "#451A03"},
    "🏘️ Pueblo": {"marker": "h", "color": "#F97316", "size": 14, "edge": "#9A3412"},
    "🌉 Puente": {"marker": "d", "color": "#64748B", "size": 12, "edge": "#334155"},
    "🥪 Avituallamiento": {"marker": "P", "color": "#22C55E", "size": 13, "edge": "#15803D"},
    "📷 Foto": {"marker": "p", "color": "#A855F7", "size": 12, "edge": "#6B21A8"},
    "🚩 Cima": {"marker": "^", "color": "#DC2626", "size": 14, "edge": "#7F1D1D"},
    "⛰️ Puerto": {"marker": "D", "color": "#64748B", "size": 13, "edge": "#1E293B"},
    "⚠️ Peligro": {"marker": "X", "color": "#FBBF24", "size": 13, "edge": "#B45309"},
    "🅿️ Parking": {"marker": "s", "color": "#3B82F6", "size": 12, "edge": "#1E40AF"},
    "🌲 Bosque": {"marker": "^", "color": "#16A34A", "size": 12, "edge": "#14532D"},
    "⛪ Iglesia": {"marker": "P", "color": "#8B5CF6", "size": 13, "edge": "#5B21B6"},
    "🏔️ Mirador": {"marker": "*", "color": "#F59E0B", "size": 15, "edge": "#92400E"}
}

# --- FUNCIONES AUXILIARES ---

def hex_to_rgba(hex_code, alpha):
    """Convierte hex (#RRGGBB) a cadena rgba(r, g, b, a) para Plotly"""
    hex_code = hex_code.lstrip('#')
    return f"rgba({int(hex_code[0:2], 16)}, {int(hex_code[2:4], 16)}, {int(hex_code[4:6], 16)}, {alpha})"

def repair_gpx_content(content_str):
    content_str = re.sub(r'(<|/)[a-zA-Z0-9]+:', r'\1', content_str)
    return content_str

def parse_gpx_robust(uploaded_file):
    content_bytes = uploaded_file.getvalue()
    try:
        content_str = content_bytes.decode('utf-8')
        gpx = gpxpy.parse(content_str)
    except Exception:
        try:
            content_str = content_bytes.decode('utf-8', errors='ignore')
            clean_content = repair_gpx_content(content_str)
            gpx = gpxpy.parse(clean_content)
        except Exception as e2:
            st.error(f"No se pudo leer el archivo GPX. Error: {e2}")
            return None, 0

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({'lat': point.latitude, 'lon': point.longitude, 'ele': point.elevation})
    
    if not points:
        for route in gpx.routes:
            for point in route.points:
                 points.append({'lat': point.latitude, 'lon': point.longitude, 'ele': point.elevation})

    if not points:
        st.error("El archivo GPX no contiene datos de elevación válidos.")
        return None, 0
        
    df = pd.DataFrame(points)
    if df['ele'].isnull().any():
        df['ele'] = df['ele'].interpolate().fillna(0)

    distances = [0]
    total_dist = 0
    for i in range(1, len(points)):
        d = gpxpy.geo.haversine_distance(
            points[i-1]['lat'], points[i-1]['lon'],
            points[i]['lat'], points[i]['lon']
        )
        total_dist += (d / 1000)
        distances.append(total_dist)
    
    df['dist'] = distances
    return df, total_dist

def smooth_data(df, window_size=5):
    if len(df) < window_size:
        window_size = max(1, len(df) // 2)
        if window_size % 2 == 0: window_size += 1
    if window_size < 1: window_size = 1
    df['ele_smooth'] = df['ele'].rolling(window=window_size, center=True, min_periods=1).mean()
    return df

def calculate_stats(df, y_col='ele'):
    min_ele = df[y_col].min()
    max_ele = df[y_col].max()
    threshold = 0.5 
    diffs = df[y_col].diff()
    gain = diffs[diffs > threshold].sum()
    dist_diff = df['dist'].diff() * 1000
    ele_diff = df[y_col].diff()
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = (ele_diff / dist_diff) * 100
    valid_slopes = slopes[slopes.abs() < 35] 
    max_slope = valid_slopes.max() if not valid_slopes.empty else 0
    return min_ele, max_ele, gain, max_slope

# --- INTERFAZ ---

st.title("🏔️ GPX Altimetry Studio Pro")
st.markdown("Generador profesional de perfiles altimétricos.")

with st.sidebar:
    st.header("1. Archivo y Datos")
    uploaded_file = st.file_uploader("Archivo GPX", type=['gpx'])

    # Localidades Inicio/Fin
    if uploaded_file:
        st.subheader("📍 Localidades")
        start_loc = st.text_input("Salida (Inicio)", placeholder="Ej. Madrid")
        end_loc = st.text_input("Llegada (Fin)", placeholder="Ej. Segovia")

    st.header("2. Diseño")
    
    with st.expander("🎨 Colores y Estilo", expanded=True):
        col_c1, col_c2 = st.columns(2)
        line_color = col_c1.color_picker("Línea", "#EF4444")
        fill_color = col_c2.color_picker("Relleno", "#FCA5A5")
        bg_color = st.color_picker("Fondo", "#FFFFFF")
        text_color = st.color_picker("Texto", "#374151")
        st.divider()
        line_width = st.slider("Grosor de Línea", 0.5, 12.0, 2.5, 0.5)
        fill_alpha = st.slider("Opacidad Relleno", 0.0, 1.0, 0.6, step=0.05)

    with st.expander("⚙️ Opciones del Gráfico", expanded=True):
        smooth_curve = st.checkbox("Activar Suavizado", value=True)
        if smooth_curve:
            smooth_strength = st.slider("Intensidad de Suavizado", 1, 21, 5, step=2)
        else:
            smooth_strength = 1

        show_grid = st.checkbox("Mostrar Rejilla", value=True)
        fill_area = st.checkbox("Rellenar Área", value=True)
        
        st.write("---")
        label_rotation = st.radio("Orientación Etiquetas (Waypoints)", ["Horizontal", "Vertical"], index=0, horizontal=True)
        
        st.subheader("Proporción de Exportación")
        aspect_ratio = st.slider("Ancho vs Alto", 1.0, 10.0, 4.0, 0.5)
        st.caption(f"Formato: {aspect_ratio}:1")

if uploaded_file is not None:
    with st.spinner('Procesando ruta...'):
        df, total_km = parse_gpx_robust(uploaded_file)
    
    if df is not None:
        if smooth_curve:
            df = smooth_data(df, window_size=smooth_strength)
            y_col = 'ele_smooth'
        else:
            y_col = 'ele'

        min_ele, max_ele, gain, max_slope = calculate_stats(df, y_col)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Distancia", f"{total_km:.2f} km")
        col2.metric("Desnivel +", f"{gain:.0f} m")
        col3.metric("Alt. Máx", f"{max_ele:.0f} m")
        col4.metric("Alt. Mín", f"{min_ele:.0f} m")
        col5.metric("Pendiente Máx", f"{max_slope:.1f}%")

        st.divider()

        # --- GESTIÓN DE WAYPOINTS ---
        if 'waypoints' not in st.session_state:
            st.session_state.waypoints = []

        with st.container():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.subheader("📍 Puntos de Interés")
            with c2:
                if st.button("🏆 Auto-Cima"):
                    peak_idx = df[y_col].idxmax()
                    peak_km = df.loc[peak_idx, 'dist']
                    peak_ele = df.loc[peak_idx, y_col]
                    if not any("Cima" in d['label'] for d in st.session_state.waypoints):
                        st.session_state.waypoints.append({
                            "km": peak_km, "label": "Cima", "ele": peak_ele, 
                            "icon_key": "🚩 Cima", "icon": "🚩" 
                        })
                        st.rerun()
            with c3:
                if st.button("Limpiar Puntos"):
                    st.session_state.waypoints = []
                    st.rerun()

            st.info("Desliza para elegir la posición y añade el punto.")
            
            map_km_sel = st.slider("Posición en Ruta (km)", 0.0, total_km, total_km/2, 0.1, key="map_selector")
            idx_map = (df['dist'] - map_km_sel).abs().idxmin()
            sel_point = df.loc[idx_map]
            
            col_map, col_add = st.columns([3, 1])
            with col_map:
                fig_map = go.Figure()
                fig_map.add_trace(go.Scattermap(
                    mode="lines", lon=df['lon'], lat=df['lat'],
                    marker={'size': 10}, line={'width': 3, 'color': 'blue'}, name="Ruta"
                ))
                fig_map.add_trace(go.Scattermap(
                    mode="markers", lon=[sel_point['lon']], lat=[sel_point['lat']],
                    marker={'size': 15, 'color': 'red'}, name="Selección"
                ))
                fig_map.update_layout(
                    map={'style': "open-street-map", 'center': {'lon': sel_point['lon'], 'lat': sel_point['lat']}, 'zoom': 11},
                    showlegend=False, margin={'l':0, 'r':0, 'b':0, 't':0}, height=300
                )
                st.plotly_chart(fig_map, use_container_width=True)
            
            with col_add:
                st.write(f"**Km:** {map_km_sel:.1f}")
                st.write(f"**Alt:** {sel_point[y_col]:.0f}m")
                
                map_icon_key = st.selectbox("Tipo Punto", list(WAYPOINT_ICONS.keys()), key="map_icon_sel")
                map_icon_emoji = WAYPOINT_ICONS[map_icon_key]
                
                label_map = st.text_input("Nombre Punto", value="Punto", key="map_label_in")
                
                if st.button("📍 Añadir", key="btn_add_wp"):
                    st.session_state.waypoints.append({
                        "km": map_km_sel, "label": label_map, "ele": sel_point[y_col], 
                        "icon": map_icon_emoji, "icon_key": map_icon_key
                    })
                    st.rerun()

            if st.session_state.waypoints:
                st.write("---")
                cols = st.columns(4)
                for i, wp in enumerate(st.session_state.waypoints):
                    col_idx = i % 4
                    with cols[col_idx]:
                        icon_display = wp.get('icon', '📍')
                        st.info(f"{icon_display} **{wp['km']:.1f}km**: {wp['label']}")

        st.divider()

        # --- VISTA PREVIA INTERACTIVA ---
        st.subheader("Vista Previa Interactiva")
        fig_interactive = go.Figure()
        
        plotly_fill_color = hex_to_rgba(fill_color, fill_alpha) if fill_area else None

        fig_interactive.add_trace(go.Scatter(
            x=df['dist'], y=df[y_col], mode='lines',
            line=dict(color=line_color, width=line_width),
            fill='tozeroy' if fill_area else 'none',
            fillcolor=plotly_fill_color
        ))

        for wp in st.session_state.waypoints:
            icon_txt = wp.get('icon', '📍')
            wrapped_label_plotly = "<br>".join(textwrap.wrap(wp['label'], width=15))
            full_label = f"{icon_txt} {wrapped_label_plotly}"
            
            fig_interactive.add_trace(go.Scatter(
                x=[wp['km']], y=[wp['ele']], mode='markers+text',
                text=[full_label], textposition="top center",
                marker=dict(color=text_color, size=10, symbol='circle'), showlegend=False
            ))

        padding = (max_ele - min_ele) * 0.1
        
        preview_height = int(800 / aspect_ratio)
        if preview_height < 300: preview_height = 300 
        
        fig_interactive.update_layout(
            height=preview_height,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title='Distancia (km)', showgrid=show_grid, gridcolor='#eee'),
            yaxis=dict(title='Altitud (m)', showgrid=show_grid, gridcolor='#eee', range=[min_ele - padding, max_ele + padding]),
        )
        st.plotly_chart(fig_interactive, use_container_width=True)

        st.divider()
        
        # --- EXPORTACIÓN ESTÁTICA OPTIMIZADA ---
        st.subheader("Descargar Imagen")
        
        base_height = 5
        fig_width = base_height * aspect_ratio
        
        fig_static, ax = plt.subplots(figsize=(fig_width, base_height))
        fig_static.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Línea del perfil con antialiasing
        ax.plot(df['dist'], df[y_col], color=line_color, linewidth=line_width, 
                zorder=3, solid_capstyle='round', solid_joinstyle='round')
        
        if fill_area:
            ax.fill_between(df['dist'], df[y_col], min_ele - padding, 
                            color=fill_color, alpha=fill_alpha, zorder=2)

        # Configuración para texto de waypoints - OPTIMIZADO
        rotation_deg = 90 if label_rotation == "Vertical" else 0
        
        if label_rotation == "Vertical":
            vertical_align = 'bottom'  
            horizontal_align = 'center'
            # SEPARACIÓN CORREGIDA: Mucho más cerca del punto
            y_offset_label = padding * 0.5 
        else:
            vertical_align = 'bottom'
            horizontal_align = 'center'
            y_offset_label = padding * 0.5

        # LOCALIDADES INICIO/FIN - CORRECCIÓN DE "DESTROZO"
        # Usamos alineación 'left' para Inicio y 'right' para Fin
        # Esto empuja el texto hacia DENTRO del gráfico, evitando que pise el eje Y
        if start_loc:
            start_ele = df[y_col].iloc[0]
            ax.plot(0, start_ele, marker='D', color='green', markersize=9, 
                    markeredgecolor='white', markeredgewidth=1.5, zorder=6)
            
            ax.text(0, start_ele + y_offset_label, start_loc, 
                    ha='left',  # Alineado a la izquierda (el texto sale hacia la derecha)
                    va='bottom', rotation=90,
                    fontsize=10, fontweight='bold', color=text_color,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', 
                              pad=4, boxstyle='round,pad=0.4'), zorder=5)

        if end_loc:
            end_ele = df[y_col].iloc[-1]
            ax.plot(total_km, end_ele, marker='D', color='black', markersize=9,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=6)
            
            ax.text(total_km, end_ele + y_offset_label, end_loc, 
                    ha='right', # Alineado a la derecha (el texto sale hacia la izquierda)
                    va='bottom', rotation=90,
                    fontsize=10, fontweight='bold', color=text_color,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', 
                              pad=4, boxstyle='round,pad=0.4'), zorder=5)

        # WAYPOINTS INTERMEDIOS - ESTILO MEJORADO
        for wp in st.session_state.waypoints:
            # Línea vertical punteada más suave
            ax.plot([wp['km'], wp['km']], [min_ele - padding, wp['ele']], 
                    color=text_color, linestyle=':', linewidth=1.2, alpha=0.5, zorder=4)
            
            icon_key = wp.get('icon_key', "📍 Genérico")
            style = MPL_STYLES.get(icon_key, MPL_STYLES["📍 Genérico"])
            
            # Marcador con borde de color para efecto profesional
            ax.plot(wp['km'], wp['ele'], 
                    marker=style['marker'], 
                    color=style['color'], 
                    markersize=style.get('size', 12), 
                    markeredgecolor=style.get('edge', 'white'), 
                    markeredgewidth=2,
                    zorder=6)
            
            wrapped_label = "\n".join(textwrap.wrap(wp['label'], width=15))
            
            ax.text(wp['km'], wp['ele'] + y_offset_label, wrapped_label, 
                    ha=horizontal_align, va=vertical_align, 
                    rotation=rotation_deg,
                    fontsize=9, fontweight='bold', color=text_color,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', 
                              pad=4, boxstyle='round,pad=0.4'), 
                    zorder=5)

        # Ejes y etiquetas
        ax.set_xlabel("Distancia (km)", color=text_color, fontsize=13, fontweight='bold')
        ax.set_ylabel("Altitud (m)", color=text_color, fontsize=13, fontweight='bold')
        
        # Margen superior optimizado para textos verticales
        ax.set_ylim(min_ele - padding, max_ele + padding * 2.8)
        
        # Márgenes laterales ajustados para que quepan los marcadores de inicio/fin
        ax.set_xlim(-total_km * 0.02, total_km * 1.02)
        
        ax.tick_params(colors=text_color, labelsize=10)
        
        # Bordes limpios
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
            spine.set_linewidth(1.2)
            spine.set_visible(False)
        
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # MEJORADO: Rejilla más visible
        if show_grid:
            ax.grid(True, color='#9ca3af', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)

        plt.tight_layout(pad=0.5)

        c_d1, c_d2 = st.columns(2)
        fn = uploaded_file.name.replace('.gpx', '')

        buf_png = io.BytesIO()
        fig_static.savefig(buf_png, format='png', dpi=200, bbox_inches='tight', 
                          facecolor=bg_color, edgecolor='none')
        c_d1.download_button("💾 Descargar PNG", buf_png.getvalue(), 
                             f"{fn}_perfil.png", "image/png", key="dl_png")

        buf_jpg = io.BytesIO()
        fig_static.savefig(buf_jpg, format='jpg', dpi=200, bbox_inches='tight', 
                          facecolor=bg_color, edgecolor='none', pil_kwargs={'quality': 95})
        c_d2.download_button("💾 Descargar JPG", buf_jpg.getvalue(), 
                             f"{fn}_perfil.jpg", "image/jpeg", key="dl_jpg")
        
        plt.close(fig_static)

else:
    st.info("👆 Carga un archivo GPX en el menú lateral para empezar.")
