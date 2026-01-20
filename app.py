import streamlit as st
import gpxpy
import gpxpy.gpx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="GPX Altimetry Studio Pro", page_icon="🏔️", layout="wide")

# --- FUNCIONES DE ROBUSTEZ Y CÁLCULO ---

def repair_gpx_content(content_str):
    """
    Intenta reparar errores comunes de XML en archivos GPX,
    como prefijos no declarados (ej: gpxx:).
    """
    content_str = re.sub(r'(<|/)[a-zA-Z0-9]+:', r'\1', content_str)
    return content_str

def parse_gpx_robust(uploaded_file):
    """
    Lee el archivo GPX con manejo de errores.
    Devuelve DataFrame y distancia total.
    """
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
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation
                })
    
    if not points:
        for route in gpx.routes:
            for point in route.points:
                 points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation
                })

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
    """Suaviza la elevación usando una media móvil"""
    # window_size debe ser impar y mayor que 0
    if window_size < 1: window_size = 1
    if window_size % 2 == 0: window_size += 1
    
    df['ele_smooth'] = df['ele'].rolling(window=window_size, center=True, min_periods=1).mean()
    return df

def calculate_stats(df, y_col='ele'):
    """Calcula estadísticas: Desnivel, Alturas y Pendiente Máxima"""
    min_ele = df[y_col].min()
    max_ele = df[y_col].max()
    
    # Desnivel
    threshold = 0.5 
    diffs = df[y_col].diff()
    gain = diffs[diffs > threshold].sum()
    
    # Pendiente (Slope) en %
    # Pendiente = (Diferencia Altura / Diferencia Distancia) * 100
    # Convertimos distancia de km a m (*1000)
    dist_diff = df['dist'].diff() * 1000
    ele_diff = df[y_col].diff()
    
    # Evitamos división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = (ele_diff / dist_diff) * 100
    
    # Filtramos pendientes irreales (>35% suele ser error de GPS o precipicio)
    valid_slopes = slopes[slopes.abs() < 35] 
    max_slope = valid_slopes.max() if not valid_slopes.empty else 0
    
    return min_ele, max_ele, gain, max_slope

# --- INTERFAZ ---

st.title("🏔️ GPX Altimetry Studio Pro")
st.markdown("Generador profesional de perfiles altimétricos.")

# Barra lateral
with st.sidebar:
    st.header("1. Archivo y Datos")
    uploaded_file = st.file_uploader("Archivo GPX", type=['gpx'])

    st.header("2. Diseño")
    
    with st.expander("🎨 Colores y Líneas", expanded=True):
        col_c1, col_c2 = st.columns(2)
        line_color = col_c1.color_picker("Línea", "#EF4444")
        fill_color = col_c2.color_picker("Relleno", "#FCA5A5")
        bg_color = st.color_picker("Fondo", "#FFFFFF")
        text_color = st.color_picker("Texto", "#374151")
        
        # NUEVO: Selector de Grosor de Línea
        line_width = st.slider("Grosor de Línea", 0.5, 5.0, 2.0, 0.5)

    with st.expander("⚙️ Opciones del Gráfico", expanded=True):
        smooth_curve = st.checkbox("Activar Suavizado", value=True)
        # NUEVO: Selector de intensidad de suavizado
        if smooth_curve:
            smooth_strength = st.slider("Intensidad de Suavizado", 3, 51, 15, step=2, help="Valores más altos eliminan más ruido del GPS pero pueden aplanar cimas.")
        else:
            smooth_strength = 1

        show_grid = st.checkbox("Mostrar Rejilla", value=True)
        fill_area = st.checkbox("Rellenar Área", value=True)
        
        st.subheader("Proporción de Exportación")
        aspect_ratio = st.slider("Ancho vs Alto", 1.0, 10.0, 4.0, 0.5)
        st.caption(f"Formato: {aspect_ratio}:1")

if uploaded_file is not None:
    # Procesamiento
    with st.spinner('Procesando ruta...'):
        df, total_km = parse_gpx_robust(uploaded_file)
    
    if df is not None:
        # Aplicar suavizado dinámico
        if smooth_curve:
            df = smooth_data(df, window_size=smooth_strength)
            y_col = 'ele_smooth'
        else:
            y_col = 'ele'

        # Stats
        min_ele, max_ele, gain, max_slope = calculate_stats(df, y_col)

        # Mostrar métricas (Añadida Pendiente Máxima)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Distancia", f"{total_km:.2f} km")
        col2.metric("Desnivel +", f"{gain:.0f} m")
        col3.metric("Alt. Máx", f"{max_ele:.0f} m")
        col4.metric("Alt. Mín", f"{min_ele:.0f} m")
        col5.metric("Pendiente Máx", f"{max_slope:.1f}%", help="Pendiente máxima estimada")

        st.divider()

        # --- GESTIÓN DE WAYPOINTS ---
        with st.container():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.subheader("📍 Puntos de Interés")
            
            # NUEVO: Función Útil 3 - Botón Auto-Detectar Cima
            if 'waypoints' not in st.session_state:
                st.session_state.waypoints = []

            with c2:
                if st.button("🏆 Auto-Cima"):
                    # Encontrar el índice del punto más alto
                    peak_idx = df[y_col].idxmax()
                    peak_km = df.loc[peak_idx, 'dist']
                    peak_ele = df.loc[peak_idx, y_col]
                    
                    # Evitar duplicados simples
                    if not any(d['label'] == 'Cima' for d in st.session_state.waypoints):
                        st.session_state.waypoints.append({
                            "km": peak_km,
                            "label": "Cima",
                            "ele": peak_ele
                        })
                        st.rerun()
            
            with c3:
                if st.button("Limpiar Puntos"):
                    st.session_state.waypoints = []
                    st.rerun()

            # Formulario de entrada manual
            with st.form("add_waypoint_form", clear_on_submit=True):
                c_km, c_label, c_sub = st.columns([1, 2, 1])
                new_km = c_km.number_input("Km", min_value=0.0, max_value=total_km, step=0.1)
                new_label = c_label.text_input("Etiqueta (ej. Avituallamiento)")
                submitted = c_sub.form_submit_button("Añadir")
                
                if submitted and new_label:
                    idx = (df['dist'] - new_km).abs().idxmin()
                    ele_at_point = df.loc[idx, y_col]
                    st.session_state.waypoints.append({
                        "km": new_km,
                        "label": new_label,
                        "ele": ele_at_point
                    })
                    st.rerun()

            if st.session_state.waypoints:
                st.write("Puntos activos:")
                cols = st.columns(4)
                for i, wp in enumerate(st.session_state.waypoints):
                    col_idx = i % 4
                    with cols[col_idx]:
                        st.info(f"**{wp['km']:.1f}km**: {wp['label']}")

        st.divider()

        # --- VISUALIZACIÓN INTERACTIVA (PLOTLY) ---
        st.subheader("Vista Previa Interactiva")
        
        fig_interactive = go.Figure()
        
        fig_interactive.add_trace(go.Scatter(
            x=df['dist'], 
            y=df[y_col], 
            mode='lines',
            name='Perfil',
            line=dict(color=line_color, width=line_width), # Usamos el grosor seleccionado
            fill='tozeroy' if fill_area else 'none',
            fillcolor=fill_color if fill_area else None
        ))

        for wp in st.session_state.waypoints:
            fig_interactive.add_trace(go.Scatter(
                x=[wp['km']],
                y=[wp['ele']],
                mode='markers+text',
                text=[wp['label']],
                textposition="top center",
                marker=dict(color=text_color, size=10, symbol='circle'),
                showlegend=False
            ))
            fig_interactive.add_shape(
                type="line",
                x0=wp['km'], y0=min_ele, x1=wp['km'], y1=wp['ele'],
                line=dict(color="gray", width=1, dash="dot"),
            )

        padding = (max_ele - min_ele) * 0.1
        
        fig_interactive.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title='Distancia (km)', showgrid=show_grid, gridcolor='#eee'),
            yaxis=dict(title='Altitud (m)', showgrid=show_grid, gridcolor='#eee', range=[min_ele - padding, max_ele + padding]),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_interactive, use_container_width=True)

        # --- NUEVO: Función Útil 1 - Mapa ---
        with st.expander("🗺️ Ver Mapa de la Ruta"):
            st.map(df, latitude='lat', longitude='lon')

        st.divider()
        
        # --- EXPORTACIÓN ESTÁTICA (MATPLOTLIB) ---
        st.subheader("Descargar Imagen")
        
        base_height = 5
        fig_width = base_height * aspect_ratio
        
        fig_static, ax = plt.subplots(figsize=(fig_width, base_height))
        fig_static.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Plot data usando el grosor seleccionado
        ax.plot(df['dist'], df[y_col], color=line_color, linewidth=line_width, zorder=3)
        
        if fill_area:
            ax.fill_between(df['dist'], df[y_col], min_ele - padding, color=fill_color, alpha=1, zorder=2)

        for wp in st.session_state.waypoints:
            ax.plot([wp['km'], wp['km']], [min_ele - padding, wp['ele']], 
                    color=text_color, linestyle='--', linewidth=1, alpha=0.7, zorder=4)
            
            ax.text(wp['km'], wp['ele'] + padding * 0.5, wp['label'], 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3, boxstyle='round,pad=0.3'), zorder=5)

        ax.set_xlabel("Distancia (km)", color=text_color, fontsize=12, fontweight='bold')
        ax.set_ylabel("Altitud (m)", color=text_color, fontsize=12, fontweight='bold')
        
        ax.set_ylim(min_ele - padding, max_ele + padding)
        ax.set_xlim(0, total_km)
        
        ax.tick_params(colors=text_color, labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
            spine.set_visible(False)
        
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        if show_grid:
            # NUEVO: Rejilla más visible (alpha=0.8)
            ax.grid(True, color='#e5e7eb', linestyle='-', linewidth=0.8, alpha=0.8, zorder=0)

        plt.tight_layout()

        c_d1, c_d2 = st.columns(2)
        fn = uploaded_file.name.replace('.gpx', '')

        buf_png = io.BytesIO()
        fig_static.savefig(buf_png, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
        c_d1.download_button("💾 Descargar PNG", buf_png.getvalue(), f"{fn}_perfil.png", "image/png")

        buf_jpg = io.BytesIO()
        fig_static.savefig(buf_jpg, format='jpg', dpi=150, bbox_inches='tight', facecolor=bg_color)
        c_d2.download_button("💾 Descargar JPG", buf_jpg.getvalue(), f"{fn}_perfil.jpg", "image/jpeg")
        
        plt.close(fig_static)

else:
    st.info("👆 Carga un archivo GPX en el menú lateral para empezar.")
