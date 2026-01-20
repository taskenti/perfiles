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
    # 1. Eliminar prefijos de namespaces no declarados (ej: <gpxx:Track> -> <Track>)
    # Esto es un regex agresivo pero efectivo para leer la geometría básica
    content_str = re.sub(r'(<|/)[a-zA-Z0-9]+:', r'\1', content_str)
    return content_str

def parse_gpx_robust(uploaded_file):
    """
    Lee el archivo GPX con manejo de errores y reintentos.
    Devuelve un DataFrame y la distancia total.
    """
    # Leemos el contenido una vez
    content_bytes = uploaded_file.getvalue()
    
    try:
        # Intento 1: Decodificar utf-8 estándar y parsear
        content_str = content_bytes.decode('utf-8')
        gpx = gpxpy.parse(content_str)
    except Exception as e1:
        try:
            # Intento 2: Reparar XML (eliminar namespaces problemáticos)
            content_str = content_bytes.decode('utf-8', errors='ignore')
            clean_content = repair_gpx_content(content_str)
            gpx = gpxpy.parse(clean_content)
        except Exception as e2:
            st.error(f"No se pudo leer el archivo GPX. Error crítico: {e2}")
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
        # A veces los GPX son rutas (routes) y no tracks
        for route in gpx.routes:
            for point in route.points:
                 points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation
                })

    if not points:
        st.error("El archivo GPX no contiene puntos de track ni de ruta con elevación.")
        return None, 0
        
    df = pd.DataFrame(points)
    
    # Rellenar elevaciones nulas si las hubiera
    if df['ele'].isnull().any():
        df['ele'] = df['ele'].interpolate().fillna(0)

    # Calcular distancias acumuladas
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
    """Suaviza la elevación usando una media móvil simple para reducir ruido"""
    df['ele_smooth'] = df['ele'].rolling(window=window_size, center=True, min_periods=1).mean()
    return df

def calculate_stats(df):
    """Calcula estadísticas filtrando ruido menor"""
    min_ele = df['ele'].min()
    max_ele = df['ele'].max()
    
    # Cálculo de ganancia con umbral (threshold) para evitar sumar el "jitter" del GPS
    threshold = 0.5 # metros
    diffs = df['ele'].diff()
    gain = diffs[diffs > threshold].sum()
    loss = diffs[diffs < -threshold].sum()
    
    return min_ele, max_ele, gain, abs(loss)

# --- INTERFAZ ---

st.title("🏔️ GPX Altimetry Studio Pro")
st.markdown("Generador profesional de perfiles altimétricos.")

# Barra lateral
with st.sidebar:
    st.header("1. Archivo y Datos")
    uploaded_file = st.file_uploader("Archivo GPX", type=['gpx'])

    st.header("2. Diseño")
    
    with st.expander("🎨 Colores", expanded=True):
        col_c1, col_c2 = st.columns(2)
        line_color = col_c1.color_picker("Línea", "#EF4444")
        fill_color = col_c2.color_picker("Relleno", "#FCA5A5")
        bg_color = st.color_picker("Fondo Imagen", "#FFFFFF")
        text_color = st.color_picker("Texto/Ejes", "#374151")

    with st.expander("⚙️ Opciones del Gráfico", expanded=True):
        smooth_curve = st.checkbox("Suavizar Curva", value=True)
        show_grid = st.checkbox("Mostrar Rejilla", value=True)
        fill_area = st.checkbox("Rellenar Área", value=True)
        
        st.subheader("Proporción de Exportación")
        aspect_ratio = st.slider("Ancho vs Alto", 1.0, 10.0, 4.0, 0.5, help="4.0 significa que la imagen será 4 veces más ancha que alta")
        st.caption(f"Formato: {aspect_ratio}:1")

if uploaded_file is not None:
    # Procesamiento
    with st.spinner('Procesando ruta...'):
        df, total_km = parse_gpx_robust(uploaded_file)
    
    if df is not None:
        # Aplicar suavizado si se selecciona (para visualización y stats)
        # Usamos una columna separada para no perder los datos crudos
        if smooth_curve:
            df = smooth_data(df, window_size=15) # Ventana más grande para que se note
            y_col = 'ele_smooth'
        else:
            y_col = 'ele'

        # Stats
        min_ele, max_ele, gain, loss = calculate_stats(df)

        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Distancia", f"{total_km:.2f} km")
        col2.metric("Desnivel +", f"{gain:.0f} m")
        col3.metric("Alt. Máx", f"{max_ele:.0f} m")
        col4.metric("Alt. Mín", f"{min_ele:.0f} m")

        st.divider()

        # --- GESTIÓN DE WAYPOINTS ---
        # Contenedor para que la interfaz no salte al añadir
        with st.container():
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("📍 Puntos de Interés")
            with c2:
                if st.button("Limpiar todos los puntos"):
                    st.session_state.waypoints = []
                    st.rerun()

            if 'waypoints' not in st.session_state:
                st.session_state.waypoints = []

            # Formulario de entrada
            with st.form("add_waypoint_form", clear_on_submit=True):
                c_km, c_label, c_sub = st.columns([1, 2, 1])
                new_km = c_km.number_input("Km", min_value=0.0, max_value=total_km, step=0.1)
                new_label = c_label.text_input("Etiqueta (ej. Cima)")
                submitted = c_sub.form_submit_button("Añadir")
                
                if submitted and new_label:
                    # Buscar elevación más cercana
                    idx = (df['dist'] - new_km).abs().idxmin()
                    ele_at_point = df.loc[idx, y_col]
                    st.session_state.waypoints.append({
                        "km": new_km,
                        "label": new_label,
                        "ele": ele_at_point
                    })
                    st.rerun()

            # Lista de waypoints (tags)
            if st.session_state.waypoints:
                st.write("Puntos activos:")
                cols = st.columns(4)
                for i, wp in enumerate(st.session_state.waypoints):
                    col_idx = i % 4
                    with cols[col_idx]:
                        st.info(f"**{wp['km']}km**: {wp['label']}")


        st.divider()

        # --- VISUALIZACIÓN INTERACTIVA (PLOTLY) ---
        # Usamos Plotly para la pantalla porque es interactivo (como JS)
        st.subheader("Vista Previa Interactiva")
        
        fig_interactive = go.Figure()
        
        # Línea principal
        fig_interactive.add_trace(go.Scatter(
            x=df['dist'], 
            y=df[y_col], 
            mode='lines',
            name='Perfil',
            line=dict(color=line_color, width=3),
            fill='tozeroy' if fill_area else 'none',
            fillcolor=fill_color if fill_area else None
        ))

        # Waypoints en Plotly
        for wp in st.session_state.waypoints:
            fig_interactive.add_trace(go.Scatter(
                x=[wp['km']],
                y=[wp['ele']],
                mode='markers+text',
                text=[wp['label']],
                textposition="top center",
                marker=dict(color=text_color, size=10, symbol='circle'),
                name='Waypoint',
                showlegend=False
            ))
            # Línea vertical para waypoint
            fig_interactive.add_shape(
                type="line",
                x0=wp['km'], y0=min_ele, x1=wp['km'], y1=wp['ele'],
                line=dict(color="gray", width=1, dash="dot"),
            )

        # Ajuste adaptativo de ejes
        padding = (max_ele - min_ele) * 0.1
        
        fig_interactive.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', # Transparente para adaptarse al tema de Streamlit
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title='Distancia (km)', showgrid=show_grid, gridcolor='#eee'),
            yaxis=dict(title='Altitud (m)', showgrid=show_grid, gridcolor='#eee', range=[min_ele - padding, max_ele + padding]),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_interactive, use_container_width=True)

        
        # --- EXPORTACIÓN ESTÁTICA (MATPLOTLIB) ---
        # Usamos Matplotlib para generar el archivo de descarga porque:
        # 1. Permite controlar el Aspect Ratio exacto (4:1) al píxel.
        # 2. No depende de librerías binarias complejas (Kaleido) en el servidor.
        
        st.subheader("Descargar Imagen")
        
        # Preparar figura de Matplotlib (Backend Agg)
        base_height = 5
        fig_width = base_height * aspect_ratio
        
        fig_static, ax = plt.subplots(figsize=(fig_width, base_height))
        fig_static.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Plot data
        ax.plot(df['dist'], df[y_col], color=line_color, linewidth=2, zorder=3)
        
        if fill_area:
            ax.fill_between(df['dist'], df[y_col], min_ele - padding, color=fill_color, alpha=1, zorder=2)

        # Plot Waypoints estáticos
        for wp in st.session_state.waypoints:
            ax.plot([wp['km'], wp['km']], [min_ele - padding, wp['ele']], 
                    color=text_color, linestyle='--', linewidth=1, alpha=0.7, zorder=4)
            
            # Etiqueta mejorada
            ax.text(wp['km'], wp['ele'] + padding * 0.5, wp['label'], 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_color,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3, boxstyle='round,pad=0.3'), zorder=5)

        # Configuración Ejes Estáticos
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
            ax.grid(True, color='#e5e7eb', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)

        plt.tight_layout()

        # Botones de descarga
        c_d1, c_d2 = st.columns(2)
        fn = uploaded_file.name.replace('.gpx', '')

        # PNG Buffer
        buf_png = io.BytesIO()
        fig_static.savefig(buf_png, format='png', dpi=150, bbox_inches='tight', facecolor=bg_color)
        c_d1.download_button("💾 Descargar PNG", buf_png.getvalue(), f"{fn}_perfil.png", "image/png")

        # JPG Buffer
        buf_jpg = io.BytesIO()
        fig_static.savefig(buf_jpg, format='jpg', dpi=150, bbox_inches='tight', facecolor=bg_color)
        c_d2.download_button("💾 Descargar JPG", buf_jpg.getvalue(), f"{fn}_perfil.jpg", "image/jpeg")
        
        # Cerrar figura para liberar memoria
        plt.close(fig_static)

else:
    # Mensaje de bienvenida
    st.info("👆 Carga un archivo GPX en el menú lateral para empezar.")
