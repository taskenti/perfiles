import streamlit as st
import gpxpy
import gpxpy.gpx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Configuración de la página
st.set_page_config(page_title="GPX Altimetry Studio", page_icon="🏔️", layout="wide")

# --- FUNCIONES ---

def parse_gpx(file):
    """Lee el archivo GPX y devuelve un DataFrame con distancia y elevación"""
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation
                })
    
    df = pd.DataFrame(points)
    
    # Calcular distancias acumuladas
    distances = [0]
    total_dist = 0
    for i in range(1, len(points)):
        # gpxpy tiene una función precisa para calcular distancia entre puntos
        d = gpxpy.geo.haversine_distance(
            points[i-1]['lat'], points[i-1]['lon'],
            points[i]['lat'], points[i]['lon']
        )
        total_dist += (d / 1000) # Convertir a Km
        distances.append(total_dist)
    
    df['dist'] = distances
    return df, total_dist

# --- INTERFAZ ---

st.title("🏔️ GPX Altimetry Studio")
st.markdown("Genera perfiles altimétricos limpios y exportables.")

# Barra lateral de configuración
with st.sidebar:
    st.header("1. Cargar Archivo")
    uploaded_file = st.file_uploader("Sube tu archivo GPX", type=['gpx'])

    st.header("2. Configuración Gráfica")
    col_color1, col_color2 = st.columns(2)
    line_color = col_color1.color_picker("Línea", "#EF4444")
    fill_color = col_color2.color_picker("Relleno", "#FCA5A5")
    
    bg_color = st.color_picker("Fondo Gráfico", "#FFFFFF")
    text_color = st.color_picker("Texto", "#374151")
    
    st.divider()
    
    show_grid = st.checkbox("Mostrar Rejilla", value=True)
    fill_area = st.checkbox("Rellenar Área", value=True)
    
    st.subheader("Proporción (Ancho:Alto)")
    aspect_ratio = st.slider("Relación de aspecto", 2.0, 8.0, 4.0, 0.5)
    st.caption(f"Formato actual: {aspect_ratio}:1")

# Lógica Principal
if uploaded_file is not None:
    try:
        # Procesar datos
        df, total_km = parse_gpx(uploaded_file)
        
        # Calcular estadísticas básicas
        min_ele = df['ele'].min()
        max_ele = df['ele'].max()
        gain = 0 
        # Cálculo simple de desnivel positivo
        diffs = df['ele'].diff()
        gain = diffs[diffs > 0].sum()

        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Distancia", f"{total_km:.2f} km")
        col2.metric("Desnivel +", f"{gain:.0f} m")
        col3.metric("Alt. Máx", f"{max_ele:.0f} m")
        col4.metric("Alt. Mín", f"{min_ele:.0f} m")

        st.divider()

        # --- GESTIÓN DE WAYPOINTS ---
        st.subheader("📍 Puntos de Interés (Waypoints)")
        
        # Usamos session_state para mantener los waypoints entre recargas
        if 'waypoints' not in st.session_state:
            st.session_state.waypoints = []

        c_km, c_label, c_btn = st.columns([1, 2, 1])
        with c_km:
            new_km = st.number_input("Km", min_value=0.0, max_value=total_km, step=0.5, key="wp_km")
        with c_label:
            new_label = st.text_input("Etiqueta", placeholder="Ej. Cima", key="wp_label")
        with c_btn:
            st.write("") # Espaciador vertical
            st.write("")
            if st.button("Añadir Punto"):
                if new_label:
                    # Buscar elevación en ese Km
                    # Buscamos el índice donde la distancia es más cercana
                    idx = (df['dist'] - new_km).abs().idxmin()
                    ele_at_point = df.loc[idx, 'ele']
                    st.session_state.waypoints.append({
                        "km": new_km,
                        "label": new_label,
                        "ele": ele_at_point
                    })
        
        # Mostrar waypoints activos
        if st.session_state.waypoints:
            st.write("Puntos activos:")
            for i, wp in enumerate(st.session_state.waypoints):
                c1, c2, c3 = st.columns([1, 3, 1])
                c1.write(f"**{wp['km']} km**")
                c2.write(wp['label'])
                if c3.button("Borrar", key=f"del_{i}"):
                    st.session_state.waypoints.pop(i)
                    st.rerun()

        st.divider()

        # --- GENERACIÓN DEL GRÁFICO (Matplotlib) ---
        
        # Definir tamaño basado en aspect ratio
        # Fijamos una altura base de 4 pulgadas, el ancho será 4 * ratio
        base_height = 4
        fig_width = base_height * aspect_ratio
        
        fig, ax = plt.subplots(figsize=(fig_width, base_height))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Gráfico principal
        ax.plot(df['dist'], df['ele'], color=line_color, linewidth=2, zorder=3)
        
        if fill_area:
            ax.fill_between(df['dist'], df['ele'], min_ele*0.9, color=fill_color, alpha=0.5, zorder=2)

        # Waypoints
        for wp in st.session_state.waypoints:
            ax.plot([wp['km'], wp['km']], [min_ele*0.9, wp['ele']], 
                    color=text_color, linestyle='--', linewidth=1, alpha=0.5, zorder=4)
            
            # Etiqueta con fondo blanco semitransparente
            ax.text(wp['km'], wp['ele'] + (max_ele-min_ele)*0.05, wp['label'], 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=text_color,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2), zorder=5)

        # Configuración de Ejes
        ax.set_xlabel("Distancia (km)", color=text_color, fontweight='bold')
        ax.set_ylabel("Altitud (m)", color=text_color, fontweight='bold')
        
        # Ajuste de límites (Adaptive Y-axis)
        padding = (max_ele - min_ele) * 0.1
        ax.set_ylim(min_ele - padding, max_ele + padding)
        ax.set_xlim(0, total_km)
        
        # Estilo de ticks y bordes
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(text_color)
            spine.set_visible(False) # Estilo limpio, sin caja
        
        # Recuperar ejes X e Y (bottom y left)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        if show_grid:
            ax.grid(True, color='#e5e7eb', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)

        plt.tight_layout()
        
        # Mostrar en pantalla
        st.pyplot(fig)

        # --- EXPORTACIÓN ---
        col_down1, col_down2 = st.columns(2)
        
        # Guardar en buffer para descargar
        fn = uploaded_file.name.replace('.gpx', '')
        
        # PNG
        img_buffer_png = io.BytesIO()
        fig.savefig(img_buffer_png, format='png', dpi=300, bbox_inches='tight', facecolor=bg_color)
        col_down1.download_button(
            label="Descargar PNG",
            data=img_buffer_png.getvalue(),
            file_name=f"{fn}_perfil.png",
            mime="image/png"
        )
        
        # JPG
        img_buffer_jpg = io.BytesIO()
        fig.savefig(img_buffer_jpg, format='jpg', dpi=300, bbox_inches='tight', facecolor=bg_color)
        col_down2.download_button(
            label="Descargar JPG",
            data=img_buffer_jpg.getvalue(),
            file_name=f"{fn}_perfil.jpg",
            mime="image/jpeg"
        )

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

else:
    st.info("👆 Sube un archivo GPX en la barra lateral para comenzar.")
