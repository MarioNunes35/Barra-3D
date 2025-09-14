# -*- coding: utf-8 -*-
"""
Streamlit – 3D Bar Chart (Enhanced Export Version)
Author: Enhanced Version
Date: 2025-09-14

Melhorias principais:
- Área de visualização expandida e responsiva
- Exportação em alta resolução com múltiplas opções
- Controles avançados de qualidade de exportação
- Suporte para diferentes aspectos e resoluções predefinidas
"""

import io
import base64
from typing import Optional, List, Tuple
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Configuração da Página
# -----------------------------
st.set_page_config(
    page_title="Barras 3D Interativo - HD Export",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para maximizar área de visualização
st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    [data-testid="stSidebar"] {
        min-width: 250px;
        max-width: 350px;
    }
    .stPlotlyChart {
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Utilidades
# -----------------------------
def sample_dataframe(n_x=4, n_y=4, seed=42) -> pd.DataFrame:
    """Cria dados de exemplo no formato LONGO."""
    rng = np.random.default_rng(seed)
    xs = [f"Grupo {i+1}" for i in range(n_x)]
    ys = [f"Série {j+1}" for j in range(n_y)]
    data = []
    for x in xs:
        for y in ys:
            z = float(rng.uniform(10, 100))
            e = float(rng.uniform(1, 10))
            data.append([x, y, z, e])
    return pd.DataFrame(data, columns=["X", "Y", "Z", "Err"])

def ensure_min_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Garante ao menos 3 colunas (X,Y,Z)."""
    if df is None or df.empty or df.shape[1] < 3:
        return sample_dataframe()
    return df

def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def degrees_to_plotly_camera(elev, azim, dist=1.75):
    """Converte elevação/azimute para o vetor 'eye' do Plotly."""
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    x = dist * np.cos(elev_rad) * np.cos(azim_rad)
    y = dist * np.cos(elev_rad) * np.sin(azim_rad)
    z = dist * np.sin(elev_rad)
    return dict(x=x, y=y, z=z)

def create_bar_mesh_data(x_pos, y_pos, z_val, dx=0.8, dy=0.8):
    """Cria os vértices e faces para uma barra 3D."""
    x_verts = [x_pos - dx/2, x_pos + dx/2, x_pos + dx/2, x_pos - dx/2, 
               x_pos - dx/2, x_pos + dx/2, x_pos + dx/2, x_pos - dx/2]
    y_verts = [y_pos - dy/2, y_pos - dy/2, y_pos + dy/2, y_pos + dy/2,
               y_pos - dy/2, y_pos - dy/2, y_pos + dy/2, y_pos + dy/2]
    z_verts = [0, 0, 0, 0, z_val, z_val, z_val, z_val]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return x_verts, y_verts, z_verts, i, j, k

def create_bar_outline_data(x_pos, y_pos, z_val, dx=0.8, dy=0.8):
    """Cria os segmentos de linha para o contorno de uma barra."""
    x0, x1 = x_pos - dx/2, x_pos + dx/2
    y0, y1 = y_pos - dy/2, y_pos + dy/2
    z0, z1 = 0, z_val
    
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    
    x_lines, y_lines, z_lines = [], [], []
    for p1_idx, p2_idx in edges:
        p1, p2 = verts[p1_idx], verts[p2_idx]
        x_lines.extend([p1[0], p2[0], None])
        y_lines.extend([p1[1], p2[1], None])
        z_lines.extend([p1[2], p2[2], None])
    return x_lines, y_lines, z_lines

# -----------------------------
# Presets de Resolução
# -----------------------------
RESOLUTION_PRESETS = {
    "HD (1280x720)": (1280, 720),
    "Full HD (1920x1080)": (1920, 1080),
    "2K (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160),
    "Square (2000x2000)": (2000, 2000),
    "Portrait (1080x1920)": (1080, 1920),
    "Wide (2560x1080)": (2560, 1080),
    "Custom": None
}

DPI_PRESETS = {
    "Web (72 DPI)": 1,
    "Screen (96 DPI)": 1.33,
    "Print Low (150 DPI)": 2.08,
    "Print Medium (300 DPI)": 4.17,
    "Print High (600 DPI)": 8.33,
}

# -----------------------------
# Função Principal de Renderização
# -----------------------------
def render_plotly_3d_bars(df: pd.DataFrame, **kwargs) -> go.Figure:
    df = df.copy()
    z_col = kwargs['z_col']
    x_col = kwargs['x_col']
    y_col = kwargs['y_col']
    
    df[z_col] = to_numeric_safe(df[z_col])
    df = df.dropna(subset=[x_col, y_col, z_col]).reset_index(drop=True)
    
    fig = go.Figure()
    
    x_cats = pd.unique(df[x_col])
    y_cats = pd.unique(df[y_col])
    x_map = {cat: i for i, cat in enumerate(x_cats)}
    y_map = {cat: i for i, cat in enumerate(y_cats)}
    df['_x_num'] = df[x_col].map(x_map)
    df['_y_num'] = df[y_col].map(y_map)
    
    bar_thickness = kwargs.get('bar_thickness', 0.8)

    # Renderização das barras baseada no modo de cor
    if kwargs['color_mode'] == "Por série (Y)":
        color_map = getattr(px.colors.qualitative, kwargs['series_palette_name'], px.colors.qualitative.Plotly)
        for i, y_val in enumerate(y_cats):
            df_series = df[df[y_col] == y_val]
            color = color_map[i % len(color_map)]
            mesh_x, mesh_y, mesh_z, mesh_i, mesh_j, mesh_k = [], [], [], [], [], []
            base_index = 0
            
            for _, row in df_series.iterrows():
                x_v, y_v, z_v, i_v, j_v, k_v = create_bar_mesh_data(
                    row['_x_num'], row['_y_num'], row[z_col], 
                    dx=bar_thickness, dy=bar_thickness
                )
                mesh_x.extend(x_v)
                mesh_y.extend(y_v)
                mesh_z.extend(z_v)
                mesh_i.extend([idx + base_index for idx in i_v])
                mesh_j.extend([idx + base_index for idx in j_v])
                mesh_k.extend([idx + base_index for idx in k_v])
                base_index += 8
            
            fig.add_trace(go.Mesh3d(
                x=mesh_x, y=mesh_y, z=mesh_z,
                i=mesh_i, j=mesh_j, k=mesh_k,
                color=color, opacity=kwargs['alpha'],
                name=str(y_val)
            ))
    else:
        mesh_x, mesh_y, mesh_z, mesh_i, mesh_j, mesh_k, intensities = [], [], [], [], [], [], []
        base_index = 0
        
        for _, row in df.iterrows():
            x_v, y_v, z_v, i_v, j_v, k_v = create_bar_mesh_data(
                row['_x_num'], row['_y_num'], row[z_col],
                dx=bar_thickness, dy=bar_thickness
            )
            mesh_x.extend(x_v)
            mesh_y.extend(y_v)
            mesh_z.extend(z_v)
            mesh_i.extend([idx + base_index for idx in i_v])
            mesh_j.extend([idx + base_index for idx in j_v])
            mesh_k.extend([idx + base_index for idx in k_v])
            base_index += 8
            
            if kwargs['color_mode'] == "Por altura (colormap)":
                intensities.extend([row[z_col]] * 8)
        
        trace_params = {
            'x': mesh_x, 'y': mesh_y, 'z': mesh_z,
            'i': mesh_i, 'j': mesh_j, 'k': mesh_k,
            'opacity': kwargs['alpha'], 'name': z_col
        }
        
        if kwargs['color_mode'] == "Única":
            trace_params['color'] = kwargs['base_color']
        elif kwargs['color_mode'] == "Por altura (colormap)":
            trace_params['intensity'] = intensities
            trace_params['colorscale'] = kwargs['colormap_name']
            trace_params['colorbar'] = dict(title=z_col)
        
        fig.add_trace(go.Mesh3d(**trace_params))

    # Adicionar contornos se necessário
    if kwargs['show_outline']:
        line_x, line_y, line_z = [], [], []
        for _, row in df.iterrows():
            lx, ly, lz = create_bar_outline_data(
                row['_x_num'], row['_y_num'], row[z_col],
                dx=bar_thickness, dy=bar_thickness
            )
            line_x.extend(lx)
            line_y.extend(ly)
            line_z.extend(lz)
        
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode='lines',
            line=dict(color='black', width=kwargs['outline_width']),
            showlegend=False
        ))

    # Adicionar barras de erro
    if kwargs['err_style'] != "Nenhum":
        err_df = df.copy()
        
        if kwargs['err_style'] == "Simétrico" and kwargs['err_col'] and kwargs['err_col'] in err_df.columns:
            e = to_numeric_safe(err_df[kwargs['err_col']])
            err_df['z_low'] = err_df[z_col] - e
            err_df['z_high'] = err_df[z_col] + e
        elif kwargs['err_style'] == "Assimétrico" and kwargs['err_low_col'] and kwargs['err_high_col']:
            if kwargs['err_low_col'] in err_df.columns and kwargs['err_high_col'] in err_df.columns:
                low = to_numeric_safe(err_df[kwargs['err_low_col']])
                high = to_numeric_safe(err_df[kwargs['err_high_col']])
                err_df['z_low'] = err_df[z_col] - low
                err_df['z_high'] = err_df[z_col] + high
        
        err_df.dropna(subset=['z_low', 'z_high'], inplace=True)
        err_x, err_y, err_z = [], [], []
        for _, row in err_df.iterrows():
            err_x.extend([row['_x_num'], row['_x_num'], None])
            err_y.extend([row['_y_num'], row['_y_num'], None])
            err_z.extend([row['z_low'], row['z_high'], None])
        
        fig.add_trace(go.Scatter3d(
            x=err_x, y=err_y, z=err_z,
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False
        ))

    # Adicionar valores no topo das barras
    if kwargs['show_values']:
        fig.add_trace(go.Scatter3d(
            x=df['_x_num'], y=df['_y_num'],
            z=df[z_col] + kwargs['value_offset'],
            mode='text',
            text=[kwargs['value_fmt'].format(val) for val in df[z_col]],
            textfont=dict(size=kwargs['tick_size'], color='black'),
            showlegend=False
        ))
    
    # Configurar câmera e layout
    camera = degrees_to_plotly_camera(kwargs['elev'], kwargs['azim'])
    
    fig.update_layout(
        title=dict(text=kwargs['title'], font=dict(size=kwargs['title_size'])),
        scene=dict(
            xaxis=dict(
                title=kwargs['xlabel'],
                showgrid=kwargs['show_grid'],
                backgroundcolor=kwargs['pane_color'],
                tickvals=list(x_map.values()),
                ticktext=list(x_map.keys())
            ),
            yaxis=dict(
                title=kwargs['ylabel'],
                showgrid=kwargs['show_grid'],
                backgroundcolor=kwargs['pane_color'],
                tickvals=list(y_map.values()),
                ticktext=list(y_map.keys())
            ),
            zaxis=dict(
                title=kwargs['zlabel'],
                showgrid=kwargs['show_grid'],
                backgroundcolor=kwargs['pane_color'],
                range=[kwargs['zmin'], kwargs['zmax']]
            ),
            camera=dict(eye=camera),
            aspectmode='manual' if kwargs.get('use_aspect', False) else 'auto',
            aspectratio=dict(x=kwargs.get('aspect_x', 1), 
                           y=kwargs.get('aspect_y', 1), 
                           z=kwargs.get('aspect_z', 1)) if kwargs.get('use_aspect', False) else None
        ),
        font=dict(size=kwargs['label_size']),
        plot_bgcolor=kwargs['bg_color'],
        paper_bgcolor=kwargs['bg_color'],
        showlegend=True if kwargs['color_mode'] == "Por série (Y)" else False,
        margin=dict(l=0, r=0, t=50, b=0),
        height=kwargs.get('chart_height', 800)
    )
    
    return fig

# -----------------------------
# Função de Exportação Avançada
# -----------------------------
def export_high_quality_image(fig, format, width, height, scale):
    """Exporta figura em alta qualidade com tratamento de erros."""
    try:
        if format.lower() == 'svg':
            img_bytes = fig.to_image(format='svg', width=width, height=height)
        else:
            img_bytes = fig.to_image(
                format=format.lower(),
                width=width,
                height=height,
                scale=scale
            )
        return img_bytes, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Interface Principal
# -----------------------------
def main():
    # Sidebar com controles
    st.sidebar.title("⚙️ Controles 3D")
    
    params = {}
    
    # ===== SEÇÃO: DADOS =====
    with st.sidebar.expander("📊 Dados", expanded=True):
        up = st.file_uploader("CSV ou Excel", type=["csv", "xlsx", "xls"])
        col1, col2 = st.columns(2)
        with col1:
            sep = st.text_input("Separador", value=",")
        with col2:
            use_editor = st.checkbox("Editar", value=False)
    
    # Carregar dados
    df = None
    if up:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up, sep=sep)
            else:
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Erro: {e}")
    
    df = ensure_min_columns(df)
    cols = list(df.columns)
    
    # ===== SEÇÃO: MAPEAMENTO =====
    with st.sidebar.expander("🔗 Colunas", expanded=True):
        params['x_col'] = st.selectbox("X", cols, 0)
        params['y_col'] = st.selectbox("Y", cols, min(1, len(cols)-1))
        params['z_col'] = st.selectbox("Z", cols, min(2, len(cols)-1))
        
        params['err_style'] = st.radio("Erro", ["Nenhum", "Simétrico", "Assimétrico"])
        params['err_col'] = params['err_low_col'] = params['err_high_col'] = None
        
        if params['err_style'] == "Simétrico":
            params['err_col'] = st.selectbox("Erro ±", ["—"] + cols)
            if params['err_col'] == "—":
                params['err_style'] = "Nenhum"
        elif params['err_style'] == "Assimétrico":
            params['err_low_col'] = st.selectbox("Erro -", ["—"] + cols)
            params['err_high_col'] = st.selectbox("Erro +", ["—"] + cols)
            if "—" in [params['err_low_col'], params['err_high_col']]:
                params['err_style'] = "Nenhum"
    
    # ===== SEÇÃO: APARÊNCIA =====
    with st.sidebar.expander("🎨 Cores"):
        params['alpha'] = st.slider("Opacidade", 0.1, 1.0, 1.0, 0.05)
        params['color_mode'] = st.selectbox("Modo", ["Única", "Por série (Y)", "Por altura (colormap)"])
        
        params['base_color'] = "#3182bd"
        params['colormap_name'] = "Viridis"
        params['series_palette_name'] = "Plotly"
        
        if params['color_mode'] == "Única":
            params['base_color'] = st.color_picker("Cor", "#3182bd")
        elif params['color_mode'] == "Por altura (colormap)":
            params['colormap_name'] = st.selectbox("Mapa", ["Viridis", "Plasma", "Blues", "Reds"])
        else:
            params['series_palette_name'] = st.selectbox("Paleta", ["Plotly", "T10", "G10", "D3"])
    
    # ===== SEÇÃO: BARRAS =====
    with st.sidebar.expander("📊 Barras"):
        params['bar_thickness'] = st.slider("Espessura", 0.1, 1.0, 0.8, 0.05)
        params['show_outline'] = st.checkbox("Contorno")
        params['outline_width'] = 2
        if params['show_outline']:
            params['outline_width'] = st.slider("Largura", 1, 10, 2)
        
        params['show_values'] = st.checkbox("Valores", True)
        if params['show_values']:
            params['value_fmt'] = st.text_input("Formato", "{:.2f}")
            params['value_offset'] = st.number_input("Offset", 2.0, step=0.5)
    
    # ===== SEÇÃO: CÂMERA =====
    with st.sidebar.expander("📐 Câmera"):
        col1, col2 = st.columns(2)
        with col1:
            params['elev'] = st.slider("Elev", -90, 90, 20)
        with col2:
            params['azim'] = st.slider("Azim", -180, 180, -45)
        
        params['use_aspect'] = st.checkbox("Proporção manual")
        if params['use_aspect']:
            c1, c2, c3 = st.columns(3)
            with c1:
                params['aspect_x'] = st.number_input("X", 1.0, 0.1, step=0.1)
            with c2:
                params['aspect_y'] = st.number_input("Y", 1.0, 0.1, step=0.1)
            with c3:
                params['aspect_z'] = st.number_input("Z", 1.0, 0.1, step=0.1)
        
        use_zmin = st.checkbox("Fixar Z mín")
        params['zmin'] = st.number_input("Z mín", 0.0, disabled=not use_zmin) if use_zmin else None
        use_zmax = st.checkbox("Fixar Z máx")
        params['zmax'] = st.number_input("Z máx", 100.0, disabled=not use_zmax) if use_zmax else None
    
    # ===== SEÇÃO: RÓTULOS =====
    with st.sidebar.expander("📝 Texto"):
        params['title'] = st.text_input("Título", "Gráfico 3D")
        params['xlabel'] = st.text_input("Eixo X", "X")
        params['ylabel'] = st.text_input("Eixo Y", "Y")
        params['zlabel'] = st.text_input("Eixo Z", "Z")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            params['title_size'] = st.number_input("Tít", 16, 8, 40)
        with c2:
            params['label_size'] = st.number_input("Eix", 12, 6, 30)
        with c3:
            params['tick_size'] = st.number_input("Val", 10, 6, 30)
    
    # ===== SEÇÃO: LAYOUT =====
    with st.sidebar.expander("🎯 Visual"):
        params['show_grid'] = st.checkbox("Grade", True)
        params['bg_color'] = st.color_picker("Fundo", "#ffffff")
        params['pane_color'] = st.color_picker("Plano", "#f0f0f0")
        chart_height = st.slider("Altura (px)", 400, 1500, 800, 50)
        params['chart_height'] = chart_height
    
    # ===== SEÇÃO: EXPORTAÇÃO =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Export HD")
    
    # Formato
    export_format = st.sidebar.selectbox("Formato", ["PNG", "JPEG", "SVG"])
    
    # Resolução
    resolution_preset = st.sidebar.selectbox(
        "Resolução",
        list(RESOLUTION_PRESETS.keys()),
        index=1
    )
    
    if resolution_preset == "Custom":
        c1, c2 = st.sidebar.columns(2)
        with c1:
            export_width = st.number_input("L (px)", min_value=100, max_value=10000, value=1920)
        with c2:
            export_height = st.number_input("A (px)", min_value=100, max_value=10000, value=1080)
    else:
        export_width, export_height = RESOLUTION_PRESETS[resolution_preset]
        st.sidebar.info(f"📐 {export_width}x{export_height}px")
    
    # Qualidade
    dpi_preset = st.sidebar.selectbox("Qualidade", list(DPI_PRESETS.keys()), 3)
    export_scale = DPI_PRESETS[dpi_preset]
    
    # ===== ÁREA PRINCIPAL =====
    # Container principal com altura customizada
    main_container = st.container()
    
    with main_container:
        # Título
        col1, col2 = st.columns([4, 1])
        with col1:
            st.title("📊 Visualização 3D Interativa")
        with col2:
            if st.button("🔄 Reset"):
                st.rerun()
        
        # Tabs
        tab1, tab2 = st.tabs(["📈 Gráfico", "📊 Dados"])
        
        with tab2:
            if use_editor:
                st.write("### Editor de Dados")
                df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=400)
            else:
                st.write("### Dados Carregados")
                st.dataframe(df, use_container_width=True, height=400)
            
            # Estatísticas
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Linhas", len(df))
            with c2:
                st.metric("Colunas", len(df.columns))
            with c3:
                if params['z_col'] in df.columns:
                    z_vals = to_numeric_safe(df[params['z_col']]).dropna()
                    if len(z_vals) > 0:
                        st.metric("Z Médio", f"{z_vals.mean():.2f}")
            with c4:
                if params['z_col'] in df.columns:
                    z_vals = to_numeric_safe(df[params['z_col']]).dropna()
                    if len(z_vals) > 0:
                        st.metric("Z Máx", f"{z_vals.max():.2f}")
        
        with tab1:
            # Renderizar gráfico
            fig = render_plotly_3d_bars(df=df, **params)
            
            # Mostrar gráfico com altura customizada
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': export_format.lower(),
                        'filename': 'grafico_3d',
                        'height': export_height,
                        'width': export_width,
                        'scale': export_scale
                    }
                }
            )
            
            # Área de download
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write("### 💾 Download HD")
                st.write(f"**Formato:** {export_format}")
                st.write(f"**Resolução:** {export_width}x{export_height}px")
                st.write(f"**DPI Efetivo:** ~{int(72 * export_scale)}")
            
            with col2:
                st.write("### ⚡ Exportar")
                if st.button("🎯 Gerar Imagem HD", type="primary"):
                    with st.spinner("Gerando imagem em alta resolução..."):
                        img_bytes, error = export_high_quality_image(
                            fig, export_format, export_width, export_height, export_scale
                        )
                        
                        if img_bytes:
                            st.success("✅ Imagem gerada com sucesso!")
                            st.download_button(
                                label=f"⬇️ Baixar {export_format} ({export_width}x{export_height})",
                                data=img_bytes,
                                file_name=f"grafico_3d_{export_width}x{export_height}.{export_format.lower()}",
                                mime=f"image/{export_format.lower()}",
                                type="primary"
                            )
                        else:
                            st.error(f"❌ Erro: {error}")
                            st.info("💡 Instale 'kaleido' no requirements.txt")
            
            with col3:
                st.write("### 📋 Info")
                if export_format in ["PNG", "JPEG"]:
                    size_mb = (export_width * export_height * export_scale * 4) / (1024 * 1024)
                    st.write(f"**Tamanho:** ~{size_mb:.1f}MB")
                else:
                    st.write("**Tipo:** Vetorial")

if __name__ == "__main__":
    main()



