# -*- coding: utf-8 -*-
"""
Streamlit — 3D Bar Chart (Advanced) with Plotly
Author: Gemini
Date: 2025-09-13 (Corrected and Updated with High-Res Export)

Recursos principais:
- Gráfico 3D interativo com Plotly (rotação, zoom, pan)
- Upload de CSV/Excel e editor de dados integrado
- Mapeamento de colunas (X, Y, Z, erros)
- Barras com coloração fixa, por série (Y) ou por altura (colormap)
- Rótulos de valores no topo das barras
- Erros simétricos ou assimétricos renderizados como linhas 3D
- Controles de espessura da barra e contornos
- Exportação de imagem em alta qualidade (PNG, JPEG, SVG) com escala de resolução
- Controle de câmera inicial (elevação/azimute), tema, grade, fundo, limites de eixos
- Exportação nativa do Plotly (PNG, SVG, etc.)
"""
import io
from typing import Optional, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# (O restante das funções utilitárias como sample_dataframe, ensure_min_columns, etc., permanecem as mesmas)
# ... [código das funções anteriores sem alteração] ...
def sample_dataframe(n_x=4, n_y=4, seed=42) -> pd.DataFrame:
    """Cria dados de exemplo no formato LONGO: colunas X (cat), Y (cat), Z (valor), Err (simétrico)."""
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
    """Garante ao menos 3 colunas (X,Y,Z). Se não houver dados, usa exemplo."""
    if df is None or df.empty or df.shape[1] < 3:
        return sample_dataframe()
    return df

def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def degrees_to_plotly_camera(elev, azim, dist=1.75):
    """Converte elevação/azimute (graus) para o vetor 'eye' do Plotly."""
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    x = dist * np.cos(elev_rad) * np.cos(azim_rad)
    y = dist * np.cos(elev_rad) * np.sin(azim_rad)
    z = dist * np.sin(elev_rad)
    return dict(x=x, y=y, z=z)

def create_bar_mesh_data(x_pos, y_pos, z_val, dx=0.8, dy=0.8) -> Tuple[List, List, List, List, List, List]:
    """Cria os vértices e faces para um único paralelepípedo (barra) no Mesh3d."""
    x_verts = [x_pos - dx/2, x_pos + dx/2, x_pos + dx/2, x_pos - dx/2, x_pos - dx/2, x_pos + dx/2, x_pos + dx/2, x_pos - dx/2]
    y_verts = [y_pos - dy/2, y_pos - dy/2, y_pos + dy/2, y_pos + dy/2, y_pos - dy/2, y_pos - dy/2, y_pos + dy/2, y_pos + dy/2]
    z_verts = [0, 0, 0, 0, z_val, z_val, z_val, z_val]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]; j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]; k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return x_verts, y_verts, z_verts, i, j, k

def create_bar_outline_data(x_pos, y_pos, z_val, dx=0.8, dy=0.8):
    """Cria os segmentos de linha para o contorno de uma barra."""
    x0, x1 = x_pos - dx/2, x_pos + dx/2; y0, y1 = y_pos - dy/2, y_pos + dy/2; z0, z1 = 0, z_val
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    x_lines, y_lines, z_lines = [], [], []
    for p1_idx, p2_idx in edges:
        p1, p2 = verts[p1_idx], verts[p2_idx]
        x_lines.extend([p1[0], p2[0], None]); y_lines.extend([p1[1], p2[1], None]); z_lines.extend([p1[2], p2[2], None])
    return x_lines, y_lines, z_lines

def render_plotly_3d_bars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    # ... (todos os outros parâmetros permanecem os mesmos)
    **kwargs 
) -> go.Figure:
    
    # Esta função continua a mesma, mas usamos **kwargs para manter a chamada limpa.
    # O código interno não precisa de alterações.
    df = df.copy()
    df[z_col] = to_numeric_safe(df[z_col])
    df = df.dropna(subset=[x_col, y_col, z_col]).reset_index(drop=True)
    fig = go.Figure()
    x_cats = pd.unique(df[x_col]); y_cats = pd.unique(df[y_col])
    x_map = {cat: i for i, cat in enumerate(x_cats)}; y_map = {cat: i for i, cat in enumerate(y_cats)}
    df['_x_num'] = df[x_col].map(x_map); df['_y_num'] = df[y_col].map(y_map)
    bar_thickness = kwargs.get('bar_thickness', 0.8)

    if kwargs['color_mode'] == "Por série (Y)":
        color_map = getattr(px.colors.qualitative, kwargs['series_palette_name'], px.colors.qualitative.Plotly)
        for i, y_val in enumerate(y_cats):
            df_series = df[df[y_col] == y_val]
            color = color_map[i % len(color_map)]
            mesh_x, mesh_y, mesh_z, mesh_i, mesh_j, mesh_k = [], [], [], [], [], []
            base_index = 0
            for _, row in df_series.iterrows():
                x_v, y_v, z_v, i_v, j_v, k_v = create_bar_mesh_data(row['_x_num'], row['_y_num'], row[z_col], dx=bar_thickness, dy=bar_thickness)
                mesh_x.extend(x_v); mesh_y.extend(y_v); mesh_z.extend(z_v)
                mesh_i.extend([idx + base_index for idx in i_v]); mesh_j.extend([idx + base_index for idx in j_v]); mesh_k.extend([idx + base_index for idx in k_v])
                base_index += 8
            fig.add_trace(go.Mesh3d(x=mesh_x, y=mesh_y, z=mesh_z, i=mesh_i, j=mesh_j, k=mesh_k, color=color, opacity=kwargs['alpha'], name=str(y_val)))
    else:
        mesh_x, mesh_y, mesh_z, mesh_i, mesh_j, mesh_k, intensities = [], [], [], [], [], [], []
        base_index = 0
        for _, row in df.iterrows():
            x_v, y_v, z_v, i_v, j_v, k_v = create_bar_mesh_data(row['_x_num'], row['_y_num'], row[z_col], dx=bar_thickness, dy=bar_thickness)
            mesh_x.extend(x_v); mesh_y.extend(y_v); mesh_z.extend(z_v)
            mesh_i.extend([idx + base_index for idx in i_v]); mesh_j.extend([idx + base_index for idx in j_v]); mesh_k.extend([idx + base_index for idx in k_v])
            base_index += 8
            if kwargs['color_mode'] == "Por altura (colormap)":
                intensities.extend([row[z_col]] * 8)
        trace_params = {'x': mesh_x, 'y': mesh_y, 'z': mesh_z, 'i': mesh_i, 'j': mesh_j, 'k': mesh_k, 'opacity': kwargs['alpha'], 'name': z_col}
        if kwargs['color_mode'] == "Única":
            trace_params['color'] = kwargs['base_color']
        elif kwargs['color_mode'] == "Por altura (colormap)":
            trace_params['intensity'] = intensities; trace_params['colorscale'] = kwargs['colormap_name']; trace_params['colorbar'] = dict(title=z_col)
        fig.add_trace(go.Mesh3d(**trace_params))

    if kwargs['show_outline']:
        line_x, line_y, line_z = [], [], []
        for _, row in df.iterrows():
            lx, ly, lz = create_bar_outline_data(row['_x_num'], row['_y_num'], row[z_col], dx=bar_thickness, dy=bar_thickness)
            line_x.extend(lx); line_y.extend(ly); line_z.extend(lz)
        fig.add_trace(go.Scatter3d(x=line_x, y=line_y, z=line_z, mode='lines', line=dict(color='black', width=kwargs['outline_width']), showlegend=False))

    if kwargs['err_style'] != "Nenhum":
        err_df = df.copy()
        if kwargs['err_style'] == "Simétrico" and kwargs['err_col'] and kwargs['err_col'] in err_df.columns:
            e = to_numeric_safe(err_df[kwargs['err_col']]); err_df['z_low'] = err_df[z_col] - e; err_df['z_high'] = err_df[z_col] + e
        elif kwargs['err_style'] == "Assimétrico" and kwargs['err_low_col'] and kwargs['err_high_col'] and kwargs['err_low_col'] in err_df.columns and kwargs['err_high_col'] in err_df.columns:
            low = to_numeric_safe(err_df[kwargs['err_low_col']]); high = to_numeric_safe(err_df[kwargs['err_high_col']]); err_df['z_low'] = err_df[z_col] - low; err_df['z_high'] = err_df[z_col] + high
        err_df.dropna(subset=['z_low', 'z_high'], inplace=True)
        err_x, err_y, err_z = [], [], []
        for _, row in err_df.iterrows():
            err_x.extend([row['_x_num'], row['_x_num'], None]); err_y.extend([row['_y_num'], row['_y_num'], None]); err_z.extend([row['z_low'], row['z_high'], None])
        fig.add_trace(go.Scatter3d(x=err_x, y=err_y, z=err_z, mode='lines', line=dict(color='black', width=3), showlegend=False))

    if kwargs['show_values']:
        fig.add_trace(go.Scatter3d(x=df['_x_num'], y=df['_y_num'], z=df[z_col] + kwargs['value_offset'], mode='text', text=[kwargs['value_fmt'].format(val) for val in df[z_col]], textfont=dict(size=kwargs['tick_size'], color='black'), showlegend=False))
        
    camera = degrees_to_plotly_camera(kwargs['elev'], kwargs['azim'])
    fig.update_layout(
        title=dict(text=kwargs['title'], font=dict(size=kwargs['title_size'])),
        scene=dict(
            xaxis=dict(title=kwargs['xlabel'], showgrid=kwargs['show_grid'], backgroundcolor=kwargs['pane_color'], tickvals=list(x_map.values()), ticktext=list(x_map.keys())),
            yaxis=dict(title=kwargs['ylabel'], showgrid=kwargs['show_grid'], backgroundcolor=kwargs['pane_color'], tickvals=list(y_map.values()), ticktext=list(y_map.keys())),
            zaxis=dict(title=kwargs['zlabel'], showgrid=kwargs['show_grid'], backgroundcolor=kwargs['pane_color'], range=[kwargs['zmin'], kwargs['zmax']]),
            camera=dict(eye=camera)
        ),
        font=dict(size=kwargs['label_size']), plot_bgcolor=kwargs['bg_color'], paper_bgcolor=kwargs['bg_color'],
        showlegend=True if kwargs['color_mode'] == "Por série (Y)" else False
    )
    return fig

def main():
    st.set_page_config(page_title="Barras 3D Interativo", layout="wide")
    st.sidebar.title("⚙️ Controles")
    
    # --- Agrupar todos os parâmetros em um dicionário para facilitar a passagem ---
    params = {}

    st.sidebar.subheader("Dados")
    up = st.sidebar.file_uploader("CSV ou Excel", type=["csv", "xlsx", "xls"])
    sep = st.sidebar.text_input("Separador (CSV)", value=",", help="Ignorado para Excel.", key="sep")
    use_editor = st.sidebar.checkbox("Editar dados após carregar", value=False)

    df = None
    if up:
        try:
            df = pd.read_csv(up, sep=sep) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error(f"Falha ao ler arquivo: {e}")
    df = ensure_min_columns(df)
    
    cols = list(df.columns)
    st.sidebar.subheader("Mapeamento de Colunas")
    params['x_col'] = st.sidebar.selectbox("Coluna X", options=cols, index=0)
    params['y_col'] = st.sidebar.selectbox("Coluna Y", options=cols, index=1 if len(cols) > 1 else 0)
    params['z_col'] = st.sidebar.selectbox("Coluna Z (altura)", options=cols, index=2 if len(cols) > 2 else 0)

    st.sidebar.subheader("Erros (opcional)")
    params['err_style'] = st.sidebar.radio("Tipo de erro", ["Nenhum", "Simétrico", "Assimétrico"], horizontal=True)
    params['err_col'], params['err_low_col'], params['err_high_col'] = None, None, None
    if params['err_style'] == "Simétrico":
        params['err_col'] = st.sidebar.selectbox("Coluna de erro (±)", ["—"] + cols)
        if params['err_col'] == "—": params['err_style'] = "Nenhum"
    elif params['err_style'] == "Assimétrico":
        params['err_low_col'] = st.sidebar.selectbox("Coluna de erro -", ["—"] + cols)
        params['err_high_col'] = st.sidebar.selectbox("Coluna de erro +", ["—"] + cols)
        if params['err_low_col'] == "—" or params['err_high_col'] == "—": params['err_style'] = "Nenhum"

    st.sidebar.subheader("Cores e Aparência")
    params['alpha'] = st.sidebar.slider("Transparência (alpha)", 0.1, 1.0, 1.0, 0.05)
    params['color_mode'] = st.sidebar.selectbox("Modo de cor", ["Única", "Por série (Y)", "Por altura (colormap)"])
    params['base_color'], params['colormap_name'], params['series_palette_name'] = "#3182bd", "Viridis", "Plotly"
    if params['color_mode'] == "Única":
        params['base_color'] = st.sidebar.color_picker("Cor base", value="#3182bd")
    elif params['color_mode'] == "Por altura (colormap)":
        params['colormap_name'] = st.sidebar.selectbox("Colormap", ["Viridis", "Cividis", "Plasma", "Blues", "Reds", "Greens"])
    elif params['color_mode'] == "Por série (Y)":
        params['series_palette_name'] = st.sidebar.selectbox("Paleta de séries", ["Plotly", "T10", "G10", "D3", "Pastel"])
        
    params['show_values'] = st.sidebar.checkbox("Mostrar valores no topo", value=True)
    params['value_fmt'] = st.sidebar.text_input("Formato do valor", value="{:.2f}")
    params['value_offset'] = st.sidebar.number_input("Offset em Z dos valores", value=2.0, step=0.5)

    st.sidebar.subheader("Formato das Barras")
    params['bar_thickness'] = st.sidebar.slider("Espessura da barra", 0.1, 1.0, 0.8, 0.05)
    params['show_outline'] = st.sidebar.checkbox("Mostrar contorno das barras", value=False)
    params['outline_width'] = 2
    if params['show_outline']:
        params['outline_width'] = st.sidebar.slider("Espessura do contorno", 1, 10, 2, 1)

    st.sidebar.subheader("Rótulos, Limites e Câmera")
    params['title'] = st.sidebar.text_input("Título", value="Gráfico de Barras 3D Interativo")
    params['xlabel'], params['ylabel'], params['zlabel'] = st.sidebar.text_input("Rótulo X", "X"), st.sidebar.text_input("Rótulo Y", "Y"), st.sidebar.text_input("Rótulo Z", "Z")
    params['title_size'] = st.sidebar.slider("Tamanho Título", 8, 40, 16); params['label_size'] = st.sidebar.slider("Tamanho Rótulos", 6, 30, 12); params['tick_size'] = st.sidebar.slider("Tamanho Ticks/Valores", 6, 30, 10)
    params['elev'] = st.sidebar.slider("Elevação da câmera", -90, 90, 20); params['azim'] = st.sidebar.slider("Azimute da câmera", -180, 180, -45)
    
    st.sidebar.subheader("Layout e Eixos")
    chart_height = st.sidebar.number_input("Altura do gráfico (pixels)", min_value=400, max_value=2000, value=700, step=50)
    params['show_grid'] = st.sidebar.checkbox("Mostrar grade", value=True)
    use_zmin = st.sidebar.checkbox("Fixar Z mín"); params['zmin'] = st.sidebar.number_input("Z mínimo", value=0.0, disabled=not use_zmin) if use_zmin else None
    use_zmax = st.sidebar.checkbox("Fixar Z máx"); params['zmax'] = st.sidebar.number_input("Z máximo", value=100.0, disabled=not use_zmax) if use_zmax else None
    params['bg_color'] = st.sidebar.color_picker("Cor de fundo", "#ffffff"); params['pane_color'] = st.sidebar.color_picker("Cor do plano 3D", "#f0f0f0")

    # --- NOVO: Seção de Exportação ---
    st.sidebar.subheader("Exportação de Alta Qualidade")
    export_format = st.sidebar.selectbox("Formato de Imagem", ["PNG", "JPEG", "SVG"])
    export_scale = st.sidebar.slider("Fator de Escala (Qualidade)", 1, 10, 6, help="Aumente para maior resolução. Um fator de 6x é aproximadamente 600 DPI.")

    # --- Área Principal da Página ---
    st.title("📊 Barras 3D — Interativo (Plotly)")
    if use_editor:
        st.write("Editor de Dados:"); df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    else:
        st.write("Prévia do DataFrame:"); st.dataframe(df, use_container_width=True)
    
    fig = render_plotly_3d_bars(df=df, **params)
    st.plotly_chart(fig, use_container_width=True, height=int(chart_height), config={'displaylogo': False})
    
    # --- NOVO: Botão de Download ---
    st.write("---")
    st.subheader("Download do Gráfico")
    try:
        # Gera a imagem em memória
        img_bytes = fig.to_image(format=export_format.lower(), scale=export_scale)
        
        # Cria o botão de download
        st.download_button(
            label=f"Baixar como {export_format}",
            data=img_bytes,
            file_name=f"grafico_3d.{export_format.lower()}",
            mime=f"image/{export_format.lower()}"
        )
    except Exception as e:
        st.error(f"Falha ao gerar imagem para exportação: {e}")
        st.info("Certifique-se de ter a biblioteca 'kaleido' instalada (`pip install kaleido`)")


if __name__ == "__main__":
    main()




