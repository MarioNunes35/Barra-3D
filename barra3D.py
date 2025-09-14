# barra3D.py — Streamlit + Plotly 3D Bars (client-side export, advanced customization, multi-file input)
# Autor: ChatGPT (GPT-5 Thinking)
# Execução: streamlit run barra3D.py

import io
import base64
import uuid
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Barras 3D (custom avançado / sem Kaleido)", layout="wide", initial_sidebar_state="expanded")

# ======================================
# Utilitário: botão de download client-side (sem Kaleido)
# ======================================
def plotly_download_button(fig, filename="grafico.png", fmt="png", width=1600, height=900, scale=2):
    """
    Baixa a figura Plotly no navegador via Plotly.downloadImage (sem Kaleido).
    Formatos: 'png', 'jpeg', 'webp', 'svg'. width/height em px; scale (1–4).
    """
    fig_json = fig.to_json()
    fig_b64 = base64.b64encode(fig_json.encode("utf-8")).decode("ascii")
    uid = "pldl_" + uuid.uuid4().hex

    html = f"""
    <div id="{uid}" style="position:absolute; left:-10000px; top:0; width:{width}px; height:{height}px;"></div>
    <button id="{uid}_btn" style="padding:0.5rem 0.75rem; border-radius:8px; border:1px solid #ccc; cursor:pointer;">
      ⬇️ Baixar {filename}
    </button>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
    (function(){{
        const uid = "{uid}";
        const container = document.getElementById(uid);
        const btn = document.getElementById(uid + "_btn");
        const fig = JSON.parse(atob("{fig_b64}"));
        Plotly.newPlot(container, fig.data, fig.layout, {{displayModeBar:false, responsive:true}}).then(() => {{
            btn.onclick = function(){{
                Plotly.downloadImage(container, {{
                    format: "{fmt}",
                    filename: "{filename.rsplit('.', 1)[0]}",
                    width: {width},
                    height: {height},
                    scale: {scale}
                }});
            }};
        }});
    }})();
    </script>
    """
    components.html(html, height=60)

# ======================================
# Geometria das barras (Mesh3d)
# ======================================
def cuboid_vertices(x0, y0, z0, dx, dy, dz):
    return np.array([
        [x0,     y0,     z0    ],
        [x0+dx,  y0,     z0    ],
        [x0+dx,  y0+dy,  z0    ],
        [x0,     y0+dy,  z0    ],
        [x0,     y0,     z0+dz ],
        [x0+dx,  y0,     z0+dz ],
        [x0+dx,  y0+dy,  z0+dz ],
        [x0,     y0+dy,  z0+dz ],
    ], dtype=float)

CUBOID_FACES = np.array([
    [0,1,2],[0,2,3],
    [4,5,6],[4,6,7],
    [0,1,5],[0,5,4],
    [1,2,6],[1,6,5],
    [2,3,7],[2,7,6],
    [3,0,4],[3,4,7],
], dtype=int)

def make_bars_mesh(Z, sx=1.0, sy=1.0, bx=0.7, by=0.7, base_z=0.0):
    rows, cols = Z.shape
    half_dx = (bx * sx) / 2.0
    half_dy = (by * sy) / 2.0

    vertices = []
    intensities = []
    faces_i = []
    faces_j = []
    faces_k = []
    idx_offset = 0

    centers = []  # para labels
    for r in range(rows):
        for c in range(cols):
            z = float(Z[r, c])
            cx = c * sx
            cy = r * sy
            x0 = cx - half_dx
            y0 = cy - half_dy
            z0 = base_z
            dx = bx * sx
            dy = by * sy
            dz = max(0.0, z - base_z)

            V = cuboid_vertices(x0, y0, z0, dx, dy, dz)
            vertices.append(V)
            intensities.append(np.full(8, z, dtype=float))

            faces = CUBOID_FACES + idx_offset
            faces_i.extend(faces[:,0])
            faces_j.extend(faces[:,1])
            faces_k.extend(faces[:,2])
            idx_offset += 8

            centers.append((cx, cy, base_z + dz))

    if not vertices:
        return [], [], [], [], [], [], [], []

    V_all = np.vstack(vertices)
    intensity_all = np.concatenate(intensities)
    x, y, z = V_all[:,0], V_all[:,1], V_all[:,2]
    return x, y, z, np.array(faces_i), np.array(faces_j), np.array(faces_k), intensity_all, np.array(centers)

def make_edges_traces(Z, sx=1.0, sy=1.0, bx=0.7, by=0.7, base_z=0.0, line_width=1.0, line_color="black"):
    rows, cols = Z.shape
    half_dx = (bx * sx) / 2.0
    half_dy = (by * sy) / 2.0

    x_lines, y_lines, z_lines = [], [], []
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    for r in range(rows):
        for c in range(cols):
            z = float(Z[r, c])
            cx, cy = c * sx, r * sy
            x0, y0, z0 = cx - half_dx, cy - half_dy, base_z
            dx, dy, dz = bx * sx, by * sy, max(0.0, z - base_z)
            verts = cuboid_vertices(x0, y0, z0, dx, dy, dz)

            for a, b in edges:
                p1, p2 = verts[a], verts[b]
                x_lines += [p1[0], p2[0], None]
                y_lines += [p1[1], p2[1], None]
                z_lines += [p1[2], p2[2], None]

    if not x_lines:
        return None

    return go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode="lines",
        line=dict(width=line_width, color=line_color),
        hoverinfo="skip",
        name="Bordas"
    )

# ======================================
# Helpers de câmera e parsing de arquivos
# ======================================
def get_camera_eye(mode, custom=(1.5,1.8,1.2)):
    if mode == "frente":
        return dict(x=0.1, y=2.0, z=1.5)
    if mode == "lado":
        return dict(x=2.0, y=0.1, z=1.5)
    if mode == "topo":
        return dict(x=0.1, y=0.1, z=3.0)
    if mode == "custom":
        cx, cy, cz = custom
        return dict(x=float(cx), y=float(cy), z=float(cz))
    return dict(x=1.5, y=1.8, z=1.2)  # isométrica

def read_table_any(file, mode_hint="auto", delimiter="auto", decimal=".", sheet=None) -> Tuple[Optional[pd.DataFrame], str]:
    name = file.name.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            if name.endswith(".xlsx"):
                df = pd.read_excel(file, sheet_name=sheet or 0, engine="openpyxl")
            else:
                df = pd.read_excel(file, sheet_name=sheet or 0)  # xlrd para .xls (suportado)
            return df, ""
        else:
            # texto: tentar detectar separador
            if delimiter == "auto":
                # Tentativas comuns
                for sep in [",", ";", "\t", "|"]:
                    try:
                        df = pd.read_csv(file, sep=sep, decimal=decimal)
                        if df.shape[1] >= 2:
                            return df, ""
                    except Exception:
                        continue
                # Último recurso: whitespace
                file.seek(0)
                df = pd.read_csv(file, sep=r"\s+", engine="python", decimal=decimal)
                return df, ""
            else:
                df = pd.read_csv(file, sep=delimiter, decimal=decimal)
                return df, ""
    except Exception as e:
        return None, f"Erro ao ler arquivo: {e}"

def grid_from_triplet(df, col_x, col_y, col_z) -> np.ndarray:
    d = df[[col_x, col_y, col_z]].rename(columns={col_x:"x", col_y:"y", col_z:"z"})
    xs = np.sort(d["x"].unique())
    ys = np.sort(d["y"].unique())
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, fill_value=0.0, dtype=float)
    for _, row in d.iterrows():
        xi = np.where(xs == row["x"])[0][0]
        yi = np.where(ys == row["y"])[0][0]
        Z[yi, xi] = float(row["z"])
    return Z

def grid_from_indices(df, col_i, col_j, col_val) -> np.ndarray:
    d = df[[col_i, col_j, col_val]].rename(columns={col_i:"i", col_j:"j", col_val:"value"})
    r = int(d["i"].max())+1
    c = int(d["j"].max())+1
    Z = np.zeros((r, c), dtype=float)
    for _, row in d.iterrows():
        Z[int(row["i"]), int(row["j"])] = float(row["value"])
    return Z

def grid_from_matrix(df, first_row_as_x=True, first_col_as_y=True) -> np.ndarray:
    temp = df.copy()
    if first_row_as_x:
        temp.columns = temp.iloc[0].values
        temp = temp.iloc[1:]
    if first_col_as_y:
        temp.index = temp.iloc[:,0].values
        temp = temp.iloc[:,1:]
    # agora o restante deve ser valores numéricos
    temp = temp.apply(pd.to_numeric, errors="coerce")
    temp = temp.fillna(0.0)
    return temp.values.astype(float)

# ======================================
# Sidebar — Dados & Parsing
# ======================================
st.sidebar.title("🎛️ Controles")

data_source = st.sidebar.radio("Origem dos dados", ["Aleatório", "Arquivo"], index=0, horizontal=True)
uploaded = None
df_preview = None
warn = ""

if data_source == "Arquivo":
    file = st.sidebar.file_uploader("Anexar: CSV, TXT, TSV, XLSX, XLS", type=["csv","txt","tsv","xlsx","xls"], accept_multiple_files=False)
    if file is not None:
        with st.sidebar.expander("⚙️ Leitura do arquivo", expanded=False):
            delimiter = st.selectbox("Separador", ["auto", ",", ";", "\\t", "|"], index=0)
            if delimiter == "\\t":
                delimiter = "\t"
            decimal = st.selectbox("Decimal", [".", ","], index=0)
            sheet = None
            if file.name.lower().endswith((".xlsx",".xls")):
                try:
                    xls = pd.ExcelFile(file)
                    sheet = st.selectbox("Planilha (Excel)", xls.sheet_names, index=0)
                    file.seek(0)
                except Exception as e:
                    st.warning(f"Não foi possível listar planilhas: {e}")
                    sheet = None

        df_preview, warn = read_table_any(file, delimiter=delimiter, decimal=decimal, sheet=sheet)
        if warn:
            st.sidebar.error(warn)
        else:
            st.sidebar.success(f"Arquivo lido: {df_preview.shape[0]} linhas × {df_preview.shape[1]} colunas")
            st.sidebar.caption("Ajuste o mapeamento de colunas em “Formato dos dados”.")

# ======================================
# Sidebar — Formato dos Dados
# ======================================
format_mode = st.sidebar.selectbox("Formato dos dados", ["Triplet (x,y,z)", "Índices (i,j,value)", "Matriz (linhas × colunas)"], index=0, help="Escolha como seu arquivo representa os dados 3D.")

if data_source == "Aleatório":
    rows = st.sidebar.slider("Linhas (rows)", 1, 50, 6)
    cols = st.sidebar.slider("Colunas (cols)", 1, 50, 6)
    z_min = st.sidebar.slider("Altura mínima (rand)", 0.0, 10.0, 1.0, 0.1)
    z_max = st.sidebar.slider("Altura máxima (rand)", 0.0, 50.0, 6.0, 0.1)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
else:
    rows = cols = None
    z_min = z_max = seed = None

# Mapeamento de colunas quando há arquivo
col_x = col_y = col_z = col_i = col_j = col_val = None
mat_first_row = mat_first_col = True

if data_source == "Arquivo" and df_preview is not None:
    with st.sidebar.expander("🧭 Mapeamento de colunas", expanded=True):
        cols_list = list(df_preview.columns)
        if format_mode.startswith("Triplet"):
            col_x = st.selectbox("Coluna X", cols_list, index=min(0, len(cols_list)-1))
            col_y = st.selectbox("Coluna Y", cols_list, index=min(1, len(cols_list)-1))
            col_z = st.selectbox("Coluna Z (valor)", cols_list, index=min(2, len(cols_list)-1))
        elif format_mode.startswith("Índices"):
            col_i = st.selectbox("Coluna i (linha)", cols_list, index=min(0, len(cols_list)-1))
            col_j = st.selectbox("Coluna j (coluna)", cols_list, index=min(1, len(cols_list)-1))
            col_val = st.selectbox("Coluna valor", cols_list, index=min(2, len(cols_list)-1))
        else:
            mat_first_row = st.checkbox("Primeira linha é cabeçalho (X)", True)
            mat_first_col = st.checkbox("Primeira coluna é rótulo de linhas (Y)", True)

# ======================================
# Sidebar — Geometria / Estilo / Cores
# ======================================
st.sidebar.markdown("---")
st.sidebar.subheader("🧱 Geometria das Barras")
spacing_x = st.sidebar.slider("Espaçamento X (sx)", 0.1, 5.0, 1.0, 0.1)
spacing_y = st.sidebar.slider("Espaçamento Y (sy)", 0.1, 5.0, 1.0, 0.1)
bar_size_x = st.sidebar.slider("Largura barra X (bx)", 0.05, 1.0, 0.7, 0.05)
bar_size_y = st.sidebar.slider("Largura barra Y (by)", 0.05, 1.0, 0.7, 0.05)
opacity = st.sidebar.slider("Opacidade", 0.1, 1.0, 0.95, 0.05)

st.sidebar.subheader("🎨 Cores")
colorscale = st.sidebar.selectbox("Escala de cores", ["Viridis","Plasma","Inferno","Magma","Cividis","Turbo","IceFire","Sunset","Jet"], index=0)
reverse_colors = st.sidebar.checkbox("Inverter escala", False)
show_colorbar = st.sidebar.checkbox("Mostrar barra de cores", True)
cmin_en = st.sidebar.checkbox("Fixar cmin/cmax", False)
cmin_val = cmax_val = None
if cmin_en:
    cmin_val = st.sidebar.number_input("cmin", value=0.0, step=0.1)
    cmax_val = st.sidebar.number_input("cmax", value=10.0, step=0.1)

st.sidebar.subheader("💡 Iluminação")
ambient = st.sidebar.slider("ambient", 0.0, 1.0, 0.5, 0.05)
diffuse = st.sidebar.slider("diffuse", 0.0, 1.0, 0.6, 0.05)
specular = st.sidebar.slider("specular", 0.0, 1.0, 0.2, 0.05)

st.sidebar.subheader("🧵 Bordas")
show_edges = st.sidebar.checkbox("Mostrar bordas", True)
edge_width = st.sidebar.slider("Espessura da borda", 0.5, 6.0, 1.2, 0.1)
edge_color = st.sidebar.color_picker("Cor da borda", "#000000")

# ======================================
# Sidebar — Layout / Câmera / Eixos
# ======================================
st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Layout")
template = st.sidebar.selectbox("Tema (Plotly)", ["plotly_white","plotly","plotly_dark","ggplot2","seaborn","simple_white","none"], index=0)
title_text = st.sidebar.text_input("Título do gráfico", "Gráfico de Coluna 3D")
title_size = st.sidebar.slider("Tamanho do título", 10, 48, 20)
label_size = st.sidebar.slider("Tamanho rótulos (eixos)", 8, 36, 14)
tick_size = st.sidebar.slider("Tamanho ticks", 8, 36, 12)

x_label = st.sidebar.text_input("Eixo X", "X")
y_label = st.sidebar.text_input("Eixo Y", "Y")
z_label = st.sidebar.text_input("Eixo Z", "Z")

scene_bg = st.sidebar.color_picker("Cor de fundo da cena", "#ffffff")
axis_bg_on = st.sidebar.checkbox("Mostrar fundo dos eixos", False)
axis_bg_color = st.sidebar.color_picker("Cor do fundo dos eixos", "#E5ECF6")

grid_on = st.sidebar.checkbox("Mostrar grid", False)
grid_color = st.sidebar.color_picker("Cor da grid", "#CCCCCC")

camera_view = st.sidebar.selectbox("Câmera", ["isométrica","frente","lado","topo","custom"], index=0)
cam_x = st.sidebar.number_input("camera.x", value=1.5, step=0.1)
cam_y = st.sidebar.number_input("camera.y", value=1.8, step=0.1)
cam_z = st.sidebar.number_input("camera.z", value=1.2, step=0.1)

aspect_mode = st.sidebar.selectbox("Aspecto", ["data","cube","manual"], index=0)
ar_x = st.sidebar.number_input("aspect.x", value=1.0, step=0.1)
ar_y = st.sidebar.number_input("aspect.y", value=1.0, step=0.1)
ar_z = st.sidebar.number_input("aspect.z", value=0.7, step=0.1)

manual_range = st.sidebar.checkbox("Fixar faixa dos eixos", False)
x_min = st.sidebar.number_input("x_min", value=0.0)
x_max = st.sidebar.number_input("x_max", value=10.0)
y_min = st.sidebar.number_input("y_min", value=0.0)
y_max = st.sidebar.number_input("y_max", value=10.0)
z_min_fixed = st.sidebar.number_input("z_min (axis)", value=0.0)
z_max_fixed = st.sidebar.number_input("z_max (axis)", value=10.0)

st.sidebar.subheader("📝 Rótulos numéricos")
show_value_labels = st.sidebar.checkbox("Exibir valores no topo", False)
label_decimals = st.sidebar.slider("Casas decimais", 0, 6, 2)
label_font_size = st.sidebar.slider("Tamanho dos rótulos", 8, 30, 12)

# ======================================
# Sidebar — Exportação (sem Kaleido)
# ======================================
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Exportação (no navegador)")
fmt = st.sidebar.selectbox("Formato", ["PNG","JPEG","WEBP","SVG"], index=0)
preset = st.sidebar.selectbox("Resolução", ["1080p (1920x1080)","2K (2560x1440)","4K (3840x2160)","8K (7680x4320)","Custom"], index=2)
presets = {
    "1080p (1920x1080)": (1920, 1080),
    "2K (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160),
    "8K (7680x4320)": (7680, 4320),
}
if preset != "Custom":
    export_w, export_h = presets[preset]
else:
    export_w = st.sidebar.number_input("Largura (px)", 800, 16000, 3840, 10)
    export_h = st.sidebar.number_input("Altura (px)", 600, 16000, 2160, 10)
export_scale = st.sidebar.slider("Escala (1–4)", 1.0, 4.0, 2.0, 0.5)

# ======================================
# Construção do Z
# ======================================
if data_source == "Aleatório":
    rng = np.random.default_rng(seed or 0)
    Z = rng.uniform(z_min, z_max, size=(rows, cols))
else:
    if df_preview is None:
        st.error("Envie um arquivo e defina o mapeamento.")
        st.stop()
    if format_mode.startswith("Triplet"):
        Z = grid_from_triplet(df_preview, col_x, col_y, col_z)
    elif format_mode.startswith("Índices"):
        Z = grid_from_indices(df_preview, col_i, col_j, col_val)
    else:
        Z = grid_from_matrix(df_preview, first_row_as_x=mat_first_row, first_col_as_y=mat_first_col)

# Aviso de carga
r, c = Z.shape
if r*c > 400:
    st.info(f"Muitas barras ({r*c}). O navegador pode ficar pesado.")

# ======================================
# Construção das traces
# ======================================
x, y, z, i, j, k, intensity, centers = make_bars_mesh(
    Z, sx=spacing_x, sy=spacing_y, bx=bar_size_x, by=bar_size_y, base_z=0.0
)

if reverse_colors:
    colorscale_name = colorscale + "_r"
else:
    colorscale_name = colorscale

data_traces = []
if len(x):
    mesh_kwargs = dict(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=intensity,
        colorscale=colorscale_name,
        opacity=opacity,
        flatshading=True,
        showscale=show_colorbar,
        lighting=dict(ambient=ambient, diffuse=diffuse, specular=specular),
        lightposition=dict(x=100, y=200, z=0),
        name="Barras 3D",
        intensitymode="vertex",
    )
    if cmin_en and cmin_val is not None and cmax_val is not None:
        mesh_kwargs["cmin"] = float(cmin_val)
        mesh_kwargs["cmax"] = float(cmax_val)

    mesh = go.Mesh3d(**mesh_kwargs)
    data_traces.append(mesh)

# Bordas
if show_edges:
    edge_trace = make_edges_traces(
        Z, sx=spacing_x, sy=spacing_y, bx=bar_size_x, by=bar_size_y, base_z=0.0, line_width=edge_width, line_color=edge_color
    )
    if edge_trace is not None:
        data_traces.append(edge_trace)

# Labels de valor
if show_value_labels and len(centers):
    cx, cy, cz = centers[:,0], centers[:,1], centers[:,2]
    texts = [f"{v:.{label_decimals}f}" for v in Z.flatten(order="C")]
    label_trace = go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="text",
        text=texts,
        textposition="top center",
        textfont=dict(size=label_font_size, color="black"),
        name="Valores"
    )
    data_traces.append(label_trace)

# ======================================
# Layout da figura
# ======================================
xa = dict(title=dict(text=x_label, font=dict(size=label_size)),
          tickfont=dict(size=tick_size),
          showgrid=grid_on, gridcolor=grid_color,
          showbackground=axis_bg_on, backgroundcolor=axis_bg_color)
ya = dict(title=dict(text=y_label, font=dict(size=label_size)),
          tickfont=dict(size=tick_size),
          showgrid=grid_on, gridcolor=grid_color,
          showbackground=axis_bg_on, backgroundcolor=axis_bg_color)
za = dict(title=dict(text=z_label, font=dict(size=label_size)),
          tickfont=dict(size=tick_size),
          showgrid=grid_on, gridcolor=grid_color,
          showbackground=axis_bg_on, backgroundcolor=axis_bg_color)

if manual_range:
    xa["range"] = [float(x_min), float(x_max)]
    ya["range"] = [float(y_min), float(y_max)]
    za["range"] = [float(z_min_fixed), float(z_max_fixed)]

scene = dict(
    xaxis=xa, yaxis=ya, zaxis=za,
    aspectmode=aspect_mode,
    camera=dict(eye=get_camera_eye(camera_view, (cam_x, cam_y, cam_z))),
    bgcolor=scene_bg
)

if aspect_mode == "manual":
    scene["aspectratio"] = dict(x=float(ar_x), y=float(ar_y), z=float(ar_z))

fig = go.Figure(
    data=data_traces,
    layout=dict(
        template=template,
        scene=scene,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=title_text, font=dict(size=title_size)),
        paper_bgcolor="white"
    )
)

# ======================================
# UI principal
# ======================================
top_left, top_right = st.columns([3, 2])
with top_left:
    st.plotly_chart(fig, use_container_width=True, config={
        "displaylogo": False,
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "grafico",
            "scale": 2,
            "width": 1600,
            "height": 900
        }
    })
with top_right:
    st.markdown("### 📤 Exportação (sem Kaleido)")
    plotly_download_button(
        fig,
        filename=f"grafico.{fmt.lower()}",
        fmt=fmt.lower(),
        width=int(export_w),
        height=int(export_h),
        scale=float(export_scale)
    )
    st.caption("Dica: Para **alta resolução**, use 4K/8K ou aumente a **escala**. Para nitidez perfeita em qualquer zoom, use **SVG**.")

# Prévia dos dados
with st.expander("👀 Pré-visualização dos dados"):
    if data_source == "Arquivo" and df_preview is not None:
        st.dataframe(df_preview.head(100), use_container_width=True)
    else:
        st.write("Dados aleatórios gerados com os parâmetros acima.")



