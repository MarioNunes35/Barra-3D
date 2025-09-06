import os, sys, subprocess, shutil
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import base64
import io

st.set_page_config(page_title="3D Bar com Desvio-Padrão", layout="wide")
st.title("3D Bar com Desvio-Padrão")

# ===================== Sidebar =====================
st.sidebar.header("Configurações")
colorscale_name = st.sidebar.selectbox(
    "Cores (colorscale)",
    ["Turbo", "Viridis", "Cividis", "Portland", "Jet", "Bluered", "Plasma", "Magma", "Inferno"],
    index=0,
)
bar_width = st.sidebar.slider("Largura das colunas", 0.2, 0.9, 0.6, 0.05)
bar_opacity = st.sidebar.slider("Opacidade das colunas", 0.1, 1.0, 0.85, 0.05)

show_error = st.sidebar.checkbox("Mostrar desvio-padrão", value=True)
symmetric_err = st.sidebar.checkbox("Erro simétrico (±DP)", value=True)
err_color = st.sidebar.color_picker("Cor do erro", "#FF8C00")
err_width = st.sidebar.slider("Espessura das linhas de erro", 1, 8, 4, 1)
cap_size = st.sidebar.slider("Tamanho da tampa (cap)", 0.15, 1.0, 0.5, 0.05)
show_bottom_cap = st.sidebar.checkbox("Mostrar tampa inferior", value=False)

camera_preset = st.sidebar.selectbox(
    "Ângulo da câmera", ["Padrão", "Topo", "Frente", "Isométrico"], index=0
)

# Opções de exportação
st.sidebar.subheader("Exportação")
export_width = st.sidebar.slider("Largura da imagem (px)", 800, 3000, 1200, 100)
export_height = st.sidebar.slider("Altura da imagem (px)", 600, 2000, 800, 100)
export_scale = st.sidebar.slider("Escala de qualidade", 1, 6, 2, 1)

# ===================== Dados =====================
st.subheader("Dados")
up = st.file_uploader(
    "CSV com colunas: x, y, value, std (std opcional).", type=["csv"]
)

if up is None:
    st.info("Sem arquivo? Usando exemplo embutido.")
    df = pd.DataFrame({
        "x": ["1887–1896","1887–1896","1887–1896",
              "1937–1946","1937–1946","1937–1946",
              "1987–1996","1987–1996","1987–1996"],
        "y": ["Winter","Spring","Summer",
              "Winter","Spring","Summer",
              "Winter","Spring","Summer"],
        "value": [4.5,12.0,26.0, 6.0,15.5,23.0, 7.0,18.5,21.5],
        "std":   [1.8, 2.2, 1.5, 2.5, 2.0, 1.8, 1.2, 2.3, 2.1],
    })
else:
    df = pd.read_csv(up)

# Padroniza nomes
df = df.rename(columns={c: c.lower() for c in df.columns})
required = {"x", "y", "value"}
if not required.issubset(set(df.columns)):
    st.error("O CSV deve conter colunas: x, y, value (e opcionalmente std).")
    st.stop()
if "std" not in df.columns:
    df["std"] = 0.0

# Ordem de categorias preservando a 1ª ocorrência
def ordered_unique(seq):
    seen, out = set(), []
    for v in seq:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

x_cats = ordered_unique(df["x"])
y_cats = ordered_unique(df["y"])
x_map = {c:i for i,c in enumerate(x_cats)}
y_map = {c:i for i,c in enumerate(y_cats)}
df["_xi"] = df["x"].map(x_map)
df["_yi"] = df["y"].map(y_map)

vmin, vmax = float(df["value"].min()), float(df["value"].max())
if np.isclose(vmin, vmax):
    vmax = vmin + 1e-9

# ===================== Helpers =====================
def color_for_value(val, scale_name, vmin, vmax):
    t = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    return sample_colorscale(scale_name, [min(max(t, 0.0), 1.0)])[0]

def bar_mesh3d(xc, yc, h, w, color_hex, hovertext=None):
    """Paralelepípedo centrado em (xc, yc) com altura h e largura w."""
    x0, x1 = xc - w/2, xc + w/2
    y0, y1 = yc - w/2, yc + w/2
    z0, z1 = 0.0, float(h)

    xs = [x0,x1,x1,x0, x0,x1,x1,x0]
    ys = [y0,y0,y1,y1, y0,y0,y1,y1]
    zs = [z0,z0,z0,z0, z1,z1,z1,z1]

    faces = [
        (0,1,2),(0,2,3), (4,5,6),(4,6,7),
        (0,1,5),(0,5,4), (2,3,7),(2,7,6),
        (0,3,7),(0,7,4), (1,2,6),(1,6,5),
    ]
    i, j, k = zip(*faces)

    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color_hex, opacity=bar_opacity,
        flatshading=True,
        hovertext=hovertext, hovertemplate="%{hovertext}<extra></extra>",
        showscale=False,
    )

def build_error_traces(rows, cap_len, color_hex, width, symmetric=True, show_bottom=False):
    """Linhas verticais + tampas (Scatter3d) para o desvio-padrão."""
    xs, ys, zs = [], [], []
    xcap1, ycap1, zcap1 = [], [], []
    xcap2, ycap2, zcap2 = [], [], []
    for _, r in rows.iterrows():
        x, y = float(r["_xi"]), float(r["_yi"])
        h, sd = float(r["value"]), float(r["std"])
        if sd <= 0:
            continue
        z_top = h + sd
        z_bot = max(0.0, h - sd) if symmetric else h

        # vertical
        xs += [x, x, None]
        ys += [y, y, None]
        zs += [z_bot, z_top, None]

        # caps topo (duas direções)
        xcap1 += [x - cap_len/2, x + cap_len/2, None]
        ycap1 += [y, y, None]
        zcap1 += [z_top, z_top, None]

        xcap2 += [x, x, None]
        ycap2 += [y - cap_len/2, y + cap_len/2, None]
        zcap2 += [z_top, z_top, None]

        if show_bottom:
            xcap1 += [x - cap_len/2, x + cap_len/2, None]
            ycap1 += [y, y, None]
            zcap1 += [z_bot, z_bot, None]

            xcap2 += [x, x, None]
            ycap2 += [y - cap_len/2, y + cap_len/2, None]
            zcap2 += [z_bot, z_bot, None]

    line = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color_hex, width=width),
                        showlegend=False, hoverinfo="skip")
    capx = go.Scatter3d(x=xcap1, y=ycap1, z=zcap1, mode="lines",
                        line=dict(color=color_hex, width=width),
                        showlegend=False, hoverinfo="skip")
    capy = go.Scatter3d(x=xcap2, y=ycap2, z=zcap2, mode="lines",
                        line=dict(color=color_hex, width=width),
                        showlegend=False, hoverinfo="skip")
    return [line, capx, capy]

# ===================== Figura =====================
fig = go.Figure()

# Barras
for _, r in df.iterrows():
    xi, yi = float(r["_xi"]), float(r["_yi"])
    val, sd = float(r["value"]), float(r["std"])
    col = color_for_value(val, colorscale_name, vmin, vmax)
    htxt = f"X: {r['x']}<br>Y: {r['y']}<br>Valor: {val:.3f}<br>DP: {sd:.3f}"
    fig.add_trace(bar_mesh3d(xi, yi, val, bar_width, col, hovertext=htxt))

# Colorbar (via marcador invisível)
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers",
    marker=dict(
        size=1, color=[vmin], colorscale=colorscale_name,
        cmin=vmin, cmax=vmax, showscale=True,
        colorbar=dict(title="Valor", thickness=18, len=0.85)
    ),
    opacity=0, showlegend=False, hoverinfo="skip"
))

# Erro
if show_error:
    for tr in build_error_traces(df, cap_size, err_color, err_width,
                                 symmetric=symmetric_err, show_bottom=show_bottom_cap):
        fig.add_trace(tr)

# Eixos e layout
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title="Time Period",
            tickmode="array",
            tickvals=list(range(len(x_cats))),
            ticktext=x_cats,
            tickangle=0,
            backgroundcolor="rgb(240,255,240)",
        ),
        yaxis=dict(
            title="Season",
            tickmode="array",
            tickvals=list(range(len(y_cats))),
            ticktext=y_cats,
            backgroundcolor="rgb(240,255,240)",
        ),
        zaxis=dict(title="Valor"),
        aspectmode="cube",
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    title="3D Bar com Desvio-Padrão",
    width=export_width,
    height=export_height,
)

# Câmera
if camera_preset == "Padrão":
    cam = dict(eye=dict(x=1.8, y=1.8, z=0.9))
elif camera_preset == "Topo":
    cam = dict(eye=dict(x=0.01, y=0.01, z=3.0))
elif camera_preset == "Frente":
    cam = dict(eye=dict(x=0.01, y=2.8, z=0.8))
else:  # Isométrico
    cam = dict(eye=dict(x=1.4, y=1.4, z=1.4))
fig.update_layout(scene_camera=cam)

# ===================== Render =====================
st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "grafico_3d",
            "scale": export_scale,
            "width": export_width,
            "height": export_height,
        }
    },
)

# ===================== Exportação Melhorada =====================
st.subheader("Exportar Gráfico")

col1, col2, col3 = st.columns(3)

with col1:
    # HTML interativo sempre funciona
    html_content = fig.to_html(
        include_plotlyjs="cdn", 
        full_html=True,
        config={
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "grafico_3d",
                "scale": export_scale,
                "width": export_width,
                "height": export_height,
            }
        }
    )
    st.download_button(
        "📄 Baixar HTML Interativo", 
        data=html_content,
        file_name="grafico_3d.html", 
        mime="text/html"
    )

with col2:
    # Método alternativo para PNG usando plotly.io
    try:
        import plotly.io as pio
        
        # Tenta usar kaleido primeiro, depois outros engines
        png_bytes = None
        engines = ['kaleido', 'orca']
        
        for engine in engines:
            try:
                pio.kaleido.scope.default_format = "png"
                png_bytes = fig.to_image(
                    format="png", 
                    width=export_width, 
                    height=export_height, 
                    scale=export_scale,
                    engine=engine
                )
                break
            except Exception as e:
                continue
        
        if png_bytes:
            st.download_button(
                "🖼️ Baixar PNG (Kaleido)", 
                data=png_bytes,
                file_name="grafico_3d.png", 
                mime="image/png"
            )
        else:
            st.warning("Engine Kaleido não disponível")
            
    except ImportError:
        st.warning("Instale: pip install plotly[kaleido]")

with col3:
    # SVG como alternativa
    try:
        svg_bytes = fig.to_image(
            format="svg", 
            width=export_width, 
            height=export_height
        )
        st.download_button(
            "🎨 Baixar SVG", 
            data=svg_bytes,
            file_name="grafico_3d.svg", 
            mime="image/svg+xml"
        )
    except Exception:
        st.info("SVG não disponível")

# ===================== Instruções de Instalação =====================
with st.expander("🔧 Solução para problemas de exportação"):
    st.markdown("""
    ### Se a exportação PNG não funcionar:
    
    **1. Instale as dependências necessárias:**
    ```bash
    pip install plotly[kaleido]
    # ou
    pip install kaleido
    ```
    
    **2. Para ambientes Docker/Cloud:**
    ```dockerfile
    RUN apt-get update && apt-get install -y \\
        chromium-browser \\
        chromium-chromedriver
    ENV CHROME_BIN=/usr/bin/chromium-browser
    ```
    
    **3. Alternativas que sempre funcionam:**
    - Use o **botão da câmera** no próprio gráfico (canto superior direito)
    - Baixe o **HTML interativo** e abra no navegador
    - Use **Ctrl+P** ou **Cmd+P** no HTML para salvar como PDF
    
    **4. Para Streamlit Cloud:**
    Adicione ao `requirements.txt`:
    ```
    plotly
    kaleido
    ```
    
    E ao `packages.txt`:
    ```
    chromium-browser
    chromium-chromedriver
    ```
    """)

# ===================== Preview dos dados =====================
with st.expander("📊 Visualizar dados carregados"):
    st.dataframe(df)
    
    # Estatísticas básicas
    st.subheader("Estatísticas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de pontos", len(df))
    with col2:
        st.metric("Valor médio", f"{df['value'].mean():.2f}")
    with col3:
        st.metric("Desvio padrão médio", f"{df['std'].mean():.2f}")

