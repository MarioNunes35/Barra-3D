import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import get_colorscale, sample_colorscale

st.set_page_config(page_title="3D Bar com Desvio Padrão", layout="wide")

st.title("Gráfico 3D de Colunas com Desvio-Padrão (Streamlit + Plotly)")

# ======== SIDEBAR ========
st.sidebar.header("Configurações")
colorscale_name = st.sidebar.selectbox(
    "Cores (colorscale)", 
    ["Turbo","Viridis","Cividis","Portland","Jet","Bluered","Plasma","Magma","Inferno"],
    index=0
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
    "Ângulo da câmera",
    ["Padrão","Topo","Frente","Isométrico"],
    index=0
)

png_scale = st.sidebar.slider("Escala PNG (alta resolução)", 1, 6, 3, 1)

# ======== DADOS ========
st.subheader("Dados")
up = st.file_uploader("Carregue um CSV com colunas: x, y, value, std (std opcional).", type=["csv"])

if up is None:
    st.info("Sem arquivo? Usando exemplo embutido.")
    df = pd.DataFrame({
        "x": ["1887–1896","1887–1896","1887–1896","1937–1946","1937–1946","1937–1946","1987–1996","1987–1996","1987–1996"],
        "y": ["Winter","Spring","Summer","Winter","Spring","Summer","Winter","Spring","Summer"],
        "value": [4.5,12.0,26.0,6.0,15.5,23.0,7.0,18.5,21.5],
        "std":   [1.8, 2.2, 1.5, 2.5, 2.0, 1.8, 1.2, 2.3, 2.1],
    })
else:
    df = pd.read_csv(up)

# Normalizar nomes esperados
expected = {"x","y","value"}
if not expected.issubset({c.lower() for c in df.columns}):
    st.error("O CSV deve conter colunas: x, y, value (e opcionalmente std).")
    st.stop()

# Padronizar keys
df = df.rename(columns={c: c.lower() for c in df.columns})
if "std" not in df.columns:
    df["std"] = 0.0

# Preservar ordem de aparição das categorias
def ordered_unique(series):
    seen = set(); out=[]
    for v in series:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

x_cats = ordered_unique(df["x"].tolist())
y_cats = ordered_unique(df["y"].tolist())

x_map = {cat:i for i,cat in enumerate(x_cats)}
y_map = {cat:i for i,cat in enumerate(y_cats)}

df["_xi"] = df["x"].map(x_map)
df["_yi"] = df["y"].map(y_map)

# Faixa de cores baseada no valor (para a colorbar)
vmin, vmax = float(df["value"].min()), float(df["value"].max())
if np.isclose(vmin, vmax):  # evita divisão por zero
    vmax = vmin + 1e-6

# ======== FUNÇÕES ========
def bar_mesh3d(xc, yc, h, w, color_hex, hovertext=None, cdata=None):
    """
    Constrói uma barra como paralelepípedo (Mesh3d) centrado em (xc, yc), altura h, largura w.
    color_hex: cor única da barra (ex.: "#RRGGBB").
    """
    x0, x1 = xc - w/2, xc + w/2
    y0, y1 = yc - w/2, yc + w/2
    z0, z1 = 0.0, float(h)

    # 8 vértices
    xs = [x0,x1,x1,x0, x0,x1,x1,x0]
    ys = [y0,y0,y1,y1, y0,y0,y1,y1]
    zs = [z0,z0,z0,z0, z1,z1,z1,z1]

    # 12 triângulos (faces)
    faces = [
        (0,1,2),(0,2,3),       # base
        (4,5,6),(4,6,7),       # topo
        (0,1,5),(0,5,4),       # frente (y-)
        (2,3,7),(2,7,6),       # trás   (y+)
        (0,3,7),(0,7,4),       # esquerda (x-)
        (1,2,6),(1,6,5),       # direita  (x+)
    ]
    i, j, k = zip(*faces)

    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color_hex, opacity=bar_opacity,
        hovertext=hovertext, hovertemplate="%{hovertext}<extra></extra>",
        showscale=False,  # colorbar separado
        flatshading=True
    )

def color_for_value(val, cs_name, vmin, vmax):
    # mapeia valor -> cor da colorscale
    t = (float(val)-vmin)/(vmax-vmin)
    return sample_colorscale(get_colorscale(cs_name), t)[0]  # hex

def build_error_traces(rows, cap_len, color_hex, width, symmetric=True, show_bottom=False):
    xs, ys, zs = [], [], []     # verticals
    xcap1, ycap1, zcap1 = [], [], []  # caps (eixo x)
    xcap2, ycap2, zcap2 = [], [], []  # caps (eixo y)

    for _, r in rows.iterrows():
        x, y = float(r["_xi"]), float(r["_yi"])
        h, sd = float(r["value"]), float(r["std"])
        if sd <= 0:  # nada a desenhar
            continue

        z_top = h + (sd if symmetric else sd)
        z_bot = max(0.0, h - sd) if symmetric else h  # não fica negativo

        # linha vertical
        xs += [x, x, None]
        ys += [y, y, None]
        zs += [z_bot, z_top, None]

        # caps no topo
        xcap1 += [x - cap_len/2, x + cap_len/2, None]
        ycap1 += [y, y, None]
        zcap1 += [z_top, z_top, None]

        xcap2 += [x, x, None]
        ycap2 += [y - cap_len/2, y + cap_len/2, None]
        zcap2 += [z_top, z_top, None]

        # caps na base (opcional)
        if show_bottom:
            xcap1 += [x - cap_len/2, x + cap_len/2, None]
            ycap1 += [y, y, None]
            zcap1 += [z_bot, z_bot, None]

            xcap2 += [x, x, None]
            ycap2 += [y - cap_len/2, y + cap_len/2, None]
            zcap2 += [z_bot, z_bot, None]

    line_trace = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color_hex, width=width),
        showlegend=False, hoverinfo="skip"
    )
    cap1_trace = go.Scatter3d(
        x=xcap1, y=ycap1, z=zcap1, mode="lines",
        line=dict(color=color_hex, width=width),
        showlegend=False, hoverinfo="skip"
    )
    cap2_trace = go.Scatter3d(
        x=xcap2, y=ycap2, z=zcap2, mode="lines",
        line=dict(color=color_hex, width=width),
        showlegend=False, hoverinfo="skip"
    )
    return [line_trace, cap1_trace, cap2_trace]

# ======== FIGURA ========
fig = go.Figure()

# barras (uma Mesh3d por barra) + colorbar "fake"
for _, r in df.iterrows():
    xi, yi = float(r["_xi"]), float(r["_yi"])
    val, sd = float(r["value"]), float(r["std"])
    color_hex = color_for_value(val, colorscale_name, vmin, vmax)
    htxt = f"X: {r['x']}<br>Y: {r['y']}<br>Valor: {val:.3f}<br>DP: {sd:.3f}"
    fig.add_trace(bar_mesh3d(xi, yi, val, bar_width, color_hex, hovertext=htxt))

# colorbar (marcador invisível só para exibir a escala)
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers",
    marker=dict(size=1, color=[vmin], colorscale=colorscale_name, cmin=vmin, cmax=vmax, showscale=True,
                colorbar=dict(title="Valor", thickness=18, len=0.8)),
    opacity=0, showlegend=False, hoverinfo="skip"
))

# barras de erro
if show_error:
    for tr in build_error_traces(df, cap_size, err_color, err_width, symmetric=symmetric_err, show_bottom=show_bottom_cap):
        fig.add_trace(tr)

# Eixos com rótulos categóricos (ticktext/tickvals)
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title="Time Period",
            tickmode="array",
            tickvals=list(range(len(x_cats))),
            ticktext=x_cats,
            backgroundcolor="rgb(240,255,240)"
        ),
        yaxis=dict(
            title="Season",
            tickmode="array",
            tickvals=list(range(len(y_cats))),
            ticktext=y_cats,
            backgroundcolor="rgb(240,255,240)"
        ),
        zaxis=dict(title="Valor"),
        aspectmode="cube"
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    title="3D Bar com Desvio-Padrão"
)

# câmera
if camera_preset == "Padrão":
    cam = dict(eye=dict(x=1.8, y=1.8, z=0.9))
elif camera_preset == "Topo":
    cam = dict(eye=dict(x=0.01, y=0.01, z=3.0))
elif camera_preset == "Frente":
    cam = dict(eye=dict(x=0.01, y=2.8, z=0.8))
else:  # Isométrico
    cam = dict(eye=dict(x=1.4, y=1.4, z=1.4))
fig.update_layout(scene_camera=cam)

st.plotly_chart(fig, use_container_width=True)

# ======== DOWNLOADS ========
st.subheader("Exportar")
col1, col2 = st.columns(2)

with col1:
    try:
        png_bytes = fig.to_image(format="png", scale=png_scale)
        st.download_button("⬇️ Baixar PNG (alta)", data=png_bytes, file_name="grafico_3d.png", mime="image/png")
    except Exception as e:
        st.warning(
            "Para PNG em alta, instale `plotly[kaleido]` e o Chrome com `plotly_get_chrome`. "
            "Como alternativa, use o download em HTML ao lado."
        )

with col2:
    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    st.download_button("⬇️ Baixar HTML interativo", data=html, file_name="grafico_3d.html", mime="text/html")
