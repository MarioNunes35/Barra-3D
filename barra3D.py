# -*- coding: utf-8 -*-
"""
Streamlit — 3D Bar Chart (Advanced)
Author: ChatGPT
Date: 2025-09-13

Recursos principais:
- Upload de CSV/Excel e editor de dados integrado
- Mapeamento de colunas (X, Y, Z, erros)
- Barras 3D com Matplotlib (sem dependência de Chrome/Kaleido)
- Barras com coloração fixa, por série (Y) ou por altura (colormap)
- Barras com largura/profundidade/espessura da borda ajustáveis
- Rótulos de valores no topo das barras (formatação customizável)
- Erros simétricos ou assimétricos renderizados como “hastes” 3D com “chapéus”
- Controle de câmera (elevação/azimute), tema claro/escuro, grade, fundo, limites de eixos
- Exportação PNG (com DPI) e SVG
"""
import io
from typing import List, Tuple, Optional

import streamlit as st
import numpy as np
import pandas as pd

# Força backend sem display (necessário no Streamlit Cloud)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from matplotlib import cm


# -----------------------------
# Utilidades
# -----------------------------
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


def make_facecolors(
    df: pd.DataFrame,
    y_order: List[str],
    z_vals: np.ndarray,
    color_mode: str,
    base_color: str,
    colormap_name: str,
    series_palette_name: str,
) -> List:
    """Define as cores de cada barra conforme o modo de cor selecionado."""
    colors = []

    if color_mode == "Única":
        colors = [base_color] * len(df)

    elif color_mode == "Por série (Y)":
        # Mapa discreto por categoria Y
        cmap = cm.get_cmap(series_palette_name, len(y_order))
        y_to_idx = {y: i for i, y in enumerate(y_order)}
        for y in df["__Y__"]:
            colors.append(cmap(y_to_idx[y]))

    else:  # "Por altura (colormap)"
        norm = Normalize(vmin=float(np.nanmin(z_vals)), vmax=float(np.nanmax(z_vals)))
        cmap = cm.get_cmap(colormap_name)
        for z in z_vals:
            colors.append(cmap(norm(z)))

    return colors


def draw_error_segments(ax, segments: List[Tuple[float, float, float, float]], cap: float, color="black", lw=1.0):
    """Desenha segmentos de erro (linha vertical com 'chapéu')."""
    for (x, y, z0, z1) in segments:
        # Haste vertical
        ax.plot([x, x], [y, y], [z0, z1], color=color, linewidth=lw)
        # Chapéu superior
        ax.plot([x - cap, x + cap], [y, y], [z1, z1], color=color, linewidth=lw)
        # Chapéu inferior
        ax.plot([x - cap, x + cap], [y, y], [z0, z0], color=color, linewidth=lw)


def render_3d_bars(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    err_style: str = "Nenhum",   # Nenhum | Simétrico | Assimétrico
    err_col: Optional[str] = None,
    err_low_col: Optional[str] = None,
    err_high_col: Optional[str] = None,
    elev: float = 20.0,
    azim: float = -60.0,
    bar_width: float = 0.6,
    bar_depth: float = 0.6,
    alpha: float = 0.95,
    edge_on: bool = True,
    edge_width: float = 0.6,
    edge_color: str = "black",
    color_mode: str = "Única",   # Única | Por série (Y) | Por altura (colormap)
    base_color: str = "#3182bd",
    colormap_name: str = "viridis",
    series_palette_name: str = "tab10",
    show_values: bool = True,
    value_fmt: str = "{:.2f}",
    value_offset: float = 0.02,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    show_grid: bool = True,
    bg_color: str = "white",
    pane_color: str = "#f5f5f5",
    title: str = "Gráfico de Barras 3D",
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    label_size: int = 12,
    tick_size: int = 10,
    title_size: int = 16,
    cap_ratio: float = 0.6,  # largura do “chapéu” de erro em relação à largura da barra
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Desenha o gráfico 3D de barras e retorna (fig, ax).
    O DataFrame pode ter X e Y categóricos. Z deve ser numérico.
    """
    df = df.copy()

    # Ordena categorias X/Y na ordem de aparição
    x_order = list(pd.unique(df[x_col]))
    y_order = list(pd.unique(df[y_col]))

    df["__X__"] = pd.Categorical(df[x_col], categories=x_order, ordered=True)
    df["__Y__"] = pd.Categorical(df[y_col], categories=y_order, ordered=True)
    df["__Z__"] = to_numeric_safe(df[z_col])

    # Filtra linhas com Z válido
    df = df[~df["__Z__"].isna()].reset_index(drop=True)

    # Coordenadas numéricas para as barras
    xs = df["__X__"].cat.codes.to_numpy(dtype=float)
    ys = df["__Y__"].cat.codes.to_numpy(dtype=float)
    zs = np.zeros_like(xs, dtype=float)  # base das barras no plano z=0
    dz = df["__Z__"].to_numpy(dtype=float)

    dx = np.full_like(xs, fill_value=float(bar_width))
    dy = np.full_like(xs, fill_value=float(bar_depth))

    # Cores
    facecolors = make_facecolors(df, y_order, dz, color_mode, base_color, colormap_name, series_palette_name)

    # Figure/Axes
    fig = plt.figure(figsize=(10, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Fundo
    fig.patch.set_facecolor(bg_color)
    ax.xaxis.set_pane_color(matplotlib.colors.to_rgba(pane_color, 1.0))
    ax.yaxis.set_pane_color(matplotlib.colors.to_rgba(pane_color, 1.0))
    ax.zaxis.set_pane_color(matplotlib.colors.to_rgba(pane_color, 1.0))

    # Barras
    ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True,
             color=facecolors,
             edgecolor=edge_color if edge_on else None,
             linewidth=edge_width,
             alpha=alpha)

    # Rótulos/ticks
    ax.set_title(title, fontsize=title_size, pad=12)
    ax.set_xlabel(xlabel, fontsize=label_size, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=label_size, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=label_size, labelpad=10)

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, fontsize=tick_size, rotation=0, ha="center")
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order, fontsize=tick_size, rotation=0, ha="center")

    # Limites Z
    if zmin is not None or zmax is not None:
        current = ax.get_zlim()
        z0 = zmin if zmin is not None else current[0]
        z1 = zmax if zmax is not None else current[1]
        if z0 == z1:
            z1 = z0 + 1.0
        ax.set_zlim(z0, z1)

    # Grid
    ax.grid(show_grid)

    # Câmera
    ax.view_init(elev=elev, azim=azim)

    # Erros
    segments = []
    if err_style == "Simétrico" and err_col and (err_col in df.columns):
        e = to_numeric_safe(df[err_col]).to_numpy(dtype=float)
        for x, y, z, ee, w in zip(xs, ys, dz, e, dx):
            if not np.isnan(ee):
                z0, z1 = z - ee, z + ee
                cap = (w * cap_ratio) / 2.0
                segments.append((x + w/2.0, y + (bar_depth/2.0), z0, z1))

    elif err_style == "Assimétrico" and (err_low_col in df.columns) and (err_high_col in df.columns):
        low = to_numeric_safe(df[err_low_col]).to_numpy(dtype=float)
        high = to_numeric_safe(df[err_high_col]).to_numpy(dtype=float)
        for x, y, z, lo, hi, w in zip(xs, ys, dz, low, high, dx):
            if not (np.isnan(lo) or np.isnan(hi)):
                z0, z1 = z - lo, z + hi
                cap = (w * cap_ratio) / 2.0
                segments.append((x + w/2.0, y + (bar_depth/2.0), z0, z1))

    if segments:
        cap = (float(bar_width) * cap_ratio) / 2.0
        draw_error_segments(ax, segments, cap=cap, color=edge_color, lw=max(1.0, edge_width))

    # Valores no topo
    if show_values:
        for x, y, z, w, d in zip(xs, ys, dz, dx, dy):
            x_center = x + w/2.0
            y_center = y + d/2.0
            ax.text(x_center, y_center, z + value_offset, value_fmt.format(z),
                    ha="center", va="bottom", fontsize=max(8, tick_size))

    return fig, ax


def main():
    st.set_page_config(page_title="Barras 3D Avançado", layout="wide")
    
    # ===== CONTROLES MOVIDOS PARA A BARRA LATERAL =====
    st.sidebar.title("⚙️ Controles")

    # Mapeamento de colunas
    st.sidebar.subheader("Mapeamento de Colunas")
    
    # Dados serão carregados na página principal, mas o mapeamento fica na sidebar
    # para que os controles apareçam antes do gráfico.
    
    # Carregamento de dados
    st.sidebar.subheader("Dados")
    up = st.sidebar.file_uploader("CSV ou Excel", type=["csv", "xlsx", "xls"])
    sep = st.sidebar.text_input("Separador (CSV)", value=",", help="Ignorado para Excel.", key="sep")
    use_editor = st.sidebar.checkbox("Editar dados após carregar", value=False)

    df = None
    if up:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up, sep=sep)
            else:
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Falha ao ler arquivo: {e}")
            df = None

    df = ensure_min_columns(df)
    
    # Mapeamento
    cols = list(df.columns)
    x_col = st.sidebar.selectbox("Coluna X (categoria ou número)", options=cols, index=0)
    y_col = st.sidebar.selectbox("Coluna Y (categoria ou número)", options=cols, index=1 if len(cols) > 1 else 0)
    z_col = st.sidebar.selectbox("Coluna Z (altura)", options=cols, index=2 if len(cols) > 2 else 0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Erros (opcional)")
    err_style = st.sidebar.radio("Tipo de erro", options=["Nenhum", "Simétrico", "Assimétrico"], horizontal=True, index=0)
    err_col = err_low_col = err_high_col = None
    if err_style == "Simétrico":
        err_col = st.sidebar.selectbox("Coluna de erro (±)", options=["—"] + cols, index=0)
        if err_col == "—":
            err_style = "Nenhum"
            err_col = None
    elif err_style == "Assimétrico":
        err_low_col = st.sidebar.selectbox("Coluna de erro -", options=["—"] + cols, index=0)
        err_high_col = st.sidebar.selectbox("Coluna de erro +", options=["—"] + cols, index=0)
        if err_low_col == "—" or err_high_col == "—":
            err_style = "Nenhum"
            err_low_col = err_high_col = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("Cores e Aparência")
    bar_width = st.sidebar.number_input("Largura (X) da barra", 0.05, 1.0, 0.6, step=0.05)
    bar_depth = st.sidebar.number_input("Profundidade (Y) da barra", 0.05, 1.0, 0.6, step=0.05)
    alpha = st.sidebar.slider("Transparência (alpha)", 0.1, 1.0, 0.95, step=0.05)
    
    edge_on = st.sidebar.checkbox("Mostrar borda", value=True)
    edge_width = st.sidebar.number_input("Espessura da borda", 0.0, 5.0, 0.6, step=0.1)
    edge_color = st.sidebar.color_picker("Cor da borda", value="#000000")
    
    color_mode = st.sidebar.selectbox("Modo de cor", ["Única", "Por série (Y)", "Por altura (colormap)"], index=0)

    if color_mode == "Única":
        base_color = st.sidebar.color_picker("Cor base", value="#3182bd")
    else:
        base_color = "#3182bd"
    
    if color_mode == "Por altura (colormap)":
        colormap_name = st.sidebar.selectbox("Colormap", sorted([m for m in plt.colormaps() if not str(m).endswith('_r')]), index=sorted([m for m in plt.colormaps() if not str(m).endswith('_r')]).index("viridis"))
    else:
        colormap_name = "viridis"
        
    if color_mode == "Por série (Y)":
        series_palette_name = st.sidebar.selectbox("Paleta de séries", options=["tab10", "tab20", "Set3", "Pastel1", "Accent"], index=0)
    else:
        series_palette_name = "tab10"
        
    show_values = st.sidebar.checkbox("Mostrar valores no topo", value=True)
    value_fmt = st.sidebar.text_input("Formato do valor", value="{:.2f}")
    value_offset = st.sidebar.number_input("Offset em Z dos valores", 0.0, 1.0, 0.02, step=0.01)

    st.sidebar.subheader("Rótulos, Limites e Câmera")
    title = st.sidebar.text_input("Título", value="Gráfico de Barras 3D")
    xlabel = st.sidebar.text_input("Rótulo eixo X", value="X")
    ylabel = st.sidebar.text_input("Rótulo eixo Y", value="Y")
    zlabel = st.sidebar.text_input("Rótulo eixo Z", value="Z")
    
    label_size = st.sidebar.number_input("Tamanho rótulos (label)", 6, 30, 12, step=1)
    tick_size = st.sidebar.number_input("Tamanho dos ticks", 6, 30, 10, step=1)
    title_size = st.sidebar.number_input("Tamanho do título", 8, 40, 16, step=1)

    elev = st.sidebar.slider("Elevação da câmera", -10, 90, 20, step=1)
    azim = st.sidebar.slider("Azimute da câmera", -180, 180, -60, step=5)
    show_grid = st.sidebar.checkbox("Mostrar grade", value=True)

    zmin = st.sidebar.number_input("Z mínimo (opcional)", value=0.0, step=1.0)
    use_zmin = st.sidebar.checkbox("Fixar Zmín", value=False)
    zmax = st.sidebar.number_input("Z máximo (opcional)", value=100.0, step=1.0)
    use_zmax = st.sidebar.checkbox("Fixar Zmáx", value=False)
    
    bg_color = st.sidebar.color_picker("Cor de fundo da figura", value="#ffffff")
    pane_color = st.sidebar.color_picker("Cor do plano 3D", value="#f5f5f5")
    cap_ratio = st.sidebar.slider("Largura do 'chapéu' de erro (fração da barra)", 0.1, 1.0, 0.6, step=0.05)


    # ===== ÁREA PRINCIPAL DA PÁGINA =====
    st.title("📊 Barras 3D — Avançado (Matplotlib)")
    with st.expander("ℹ️ Como usar", expanded=False):
        st.markdown(
            """
            1) Use os controles na barra lateral à esquerda para carregar seus dados e customizar o gráfico.
            2) O gráfico será atualizado automaticamente na área principal.
            3) Exporte como PNG (com DPI) ou SVG.
            """
        )

    if use_editor:
        st.write("Editor de Dados:")
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    else:
        st.write("Prévia do DataFrame:")
        st.dataframe(df, use_container_width=True)

    # Render
    fig, ax = render_3d_bars(
        df=df,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        err_style=err_style,
        err_col=err_col if err_style == "Simétrico" else None,
        err_low_col=err_low_col if err_style == "Assimétrico" else None,
        err_high_col=err_high_col if err_style == "Assimétrico" else None,
        elev=float(elev),
        azim=float(azim),
        bar_width=float(bar_width),
        bar_depth=float(bar_depth),
        alpha=float(alpha),
        edge_on=bool(edge_on),
        edge_width=float(edge_width),
        edge_color=str(edge_color),
        color_mode=str(color_mode),
        base_color=str(base_color),
        colormap_name=str(colormap_name),
        series_palette_name=str(series_palette_name),
        show_values=bool(show_values),
        value_fmt=str(value_fmt),
        value_offset=float(value_offset),
        zmin=float(zmin) if use_zmin else None,
        zmax=float(zmax) if use_zmax else None,
        show_grid=bool(show_grid),
        bg_color=str(bg_color),
        pane_color=str(pane_color),
        title=str(title),
        xlabel=str(xlabel),
        ylabel=str(ylabel),
        zlabel=str(zlabel),
        label_size=int(label_size),
        tick_size=int(tick_size),
        title_size=int(title_size),
        cap_ratio=float(cap_ratio),
    )

    st.pyplot(fig, use_container_width=True)

    # Exportação
    with st.expander("💾 Exportar figura"):
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            fmt = st.selectbox("Formato", options=["png", "svg"], index=0)
        with ex2:
            dpi = st.slider("DPI (apenas PNG)", 72, 600, 300, step=10)
        with ex3:
            file_name = st.text_input("Nome do arquivo (sem extensão)", value="barras3d")

        b = io.BytesIO()
        if fmt == "png":
            fig.savefig(b, format="png", dpi=int(dpi), bbox_inches="tight", facecolor=fig.get_facecolor())
            mime = "image/png"
            ext = "png"
        else:
            fig.savefig(b, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
            mime = "image/svg+xml"
            ext = "svg"
        b.seek(0)
        st.download_button(
            label=f"⬇️ Baixar {ext.upper()}",
            data=b,
            file_name=f"{file_name}.{ext}",
            mime=mime,
            use_container_width=True
        )

    st.caption("Dica: no Streamlit Cloud, inclua no requirements.txt: streamlit, numpy, pandas, matplotlib.")

if __name__ == "__main__":
    main()




